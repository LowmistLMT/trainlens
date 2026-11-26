from __future__ import annotations

import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

SKILL = Path(__file__).resolve().parents[1] / "skills" / "trainlens"
DEMOS = SKILL / "assets" / "demo-logs"
sys.path.insert(0, str(SKILL / "scripts"))

import compare_runs  # noqa: E402


class CompareTests(unittest.TestCase):
    def compare(self, *args):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(compare_runs.main([str(arg) for arg in args]), 0)
        return json.loads(stdout.getvalue())["data"]

    @staticmethod
    def write_run(path, csv_rows, config=None):
        path.mkdir(parents=True)
        (path / "metrics.csv").write_text(csv_rows, encoding="utf-8")
        if config:
            (path / "config.json").write_text(json.dumps(config), encoding="utf-8")

    def test_hf_min_ranking_and_minimal_config(self):
        data = self.compare(DEMOS / "llm-sft", "--config-diff")
        self.assertEqual((data["primary_metric"], data["mode"]), ("eval_loss", "min"))
        rows = data["ranking"]
        self.assertEqual([r["run_id"] for r in rows], [
            "sft_lora_r32_best", "sft_baseline", "sft_lr5e4_nan",
        ])
        self.assertEqual([r["rank"] for r in rows], [1, 2, 3])
        self.assertEqual([r["delta_vs_best"] for r in rows], [0, 0.18, 0.94])
        self.assertEqual((rows[0]["framework"], rows[0]["step_max"]), ("hf", 600))
        self.assertEqual((rows[1]["best_value"], rows[1]["best_step"]), (1.12, 500))
        self.assertEqual(rows[1]["best_checkpoint"]["step"], 500)
        self.assertGreater(rows[1]["last_value"], rows[1]["best_value"])
        self.assertEqual(rows[2]["status"], "crashed")
        self.assertEqual(set(data["config_diff"]), {
            "learning_rate", "warmup_ratio", "lora_r", "lora_alpha", "weight_decay",
            "max_grad_norm", "fp16", "bf16",
        })
        self.assertEqual(data["config_diff"]["lora_r"]["sft_lora_r32_best"], 32)
        self.assertEqual(data["git_provenance"]["run_commits"], {
            r["run_id"]: r["git_commit"] for r in rows
        })

    def test_mmengine_max_missing_metric_and_per_class(self):
        data = self.compare(DEMOS / "mmseg-cityscapes", "--per-class")
        self.assertEqual((data["primary_metric"], data["mode"]), ("mIoU", "max"))
        winner, base, crashed = data["ranking"]
        self.assertEqual((winner["best_value"], winner["best_step"]), (78.65, 36000))
        self.assertEqual(base["delta_vs_best"], -3.83)
        self.assertEqual((crashed["rank"], crashed["best_value"], crashed["delta_vs_best"]),
                         (None, None, None))
        self.assertEqual(data["baseline_run_id"], base["run_id"])
        pole = data["per_class_diff"]["IoU.pole"]
        self.assertEqual(pole[base["run_id"]], {"value": 56.5, "delta_vs_base": 0})
        self.assertEqual(pole[winner["run_id"]], {"value": 62.9, "delta_vs_base": 6.4})
        self.assertNotIn(crashed["run_id"], pole)
        self.assertNotIn("config_diff", data)

    def test_explicit_baseline_and_mode_override(self):
        root = DEMOS / "mmseg-cityscapes"
        b2 = root / "segformer_b2_aug"
        b0 = root / "segformer_b0_base"
        data = self.compare(b2, b0, "--metric", "IoU.pole", "--mode", "min", "--per-class")
        self.assertEqual(data["baseline_run_id"], b2.name)
        self.assertEqual(data["per_class_diff"]["IoU.pole"][b0.name]["delta_vs_base"], -6.4)
        self.assertEqual(data["ranking"][0]["run_id"], b0.name)
        self.assertLess(data["ranking"][0]["best_value"], 56.5)

    def test_custom_metric_precision_ties_and_nonfinite_observations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, value in [("a", "0.90000001"), ("b", "0.90000002"),
                                ("c", "0.90000002"), ("d", "Inf")]:
                self.write_run(root / name, f"step,quality\n1,{value}\n2,NaN\n")
            data = self.compare(root)
        self.assertEqual(data["primary_metric"], "quality")
        self.assertNotIn("git_provenance", data)
        self.assertEqual([r["run_id"] for r in data["ranking"]], ["b", "c", "a", "d"])
        self.assertEqual([r["rank"] for r in data["ranking"]], [1, 1, 3, None])
        self.assertEqual(data["ranking"][0]["last_value"], "NaN")
        self.assertAlmostEqual(data["ranking"][2]["delta_vs_best"], -1e-8, places=12)

    def test_top_k_selects_from_every_discovered_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i in range(52):
                self.write_run(root / f"run-{i:02}", f"step,loss\n1,{52-i}\n")
            data = self.compare(root, "--top-k", 1)
        self.assertEqual([r["run_id"] for r in data["ranking"]], ["run-51"])

    def test_duplicate_names_and_overlapping_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "model-a" / "baseline"
            second = root / "model-b" / "baseline"
            self.write_run(first, "step,loss\n1,1\n", {"git_commit": "1234567", "lr": 0.01})
            self.write_run(second, "step,loss\n1,2\n", {"git_commit": "abcdef0", "lr": 0.02})
            data = self.compare(root, first, "--config-diff")
        self.assertEqual(len(data["ranking"]), 2)
        ids = [r["run_id"] for r in data["ranking"]]
        self.assertNotEqual(ids[0], ids[1])
        self.assertEqual(data["config_diff"], {"lr": dict(zip(ids, [0.01, 0.02]))})
        self.assertEqual(
            data["git_provenance"]["run_commits"], dict(zip(ids, ["1234567", "abcdef0"]))
        )

    @unittest.skipUnless(shutil.which("git"), "git executable required for local provenance")
    def test_local_git_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()

            def git(*args):
                return subprocess.check_output([
                    "git", "-C", str(repo), "-c", "user.name=TrainLens Test",
                    "-c", "user.email=test@example.invalid", "-c", "commit.gpgsign=false",
                    "-c", "core.hooksPath=/dev/null", *args,
                ], text=True).strip()

            git("init", "-q")
            settings = repo / "settings.txt"
            commits = []
            for i in range(2):
                settings.write_text(str(i), encoding="utf-8")
                git("add", "settings.txt")
                git("commit", "-qm", f"update {i}")
                commits.append(git("rev-parse", "--short", "HEAD"))
            root = Path(tmp) / "runs"
            for i, sha in enumerate(commits):
                self.write_run(root / str(i), f"step,loss\n1,{i}\n", {"git_commit": sha})
            data = self.compare(root, "--repo", repo)
            provenance = data["git_provenance"]
            self.assertEqual(provenance["run_commits"], {"0": commits[0], "1": commits[1]})
            self.assertEqual(provenance["repo_head"], commits[-1])
            self.assertFalse(provenance["repo_dirty"])
            self.assertEqual(provenance["commit_diff_files"], {
                "from": commits[0], "to": commits[1], "files": ["M\tsettings.txt"],
            })
            settings.write_text("dirty", encoding="utf-8")
            self.assertTrue(self.compare(root, "--repo", repo)["git_provenance"]["repo_dirty"])


if __name__ == "__main__":
    unittest.main()
