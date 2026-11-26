from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


SKILL = Path(__file__).resolve().parents[1] / "skills" / "trainlens"
SCRIPTS = SKILL / "scripts"
DEMOS = SKILL / "assets" / "demo-logs"
TOOLS = ("scan_runs", "compare_runs", "diagnose")


def snapshot(root: Path) -> dict:
    """Capture names, types, contents and mtimes, including the experiment directories."""
    result = {}
    for path in (root, *root.rglob("*")):
        stat = path.stat()
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        result[str(path.relative_to(root))] = (stat.st_mode, stat.st_mtime_ns, digest)
    return result


def strict_json(output: str) -> dict:
    def reject_constant(token: str):
        raise ValueError(f"Invalid JSON number {token}")

    return json.loads(output, parse_constant=reject_constant)


class CliTests(unittest.TestCase):
    def test_missing_arguments_report_a_compact_error_for_each_cli(self):
        for tool in (*TOOLS, "ssh_fetch", "render_report"):
            with self.subTest(tool=tool):
                proc = subprocess.run(
                    [sys.executable, "-E", "-S", str(SCRIPTS / f"{tool}.py")],
                    capture_output=True, text=True, timeout=10,
                )
                self.assertEqual(proc.returncode, 2, proc.stderr)
                self.assertEqual(len(proc.stdout.splitlines()), 1)
                result = strict_json(proc.stdout)
                self.assertEqual(result["tool"], tool)
                self.assertIs(result["ok"], False)
                self.assertTrue(result["summary"])

    def invoke(self, tool, path, *flags, scripts=SCRIPTS):
        # Ignore Python environment overrides and exclude site-packages, including on 3.9.
        proc = subprocess.run(
            [sys.executable, "-E", "-S", str(scripts / f"{tool}.py"), str(path), *flags],
            cwd=path, text=True, capture_output=True, timeout=20, check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return proc.stdout

    def test_both_demo_suites_in_json_and_markdown_are_read_only(self):
        before = snapshot(DEMOS)
        for suite, metric, n_findings in (("llm-sft", "eval_loss", 3), ("mmseg-cityscapes", "mIoU", 1)):
            run_dir = DEMOS / suite
            expected_ids = {path.name for path in run_dir.iterdir() if path.is_dir()}
            for tool in TOOLS:
                with self.subTest(suite=suite, tool=tool):
                    flags = ("--config-diff", "--per-class") if tool == "compare_runs" else ()
                    output = self.invoke(tool, run_dir, *flags)
                    payload = strict_json(output)
                    self.assertEqual(len(output.splitlines()), 1)
                    self.assertIs(payload["ok"], True)
                    self.assertEqual(payload["tool"], tool)
                    self.assertEqual(payload["schema_version"], "1.0")
                    self.assertEqual(payload["warnings"], [])
                    data = payload["data"]
                    if tool == "scan_runs":
                        self.assertLess(len(output), 4000)
                        self.assertEqual({r["run_id"] for r in data["runs"]}, expected_ids)
                    elif tool == "compare_runs":
                        self.assertEqual(data["primary_metric"], metric)
                        self.assertEqual({r["run_id"] for r in data["ranking"]}, expected_ids)
                    else:
                        self.assertEqual(len(data["findings"]), n_findings)
                    markdown = self.invoke(tool, run_dir, *flags, "--format", "md")
                    self.assertIn(payload["summary"], markdown)
                    self.assertIn("| ---", markdown)
                    if tool == "compare_runs":
                        self.assertIn("Config Diff", markdown)
                        if suite == "mmseg-cityscapes":
                            self.assertIn("IoU.road", markdown)
                    if tool == "diagnose":
                        for finding in data["findings"]:
                            self.assertIn(finding["code"], markdown)
        self.assertEqual(before, snapshot(DEMOS))

    def test_pretty_json_includes_strictly_valid_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "trainer_state.json").write_text(json.dumps({
                "log_history": [{"step": 5, "loss": "NaN"}],
                "best_model_checkpoint": "checkpoint-5", "best_metric": "Inf",
            }), encoding="utf-8")
            for tool in TOOLS:
                with self.subTest(tool=tool):
                    compact = self.invoke(tool, root)
                    pretty = self.invoke(tool, root, "--format", "json", "--pretty")
                    self.assertGreater(len(pretty.splitlines()), 1)
                    data = strict_json(pretty)["data"]
                    self.assertEqual(strict_json(compact), strict_json(pretty))
                    if tool == "scan_runs":
                        self.assertEqual(data["runs"][0]["best_checkpoint"]["value"], "Inf")
                    elif tool == "compare_runs":
                        self.assertIsNone(data["ranking"][0]["best_value"])
                    else:
                        self.assertEqual(data["findings"][0]["code"], "NAN_LOSS")

    @unittest.skipUnless(shutil.which("git"), "Git required for local repository provenance")
    def test_scripts_inside_scanned_repository_leave_scripts_and_git_index_untouched(self):
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp)
            scripts = repo / "scripts"
            scripts.mkdir()
            for script in SCRIPTS.glob("*.py"):
                shutil.copyfile(script, scripts / script.name)
            run = repo / "experiment"
            run.mkdir()
            metrics = run / "metrics.csv"
            metrics.write_text("step,loss\n1,0.25\n", encoding="utf-8")
            env = dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")
            for args in (["init", "-q"], ["add", "."], [
                "-c", "user.name=TrainLens", "-c", "user.email=test@example.invalid",
                "commit", "-qm", "record initial experiment",
            ]):
                subprocess.run(["git", *args], cwd=repo, env=env, capture_output=True, check=True)
            # A timestamp-only change gives ordinary git status a reason to refresh its index.
            stat = metrics.stat()
            os.utime(metrics, ns=(stat.st_atime_ns, stat.st_mtime_ns - 2_000_000_000))
            before = snapshot(repo)
            for tool in TOOLS:
                with self.subTest(tool=tool):
                    flags = ("--repo", str(repo)) if tool == "compare_runs" else ()
                    report = strict_json(self.invoke(tool, repo, *flags, scripts=scripts))
                    if tool == "compare_runs":
                        git = report["data"]["git_provenance"]
                        self.assertTrue(git["repo_head"])
                        self.assertIs(git["repo_dirty"], False)
                    self.assertEqual(before, snapshot(repo))
