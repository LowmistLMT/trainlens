from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "skills" / "trainlens"
sys.path.insert(0, str(SKILL / "scripts"))
from _core import RunRecord
from diagnose import diagnose_run, main


def run_cli(path, *options):
    with contextlib.redirect_stdout(io.StringIO()) as output:
        result = main([str(path), *options])
    if result != 0:
        raise AssertionError(f"diagnose returned {result}")
    return json.loads(output.getvalue())["data"]["findings"]


def rules(series, **options):
    run = RunRecord("example", "example", "csv", series=series)
    return {finding["code"]: finding for finding in diagnose_run(run, **options)}


class DiagnoseTests(unittest.TestCase):
    def test_demo_findings_with_steps_severity_and_playbooks(self):
        findings = run_cli(SKILL / "assets" / "demo-logs")
        self.assertEqual(
            [(f["run_id"], f["code"], f["severity"], f["step"]) for f in findings],
            [
                ("sft_baseline", "OVERFITTING", "warning", 600),
                ("sft_lr5e4_nan", "NAN_LOSS", "critical", 250),
                ("sft_lr5e4_nan", "GRAD_EXPLOSION", "critical", 200),
                ("segformer_b2_bs16_oom", "CUDA_OOM", "critical", 1000),
            ],
        )
        self.assertIn("1.12", findings[0]["evidence"])
        self.assertIn("NaN", findings[1]["evidence"])
        self.assertIn("142.7", findings[2]["evidence"])
        self.assertIn("out of memory", findings[3]["evidence"].lower())
        playbooks = (SKILL / "references" / "failure-playbooks.md").read_text(encoding="utf-8")
        for finding in findings:
            self.assertEqual(finding["code"], finding["suggested_playbook"])
            self.assertIn(f'## `{finding["suggested_playbook"]}`', playbooks)

    def test_stalled_and_plateau_with_cli_thresholds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics = root / "metrics.csv"
            metrics.write_text(
                "step,loss\n1,2\n2,1\n3,1.00001\n4,1.00002\n5,1.00001\n6,1\n",
                encoding="utf-8",
            )
            (root / "config.json").write_text('{"max_steps":10}', encoding="utf-8")
            os.utime(metrics, (1_600_000_000, 1_600_000_000))
            findings = run_cli(root)
            self.assertEqual(
                [(f["code"], f["step"], f["severity"]) for f in findings],
                [("STALLED_LOG", 6, "warning"), ("LOSS_PLATEAU", 6, "warning")],
            )
            self.assertTrue(all(f["code"] == f["suggested_playbook"] for f in findings))
            self.assertEqual(run_cli(root, "--stale-minutes", "1e12", "--plateau-window", "6"), [])
            self.assertNotIn("LOSS_PLATEAU", [f["code"] for f in run_cli(root, "--min-delta", "1e-5")])

    def test_gradient_threshold_running_median_and_floor(self):
        def gradient(values, **options):
            return rules({"train/grad_norm": list(enumerate(values))}, **options)

        self.assertEqual(gradient([2, 1, 1.5, 15, 20])["GRAD_EXPLOSION"]["step"], 4)
        self.assertEqual(gradient([1, 15]), {})
        self.assertEqual(gradient([2, 20]), {})
        self.assertEqual(gradient([51])["GRAD_EXPLOSION"]["step"], 0)
        self.assertEqual(gradient([51], grad_threshold=60), {})
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                self.assertIn("GRAD_EXPLOSION", gradient([value]))

    def test_nonfinite_component_losses_but_not_other_metrics(self):
        for key in ("eval_loss", "train/box_loss", "val/cls_loss"):
            with self.subTest(key=key):
                result = rules({"accuracy": [(1, float("nan"))], key: [(5, float("-inf"))]})
                self.assertEqual(result["NAN_LOSS"]["step"], 5)
                self.assertIn("-Inf", result["NAN_LOSS"]["evidence"])
        self.assertEqual(rules({"accuracy": [(1, float("nan"))]}), {})

    def test_overfitting_requires_concurrent_training_decline_and_rebound(self):
        validation = [(2, 1.0), (4, 1.016)]
        self.assertNotIn("OVERFITTING", rules({"loss": [(1, 3), (2, 1), (4, 2)],
                                               "eval_loss": validation}))
        train = [(1, 3), (2, 2), (3, 1)]
        self.assertIn("OVERFITTING", rules({"loss": train, "val_loss": validation}))
        self.assertNotIn("OVERFITTING", rules({"loss": train, "val_loss": [(2, 1), (4, 1.015)]}))

    def test_plateau_requires_finite_nearly_constant_training_window(self):
        self.assertNotIn("LOSS_PLATEAU", rules({"val_loss": [(i, 2) for i in range(7)]}))
        for values in ([1, 1, 1], [1, 1, 3, 1, 1], [1, 1, float("nan"), 1, 1]):
            self.assertNotIn("LOSS_PLATEAU", rules({"loss": list(enumerate(values))}))
        self.assertIn("LOSS_PLATEAU", rules({"train/box_loss": [(i, 1) for i in range(5)]}))

    def test_oom_is_detected_within_requested_log_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "metrics.csv").write_text("step,loss\n12,1\n", encoding="utf-8")
            lines = [f"step={i} FloatingPointError" for i in range(6)]
            lines.extend(["prefix " * 80 + "CUBLAS_STATUS_ALLOC_FAILED", "last line"])
            (root / "train.log").write_text("\n".join(lines), encoding="utf-8")
            findings = run_cli(root)
            self.assertEqual([f["code"] for f in findings], ["CUDA_OOM"])
            self.assertIn("CUBLAS_STATUS_ALLOC_FAILED", findings[0]["evidence"])
            self.assertEqual(findings[0]["step"], 12)
            self.assertEqual(run_cli(root, "--tail-lines", "1"), [])


if __name__ == "__main__":
    unittest.main()
