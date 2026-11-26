from __future__ import annotations

import json
import math
import struct
import sys
import tempfile
import unittest
from pathlib import Path


SKILL = Path(__file__).resolve().parents[1] / "skills" / "trainlens"
DEMO = SKILL / "assets" / "demo-logs"
sys.path.insert(0, str(SKILL / "scripts"))

from _core import discover_runs, serialize_run_summary, summarize_series
from _tfevents import read_tfevents_scalars


def _varint(value: int) -> bytes:
    encoded = bytearray()
    while value >= 0x80:
        encoded.append((value & 0x7F) | 0x80)
        value >>= 7
    encoded.append(value)
    return bytes(encoded)


def _bytes_field(number: int, payload: bytes) -> bytes:
    """A length-delimited protobuf field (string, message, or packed values)."""
    return _varint((number << 3) | 2) + _varint(len(payload)) + payload


def _masked_crc32c(payload: bytes) -> bytes:
    crc = 0xFFFFFFFF
    for byte in payload:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0x82F63B78 if crc & 1 else 0)
    crc ^= 0xFFFFFFFF
    rotated = (crc >> 15) | (crc << 17)
    return struct.pack("<I", (rotated + 0xA282EAD8) & 0xFFFFFFFF)


def _tfrecord(event: bytes) -> bytes:
    length = struct.pack("<Q", len(event))
    return length + _masked_crc32c(length) + event + _masked_crc32c(event)


def _summary_event(step: int, *values: bytes) -> bytes:
    wall_time = b"\x09" + struct.pack("<d", 1700000000.0)
    summary = b"".join(_bytes_field(1, value) for value in values)
    return wall_time + b"\x10" + _varint(step) + _bytes_field(5, summary)


class DemoParserTests(unittest.TestCase):
    def test_hf_demo(self):
        runs = {run.run_id: run for run in discover_runs([str(DEMO / "llm-sft")])}
        expected = {
            "sft_baseline": (600, "completed", 500, 1.12),
            "sft_lora_r32_best": (600, "completed", 600, 0.94),
            "sft_lr5e4_nan": (250, "crashed", 100, 1.88),
        }
        self.assertEqual(runs.keys(), expected.keys())
        for name, (last_step, status, best_step, best_loss) in expected.items():
            with self.subTest(run=name):
                run = runs[name]
                self.assertEqual(
                    (run.framework, run.step_max, run.status), ("hf", last_step, status)
                )
                self.assertEqual(run.max_steps, 600)
                self.assertEqual(run.per_class, {})
                self.assertIn((best_step, best_loss), run.series["eval_loss"])
                self.assertEqual(
                    run.best_checkpoint,
                    {"path": f"checkpoint-{best_step}", "step": best_step, "value": best_loss},
                )
        self.assertEqual(runs["sft_lora_r32_best"].config["lora_r"], 32)
        self.assertTrue(math.isnan(runs["sft_lr5e4_nan"].series["loss"][-1][1]))
        self.assertEqual(runs["sft_lr5e4_nan"].series["grad_norm"][-1], (250, math.inf))
        self.assertEqual(serialize_run_summary(runs["sft_baseline"])["metrics"]["grad_norm"], {
            "last": 0.69, "last_step": 600, "max": 1.42, "max_step": 50,
        })
        self.assertEqual(serialize_run_summary(runs["sft_lr5e4_nan"])["metrics"]["grad_norm"], {
            "last": "Inf", "last_step": 250, "max": 142.7, "max_step": 200,
        })

    def test_namespaced_gradient_summary_handles_nonfinite_values_and_ties(self):
        self.assertEqual(summarize_series("train/grad_norm", [(1, 5.0), (2, 5.0), (3, 0.1)]), {
            "last": 0.1, "last_step": 3, "max": 5.0, "max_step": 1,
        })
        self.assertEqual(summarize_series("model/train/grad_norm", [(2, math.nan)]), {
            "last": "NaN", "last_step": 2, "max": None, "max_step": None,
        })

    def test_mmengine_demo(self):
        runs = {run.run_id: run for run in discover_runs([str(DEMO / "mmseg-cityscapes")])}
        self.assertEqual(
            runs.keys(), {"segformer_b0_base", "segformer_b2_aug", "segformer_b2_bs16_oom"}
        )
        for name, miou, pole_iou in (
            ("segformer_b0_base", 74.82, 56.8),
            ("segformer_b2_aug", 78.65, 63.2),
        ):
            with self.subTest(run=name):
                run = runs[name]
                self.assertEqual((run.framework, run.status), ("mmengine", "completed"))
                self.assertEqual((run.step_max, run.max_steps), (40000, 40000))
                self.assertIn((36000, miou), run.series["mIoU"])
                self.assertIn((36000, pole_iou), run.per_class["IoU.pole"])
                self.assertNotIn("IoU.pole", run.series)
                self.assertEqual(
                    run.best_checkpoint, {"path": "best_mIoU_iter_36000.pth", "step": 36000}
                )
        oom = runs["segformer_b2_bs16_oom"]
        self.assertEqual((oom.framework, oom.step_max, oom.status), ("mmengine", 1000, "crashed"))
        self.assertEqual(oom.config["batch_size"], 16)
        self.assertIsNone(oom.best_checkpoint)


class TemporaryRunTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.run_dir = Path(directory.name)

    def test_latest_hf_checkpoint_uses_numeric_step(self):
        for step in (9, 10):
            checkpoint = self.run_dir / f"checkpoint-{step}"
            checkpoint.mkdir()
            (checkpoint / "trainer_state.json").write_text(
                json.dumps({"global_step": step, "log_history": [{"step": step, "loss": 0.5}]}),
                encoding="utf-8",
            )
        run, = discover_runs([str(self.run_dir)])
        self.assertEqual((run.framework, run.step_max), ("hf", 10))
        self.assertEqual(run.series["loss"], [(10, 0.5)])
        self.assertEqual(run.log_files, ["checkpoint-10/trainer_state.json"])

    def test_partial_trainer_state_does_not_hide_other_runs(self):
        for name, content in (("active", '{"log_history":['), ("complete", '{"global_step":7,"max_steps":7}')):
            path = self.run_dir / name
            path.mkdir()
            (path / "trainer_state.json").write_text(content, encoding="utf-8")
        active, complete = discover_runs([str(self.run_dir)])
        self.assertEqual(active.status, "unknown")
        self.assertIn("active/trainer_state.json", active.warnings[0])
        self.assertEqual((complete.run_id, complete.status, complete.step_max), ("complete", "completed", 7))

    def test_ultralytics_csv_and_yaml(self):
        (self.run_dir / "results.csv").write_text(
            " epoch, train/box_loss , metrics/mAP50-95(B) , lr/pg0\n"
            "1,1.5,0.25,0.001\n2,1.25,0.5,0.0005\n",
            encoding="utf-8",
        )
        (self.run_dir / "args.yaml").write_text(
            "model: yolo.pt\nepochs: 2\namp: true\noptimizer:\n  lr: 0.001\n", encoding="utf-8"
        )
        (self.run_dir / "weights").mkdir()
        (self.run_dir / "weights" / "best.pt").touch()
        run, = discover_runs([str(self.run_dir)])
        self.assertEqual((run.framework, run.step_max, run.status), ("ultralytics", 2, "completed"))
        self.assertEqual(run.series["metrics/mAP50-95(B)"], [(1, 0.25), (2, 0.5)])
        self.assertEqual(run.series["train/box_loss"][-1], (2, 1.25))
        self.assertEqual(
            run.config, {"model": "yolo.pt", "epochs": 2, "amp": True, "optimizer.lr": 0.001}
        )
        self.assertEqual(run.best_checkpoint, {"path": "weights/best.pt", "step": None})

    def test_generic_csv(self):
        (self.run_dir / "metrics.csv").write_text(
            " iteration ,val_loss,IoU.car,note\n12,0.75,0.5,first\n24,0.5,0.625,second\n",
            encoding="utf-8",
        )
        run, = discover_runs([str(self.run_dir)])
        self.assertEqual((run.framework, run.step_max), ("csv", 24))
        self.assertEqual(run.series, {"val_loss": [(12, 0.75), (24, 0.5)]})
        self.assertEqual(run.per_class, {"IoU.car": [(12, 0.5), (24, 0.625)]})

    def test_generic_jsonl_fallback_step_and_invalid_tail(self):
        (self.run_dir / "metrics.jsonl").write_text(
            '{"loss": 1.25, "note": "warmup"}\n{"global_step": 8, "loss": 0.75, "Acc.car": 0.5}\n'
            '{"global_step": 9,',
            encoding="utf-8",
        )
        run, = discover_runs([str(self.run_dir)])
        self.assertEqual((run.framework, run.step_max), ("jsonl", 8))
        self.assertEqual(run.series, {"loss": [(1, 1.25), (8, 0.75)]})
        self.assertEqual(run.per_class, {"Acc.car": [(8, 0.5)]})

    def test_tfevents_scalar_records(self):
        # Summary.Value fields: tag=1, simple_value=2, tensor=8.
        loss = _bytes_field(1, b"loss")
        first_loss = loss + b"\x15" + struct.pack("<f", 1.25)
        last_loss = loss + b"\x15" + struct.pack("<f", 0.75)
        # TensorProto fields: dtype=1, tensor_shape=2, tensor_content=4, float_val=5, double_val=6.
        float_tensor = b"\x08\x01" + _bytes_field(4, struct.pack("<f", 0.625))
        double_tensor = b"\x08\x02" + _bytes_field(6, struct.pack("<d", 0.875))
        int_tensor = b"\x08\x03" + _bytes_field(4, struct.pack("<i", 12))
        shape = _bytes_field(2, b"\x08\x02")  # TensorShapeProto.dim[0].size = 2
        vector_tensor = (
            b"\x08\x01" + _bytes_field(2, shape) + _bytes_field(5, struct.pack("<ff", 0.5, 1.0))
        )
        car = _bytes_field(1, b"IoU.car") + _bytes_field(8, float_tensor)
        score = _bytes_field(1, b"score") + _bytes_field(8, double_tensor)
        count = _bytes_field(1, b"count") + _bytes_field(8, int_tensor)
        vector = _bytes_field(1, b"embedding") + _bytes_field(8, vector_tensor)

        events = (
            _bytes_field(3, b"brain.Event:2"),
            _summary_event(8, first_loss, car),
            _summary_event(300, last_loss, score, count, vector),
        )
        event_path = self.run_dir / "events.out.tfevents.123"
        event_path.write_bytes(b"".join(_tfrecord(event) for event in events))
        expected = {
            "loss": [(8, 1.25), (300, 0.75)],
            "IoU.car": [(8, 0.625)],
            "score": [(300, 0.875)],
        }
        self.assertEqual(read_tfevents_scalars(event_path), expected)

        run, = discover_runs([str(self.run_dir)])
        self.assertEqual((run.framework, run.step_max), ("tb", 300))
        self.assertEqual(run.series, {"loss": expected["loss"], "score": expected["score"]})
        self.assertEqual(run.per_class, {"IoU.car": expected["IoU.car"]})

        # A trainer may have started writing the next record during a scan.
        with event_path.open("ab") as stream:
            stream.write(_tfrecord(_summary_event(301, last_loss))[:-5])
        self.assertEqual(read_tfevents_scalars(event_path), expected)
