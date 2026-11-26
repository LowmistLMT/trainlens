from __future__ import annotations

import io
import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch


SKILL = Path(__file__).resolve().parents[1] / "skills" / "trainlens"
sys.path.insert(0, str(SKILL / "scripts"))

import ssh_fetch  # noqa: E402


class SshFetchTests(unittest.TestCase):
    def invoke(self, run_command, *args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(subprocess, "run", side_effect=run_command) as remote:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                try:
                    code = ssh_fetch.main(list(args))
                except SystemExit as exc:
                    code = exc.code
        return code, stdout.getvalue(), stderr.getvalue(), remote

    def test_gpu_csv_idle_threshold_and_ssh_timeout(self):
        response = (
            '0, "Accelerator, A", 4, 100, 1000, 46, 73.25\n'
            '1, Accelerator B, 5, 900, 1000, 50, 110.50\n'
            '2, Accelerator C, 2, 99, 1000, 42, [N/A]\n'
        ).encode()

        def run_command(argv, **kwargs):
            self.assertEqual(argv[0], "ssh")
            self.assertEqual(argv[-2], "gpu-lab")
            self.assertIn("BatchMode=yes", argv)
            self.assertIn("ConnectTimeout=7", argv)
            self.assertFalse(kwargs.get("shell", False))
            self.assertEqual(kwargs["timeout"], 7)
            remote_args = shlex.split(argv[-1])
            self.assertEqual(remote_args[0], "nvidia-smi")
            self.assertIn(
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,"
                "temperature.gpu,power.draw", remote_args,
            )
            self.assertIn("--format=csv,noheader,nounits", remote_args)
            return subprocess.CompletedProcess(argv, 0, stdout=response, stderr=b"")

        code, output, stderr, remote = self.invoke(
            run_command, "--host", "gpu-lab", "--gpu", "--timeout", "7"
        )
        self.assertEqual(code, 0, stderr)
        self.assertEqual(len(output.splitlines()), 1)
        result = json.loads(output)
        self.assertTrue(result["ok"])
        self.assertEqual(result["tool"], "ssh_fetch")
        self.assertEqual(remote.call_count, 1)
        gpus = result["data"]["gpus"]
        self.assertEqual(len(gpus), 3)
        self.assertEqual(gpus[0]["name"], "Accelerator, A")
        self.assertEqual(gpus[0]["utilization_pct"], 4)
        self.assertEqual(gpus[0]["memory_used_mib"], 100)
        self.assertEqual(gpus[0]["memory_total_mib"], 1000)
        self.assertEqual(gpus[0]["temperature_c"], 46)
        self.assertEqual(gpus[0]["power_w"], 73.25)
        self.assertIsNone(gpus[2]["power_w"])
        findings = result["data"]["findings"]
        self.assertIn("IDLE_GPU_BURN", result["warnings"])
        self.assertEqual([item["code"] for item in findings], ["IDLE_GPU_BURN"])
        code, markdown, stderr, _ = self.invoke(
            run_command, "--host", "gpu-lab", "--gpu", "--timeout", "7", "--format", "md"
        )
        self.assertEqual(code, 0, stderr)
        self.assertIn("Accelerator, A", markdown)
        self.assertIn("IDLE_GPU_BURN", markdown)
        self.assertIn("|", markdown)

    def test_argument_errors_never_start_ssh(self):
        for flags in (
            ["--gpu"],
            ["--host=-invalid", "--gpu"],
            ["--host", "gpu-lab", "--logs"],
            ["--host", "gpu-lab", "--gpu", "--timeout", "0"],
            ["--host", "gpu-lab", "--logs", "--remote-dir", "/runs", "--max-bytes", "-1"],
        ):
            with self.subTest(flags=flags):
                code, _, _, remote = self.invoke(None, *flags)
                self.assertNotEqual(code, 0)
                remote.assert_not_called()

    def test_timeout_gives_a_compact_error(self):
        def timeout(argv, **kwargs):
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

        for flags in (["--gpu"], ["--logs", "--remote-dir", "/experiments"]):
            with self.subTest(flags=flags):
                code, output, _, remote = self.invoke(timeout, "--host", "gpu-lab", *flags)
                result = json.loads(output)
                self.assertEqual(result["tool"], "ssh_fetch")
                self.assertFalse(result["ok"])
                self.assertNotEqual(code, 0)
                self.assertLess(len(output), 1000)
                self.assertNotIn("Traceback", output)
                remote.assert_called_once()

    def test_remote_collection_limits_and_local_diagnosis(self):
        real_run = subprocess.run
        with tempfile.TemporaryDirectory() as temp:
            remote_root = Path(temp) / "lab's experiments"
            experiment = remote_root / "sft"
            experiment.mkdir(parents=True)
            files = {
                "sft/trainer_state.json": json.dumps({
                    "global_step": 20, "max_steps": 30,
                    "log_history": [{"step": 20, "loss": 0.5, "grad_norm": 65}],
                }).encode(),
                "sft/training_args.json": b'{"learning_rate":0.002}',
                "sft/train.log": (
                    b"first line must stay on remote\n" + b"warmup\n" * 250
                    + b"CUDA out of memory at step=21\n"
                ),
                "sft/metrics.jsonl": b"x" * 4096,
                "sft/weights.bin": b"checkpoint",
                "sft/notes.md": b"other",
                "stale/trainer_state.json": b'{"global_step":2,"max_steps":20,"log_history":[]}',
            }
            for name, content in files.items():
                (remote_root / name).parent.mkdir(exist_ok=True)
                (remote_root / name).write_bytes(content)
            old_stamp = 1700000000
            os.utime(remote_root / "stale/trainer_state.json", (old_stamp, old_stamp))
            before = {p: p.stat().st_mtime_ns for p in remote_root.rglob("*")}
            reads = []
            cache = Path(temp) / "cache"

            def run_command(argv, **kwargs):
                self.assertEqual(argv[0], "ssh")
                self.assertFalse(kwargs.get("shell", False))
                self.assertEqual(argv[-2], "gpu-lab")
                command = shlex.split(argv[-1])
                self.assertIn(str(remote_root), command)
                # Exercise the production inventory and reader on a temp filesystem.
                # Only the SSH transport is simulated.
                self.assertEqual(command[0], "python3")
                if command[4] == "read":
                    reads.append(command[-1])
                    self.assertIn(command[-1], {"sft/trainer_state.json", "sft/training_args.json",
                                                "sft/train.log", "stale/trainer_state.json"})
                else:
                    self.assertEqual(command[4], "manifest")
                return real_run(
                    [sys.executable, *command[1:]], capture_output=True, timeout=10, check=False
                )

            code, output, stderr, _ = self.invoke(
                run_command, "--host", "gpu-lab", "--logs", "--remote-dir", str(remote_root),
                "--cache-dir", str(cache), "--max-bytes", "1600",
            )
            self.assertEqual(code, 0, stderr or output)
            self.assertCountEqual(reads, ["sft/trainer_state.json", "sft/training_args.json",
                                         "sft/train.log", "stale/trainer_state.json"])
            data = json.loads(output)["data"]
            self.assertCountEqual(
                [(r["run_id"], r["framework"], r["step_max"], r["status"]) for r in data["runs"]],
                [("sft", "hf", 20, "crashed"), ("stale", "hf", 2, "stalled")],
            )
            self.assertCountEqual(
                [(f["run_id"], f["code"]) for f in data["findings"]],
                [("sft", "CUDA_OOM"), ("sft", "GRAD_EXPLOSION"), ("stale", "STALLED_LOG")],
            )
            log_file, = cache.rglob("train.log")
            tail = log_file.read_bytes()
            self.assertLessEqual(len(tail), 1600)
            self.assertEqual(tail.splitlines(), files["sft/train.log"].splitlines()[-200:])
            cached_files = {p.name for p in cache.rglob("*") if p.is_file()}
            self.assertFalse(cached_files & {"metrics.jsonl", "weights.bin", "notes.md"})
            self.assertEqual(before, {p: p.stat().st_mtime_ns for p in remote_root.rglob("*")})
            for name, content in files.items():
                self.assertEqual((remote_root / name).read_bytes(), content)
