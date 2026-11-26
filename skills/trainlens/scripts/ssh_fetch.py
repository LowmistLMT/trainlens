#!/usr/bin/env python3
"""Read GPU telemetry and training logs from an SSH host into a private local snapshot."""

from __future__ import annotations

import csv
import fnmatch
import io
import math
import os
import re
import shlex
import subprocess
import sys
import tempfile
from contextlib import ExitStack
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Callable, Optional

sys.dont_write_bytecode = True
from _core import discover_runs, dump_json, make_envelope, render_markdown_table
from _core import EnvelopeParser, serialize_run_summary
from diagnose import diagnose_run


GPU_QUERY = (
    "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,"
    "temperature.gpu,power.draw --format=csv,noheader,nounits"
)
FILE_PATTERNS = (
    "trainer_state.json", "training_args.json", "config.json", "adapter_config.json",
    "meta.json", "args.json", "args.yaml", "scalars.json", "*.log.json", "results.csv",
    "metrics.csv", "metrics.jsonl", "events.out.tfevents.*", "*.log",
)
# This program runs from the SSH command argument. It opens only read descriptors.
# Pass the same allowlist to both ends so a manifest can never widen the read scope.
REMOTE_READER = "PATTERNS = " + repr(FILE_PATTERNS) + "\n" + r"""
import fnmatch, os, stat, sys

operation, root, limit = sys.argv[1:4]
maximum = int(limit)
directory = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
fd = os.open(os.path.expanduser(root).rstrip("/") or "/", directory)
def allowed(name):
    return any(fnmatch.fnmatchcase(name, p) for p in PATTERNS)
try:
    if operation == "manifest":
        for parent, dirs, files, dirfd in os.fwalk(".", dir_fd=fd, follow_symlinks=False):
            dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d != "__pycache__")
            for name in sorted(files):
                if not allowed(name):
                    continue
                try:
                    info = os.stat(name, dir_fd=dirfd, follow_symlinks=False)
                except FileNotFoundError:
                    continue
                if stat.S_ISREG(info.st_mode):
                    relative = os.path.relpath(os.path.join(parent, name), ".")
                    record = (os.fsencode(relative), str(info.st_size).encode(),
                              repr(info.st_mtime).encode())
                    sys.stdout.buffer.write(b"\0".join(record) + b"\0")
    elif operation == "read":
        parts = sys.argv[4].split("/")
        if any(p in ("", ".", "..") for p in parts) or not allowed(parts[-1]):
            sys.exit("Unexpected file path")
        for part in parts[:-1]:
            child = os.open(part, directory, dir_fd=fd)
            os.close(fd)
            fd = child
        flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
        with os.fdopen(os.open(parts[-1], flags, dir_fd=fd), "rb") as source:
            info = os.fstat(source.fileno())
            if not stat.S_ISREG(info.st_mode):
                sys.exit("Expected a regular file")
            if parts[-1].endswith(".log"):
                count = min(info.st_size, maximum)
                source.seek(info.st_size - count)
                data = b"".join(source.read(count).splitlines(keepends=True)[-200:])
            else:
                if info.st_size > maximum:
                    sys.exit("File exceeds byte limit")
                data = source.read(min(info.st_size + 1, maximum + 1))
                if len(data) > maximum or len(data) != info.st_size:
                    sys.exit("File changed size during read")
            sys.stdout.buffer.write(data)
finally:
    os.close(fd)
"""


Transport = Callable[[str, str], bytes]


def _error_detail(error: Exception) -> str:
    if isinstance(error, subprocess.TimeoutExpired):
        return f"SSH timed out after {error.timeout}s"
    if isinstance(error, subprocess.CalledProcessError):
        return f"SSH exited with status {error.returncode}"
    return str(error).splitlines()[-1][:200] if str(error) else type(error).__name__


def ssh_read(host: str, command: str, timeout: int) -> bytes:
    """Return binary stdout, preserving exact bytes for event files."""
    try:
        result = subprocess.run(
            ["ssh", "-T", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={timeout}",
             "-o", "ClearAllForwardings=yes", "--", host, command],
            stdin=subprocess.DEVNULL, capture_output=True, timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise OSError(_error_detail(error)) from error
    if result.returncode:
        message = result.stderr.decode("utf-8", errors="replace").strip().splitlines()
        detail = message[-1][:200] if message else f"exit {result.returncode}"
        raise OSError(detail)
    return result.stdout


def parse_gpu_csv(text: str) -> list[dict]:
    """Parse nvidia-smi CSV. Unsupported readings stay null on the GPU in question."""
    def number(raw: str):
        try:
            value = float(raw)
        except ValueError:
            return None
        if not math.isfinite(value):
            return None
        return int(value) if value.is_integer() else value

    gpus = []
    for row in csv.reader(io.StringIO(text), skipinitialspace=True):
        if not row or all(not cell.strip() for cell in row):
            continue
        if len(row) != 7 or not row[0].strip().isdigit():
            raise ValueError("Malformed nvidia-smi GPU CSV row")
        utilization, used, total, temperature, power = map(number, row[2:])
        gpus.append({
            "index": int(row[0]), "name": row[1].strip(), "utilization_pct": utilization,
            "memory_used_mib": used, "memory_total_mib": total,
            "temperature_c": temperature, "power_w": power,
        })
    return gpus


def idle_gpu_findings(gpus: list[dict]) -> list[dict]:
    findings = []
    for gpu in gpus:
        used = gpu["memory_used_mib"]
        total = gpu["memory_total_mib"]
        utilization = gpu["utilization_pct"]
        if (used is not None and total and total > 0 and used / total >= 0.10
                and utilization is not None and utilization < 5):
            findings.append({
                "code": "IDLE_GPU_BURN", "severity": "warning", "gpu_index": gpu["index"],
                "evidence": f"GPU utilization {utilization:g}%, VRAM {used:g}/{total:g} MiB",
            })
    return findings


def reader_command(operation: str, root: str, maximum: int, *args: str) -> str:
    # OpenSSH passes one command string to the remote shell. Quote every argument there.
    return " ".join(shlex.quote(arg) for arg in (
        "python3", "-IBS", "-c", REMOTE_READER, operation, root, str(maximum), *args
    ))


def manifest_entries(raw: bytes):
    if not raw:
        return
    fields = raw.split(b"\0")
    if fields[-1] or (len(fields) - 1) % 3:
        raise ValueError("Incomplete remote file manifest")
    for i in range(0, len(fields) - 1, 3):
        name = fields[i].decode("utf-8")
        size, mtime = int(fields[i + 1]), float(fields[i + 2])
        parts = name.split("/")
        if (any(p in ("", ".", "..") or "\\" in p or PureWindowsPath(p).drive for p in parts)
                or not any(fnmatch.fnmatchcase(parts[-1], p) for p in FILE_PATTERNS)):
            continue
        if size < 0 or not math.isfinite(mtime):
            raise ValueError("Invalid file metadata in remote manifest")
        yield name, size, mtime


def fetch_logs(
    transport: Transport, host: str, remote_dir: str, snapshot: Path, maximum: int
) -> tuple[list[dict], list[dict], list[str]]:
    warnings = []
    warning_count = 0

    def warn(message: str) -> None:
        nonlocal warning_count
        warning_count += 1
        if len(warnings) < 5:
            warnings.append(message[:400])

    snapshot = snapshot.resolve()
    raw = transport(host, reader_command("manifest", remote_dir, maximum))
    for relative, size, mtime in manifest_entries(raw):
        try:
            target = snapshot.joinpath(*relative.split("/")).resolve()
            if not target.is_relative_to(snapshot) or target == snapshot:
                raise ValueError("invalid cache path")
            if size > maximum and not relative.endswith(".log"):
                raise ValueError(f"{size} bytes exceeds {maximum}")
            data = transport(host, reader_command("read", remote_dir, maximum, relative))
            if len(data) > maximum:
                raise ValueError("response exceeds byte limit")
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as file:
                file.write(data)
            os.utime(target, (mtime, mtime))
        except (OSError, ValueError, OverflowError, subprocess.SubprocessError) as error:
            warn(f"Skipped {relative[:160]} ({_error_detail(error)}).")
    runs = []
    findings = []
    for rec in discover_runs([str(snapshot)], limit=None):
        relative = Path(rec.path).relative_to(snapshot)
        local_name = Path(rec.path).name
        remote = PurePosixPath(remote_dir)
        if relative == Path("."):
            rec.run_id = remote.name or str(remote)
        rec.path = str(remote.joinpath(*relative.parts))
        for note in rec.warnings:
            if note.startswith(local_name + "/"):
                note = rec.run_id + "/" + note[len(local_name) + 1:]
            warn(note)
        runs.append(serialize_run_summary(rec))
        findings.extend(diagnose_run(rec))
    if warning_count > len(warnings):
        warnings.append(f"{warning_count - len(warnings)} additional warning(s) omitted.")
    return runs, findings, warnings


def cache_parent(argument: str) -> Path:
    path = Path(argument).expanduser().resolve()
    # Writing even a hidden cache below a local run would change the experiment.
    markers = (
        "trainer_state.json", "checkpoint-*/trainer_state.json", "vis_data/scalars.json",
        "*.log.json", "results.csv", "events.out.tfevents.*", "metrics.jsonl", "scalars.jsonl",
        "events.jsonl", "log.jsonl", "metrics.csv", "log.csv", "history.csv",
    )
    for parent in (path, *path.parents):
        if parent.is_dir() and any(next(parent.glob(p), None) is not None for p in markers):
            raise ValueError("Cache path is inside a local training run")
    path.mkdir(parents=True, exist_ok=True)
    return path


def as_markdown(envelope: dict) -> str:
    data = envelope["data"]
    sections = [envelope["summary"]]
    if data.get("gpus"):
        rows = [[g["index"], g["name"], g["utilization_pct"], g["memory_used_mib"],
                 g["memory_total_mib"], g["temperature_c"], g["power_w"]] for g in data["gpus"]]
        sections.append(render_markdown_table(
            ["GPU", "Name", "Util %", "Used MiB", "Total MiB", "Temp °C", "Power W"], rows
        ))
    if data.get("runs"):
        sections.append(render_markdown_table(
            ["Run", "Framework", "Status", "Step"],
            [[r["run_id"], r["framework"], r["status"], r["step_max"]] for r in data["runs"]],
        ))
    if data["findings"]:
        sections.append(render_markdown_table(
            ["Run / GPU", "Severity", "Finding", "Evidence"],
            [[f.get("run_id", f.get("gpu_index")), f["severity"], f["code"], f["evidence"]]
             for f in data["findings"]],
        ))
    # The finding table already shows which GPUs triggered this envelope warning.
    notes = [message for message in envelope["warnings"] if message != "IDLE_GPU_BURN"]
    if notes:
        sections.append("\n".join(f"- {message}" for message in notes))
    return "\n\n".join(sections)


def main(argv: Optional[list[str]] = None, *, transport: Optional[Transport] = None) -> int:
    parser = EnvelopeParser(prog="ssh_fetch", description=__doc__)
    parser.add_argument("--host", required=True, help="SSH config alias or user@host.")
    parser.add_argument("--remote-dir", help="Remote experiment root.")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--logs", action="store_true")
    parser.add_argument("--cache-dir", help="Parent for a persistent, private local snapshot.")
    parser.add_argument("--max-bytes", type=int, default=2_097_152)
    parser.add_argument("--timeout", type=int, default=15)
    parser.add_argument("--format", choices=("json", "md"), default="json")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.@:%-]*", args.host):
        parser.error("--host must be an SSH alias or user@host")
    if not (args.gpu or args.logs):
        parser.error("select --gpu, --logs, or both")
    if args.logs and not args.remote_dir:
        parser.error("--remote-dir is required with --logs")
    if args.timeout < 1 or args.max_bytes < 1:
        parser.error("--timeout and --max-bytes must be positive")

    if transport is None:
        def transport(host: str, command: str) -> bytes:
            return ssh_read(host, command, args.timeout)

    data = {"host": args.host, "findings": []}
    warnings = []
    ok = True
    if args.gpu:
        try:
            data["gpus"] = parse_gpu_csv(transport(args.host, GPU_QUERY).decode("utf-8"))
            idle = idle_gpu_findings(data["gpus"])
            if idle:
                warnings.append("IDLE_GPU_BURN")
                data["findings"].extend(idle)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            ok = False
            warnings.append(f"GPU query failed ({_error_detail(error)}).")
    if args.logs:
        try:
            with ExitStack() as stack:
                if args.cache_dir:
                    base = cache_parent(args.cache_dir)
                    snapshot = Path(tempfile.mkdtemp(prefix="snapshot-", dir=base))
                    data["cache_dir"] = str(snapshot)
                else:
                    temporary = tempfile.TemporaryDirectory(prefix="trainlens-")
                    snapshot = Path(stack.enter_context(temporary))
                runs, findings, notes = fetch_logs(
                    transport, args.host, args.remote_dir, snapshot, args.max_bytes
                )
                data["runs"] = runs
                data["findings"].extend(findings)
                warnings.extend(notes)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            ok = False
            warnings.append(f"Log fetch failed ({_error_detail(error)}).")
    summary = (f"{args.host} returned {len(data.get('gpus', []))} GPU(s), "
               f"{len(data.get('runs', []))} run(s) and {len(data['findings'])} finding(s).")
    envelope = make_envelope("ssh_fetch", ok, summary, data, warnings)
    print(as_markdown(envelope) if args.format == "md" else dump_json(envelope, pretty=args.pretty))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
