"""Core deterministic parsing, ledger summarization, config diff, and git provenance engine.

Read-only by construction: never modifies target experiment directories, logs,
checkpoints, or repositories. Requires only the Python standard library (>=3.9).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from _tfevents import read_tfevents_scalars

SCHEMA_VERSION = "1.0"

LOWER_IS_BETTER_TOKENS = (
    "loss",
    "ppl",
    "perplexity",
    "wer",
    "cer",
    "mae",
    "rmse",
    "mse",
    "error",
    "err",
    "bpb",
)
HIGHER_IS_BETTER_TOKENS = (
    "miou",
    "macc",
    "aacc",
    "map",
    "acc",
    "accuracy",
    "f1",
    "auc",
    "bleu",
    "rouge",
    "reward",
    "pass@",
    "iou",
    "precision",
    "recall",
    "dice",
    "score",
)
PER_CLASS_PREFIXES = ("IoU.", "Acc.", "F1.", "AP.", "class_", "per_class/")
DIAGNOSTIC_ONLY_KEYS = {
    "step",
    "global_step",
    "iter",
    "iteration",
    "epoch",
    "time",
    "data_time",
    "eta",
    "eval_runtime",
    "eval_samples_per_second",
    "eval_steps_per_second",
    "train_runtime",
    "train_samples_per_second",
    "train_steps_per_second",
    "total_flos",
}

OOM_PATTERNS = (
    re.compile(r"CUDA out of memory", re.IGNORECASE),
    re.compile(r"torch\.cuda\.OutOfMemoryError", re.IGNORECASE),
    re.compile(r"CUBLAS_STATUS_ALLOC_FAILED", re.IGNORECASE),
    re.compile(r"HIP out of memory", re.IGNORECASE),
    re.compile(r"NCCL.*(?:timeout|watchdog|unhandled system error)", re.IGNORECASE),
)
NAN_LOG_PATTERNS = (
    re.compile(r"\bloss\s*[=:]\s*nan\b", re.IGNORECASE),
    re.compile(r"Loss became NaN", re.IGNORECASE),
    re.compile(r"FloatingPointError", re.IGNORECASE),
)


@dataclass
class RunRecord:
    run_id: str
    path: str
    framework: str
    status: str = "unknown"
    updated_at: Optional[str] = None
    updated_ts: float = 0.0
    step_max: int = 0
    max_steps: Optional[int] = None
    series: Dict[str, List[Tuple[int, float]]] = field(default_factory=dict)
    per_class: Dict[str, List[Tuple[int, float]]] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    best_checkpoint: Optional[Dict[str, Any]] = None
    log_files: List[str] = field(default_factory=list)
    tail_errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def safe_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        lower = text.lower()
        if lower == "nan":
            return float("nan")
        if lower in ("inf", "+inf", "infinity"):
            return float("inf")
        if lower in ("-inf", "-infinity"):
            return float("-inf")
        try:
            return float(text)
        except ValueError:
            return None
    return None


def json_safe_number(val: float) -> Any:
    if math.isnan(val):
        return "NaN"
    if math.isinf(val):
        return "Inf" if val > 0 else "-Inf"
    if abs(val) >= 1e-4 or val == 0.0:
        return round(val, 6)
    return float(f"{val:.4e}")


def infer_metric_mode(metric_name: str) -> str:
    lower = metric_name.lower()
    for tok in LOWER_IS_BETTER_TOKENS:
        if tok in lower:
            return "min"
    for tok in HIGHER_IS_BETTER_TOKENS:
        if tok in lower:
            return "max"
    return "max"


def is_per_class_metric(name: str) -> bool:
    return any(name.startswith(prefix) for prefix in PER_CLASS_PREFIXES)


def read_tail_lines(path: Path, max_lines: int = 200, max_bytes: int = 65536) -> List[str]:
    """Read only the last ``max_lines`` lines without loading large files into memory."""
    try:
        size = path.stat().st_size
        with path.open("rb") as fh:
            if size > max_bytes:
                fh.seek(size - max_bytes)
            raw = fh.read()
        lines = raw.decode("utf-8", errors="replace").splitlines()
        return lines[-max_lines:]
    except OSError:
        return []


def _parse_simple_yaml(text: str) -> Dict[str, Any]:
    """Safe, zero-dependency parser for flat or 1-level YAML config files (e.g. args.yaml)."""
    result: Dict[str, Any] = {}
    current_section: Optional[str] = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if stripped.startswith("- "):
            continue
        if ":" not in stripped:
            continue
        key, raw_val = stripped.split(":", 1)
        key = key.strip().strip("\"'")
        val_str = raw_val.strip()
        if not val_str:
            if indent == 0:
                current_section = key
            continue
        parsed_val = _coerce_scalar(val_str)
        if indent > 0 and current_section:
            result[f"{current_section}.{key}"] = parsed_val
        else:
            current_section = None
            result[key] = parsed_val
    return result


def _coerce_scalar(text: str) -> Any:
    if (text.startswith('"') and text.endswith('"')) or (
        text.startswith("'") and text.endswith("'")
    ):
        return text[1:-1]
    lower = text.lower()
    if lower in ("true", "yes"):
        return True
    if lower in ("false", "no"):
        return False
    if lower in ("null", "none", "~"):
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def flatten_dict(data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for k, v in data.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            flat.update(flatten_dict(v, prefix=key))
        elif isinstance(v, list):
            if len(v) <= 6 and all(isinstance(x, (str, int, float, bool)) for x in v):
                flat[key] = v
        elif isinstance(v, (str, int, float, bool)) or v is None:
            flat[key] = v
    return flat


def _load_run_configs(run_dir: Path) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    json_candidates = [
        "training_args.json",
        "config.json",
        "adapter_config.json",
        "meta.json",
        "args.json",
    ]
    for fname in json_candidates:
        fpath = run_dir / fname
        if fpath.is_file():
            try:
                loaded = json.loads(fpath.read_text(encoding="utf-8", errors="replace"))
                if isinstance(loaded, dict):
                    merged.update(flatten_dict(loaded))
            except (OSError, json.JSONDecodeError):
                pass
    for fname in ("args.yaml", "train_args.yaml", "config.yaml"):
        fpath = run_dir / fname
        if fpath.is_file():
            try:
                merged.update(_parse_simple_yaml(fpath.read_text(encoding="utf-8", errors="replace")))
            except OSError:
                pass
    return merged


def _find_text_logs(run_dir: Path) -> List[Path]:
    logs: List[Path] = []
    for pattern in ("*.log", "trainer_log.txt", "stdout.txt", "stderr.txt"):
        logs.extend(sorted(run_dir.glob(pattern)))
    return logs[:5]


def _scan_tail_errors(log_paths: Sequence[Path], tail_lines: int = 200) -> List[str]:
    oom: List[str] = []
    nan: List[str] = []
    for lp in log_paths:
        for line in read_tail_lines(lp, max_lines=tail_lines):
            for patterns, samples in ((OOM_PATTERNS, oom), (NAN_LOG_PATTERNS, nan)):
                if len(samples) >= 5:
                    continue
                for pattern in patterns:
                    match = pattern.search(line)
                    if match:
                        # Keep the matched text even when a logger adds a long prefix.
                        start = max(0, match.end() - 200)
                        snippet = line[start:start + 200].strip()
                        if match.end() - match.start() > 200:
                            snippet = match.group()[:96] + " ... " + match.group()[-96:]
                        if snippet not in samples:
                            samples.append(snippet)
                        break
    # Repeated lines of one kind must not crowd out evidence of the other.
    if oom and nan:
        return (oom[:4] + nan)[:5]
    return oom or nan


def _latest_mtime(paths: Iterable[Path]) -> float:
    latest = 0.0
    for p in paths:
        try:
            mt = p.stat().st_mtime
            if mt > latest:
                latest = mt
        except OSError:
            continue
    return latest


def _format_utc(ts: float) -> Optional[str]:
    if ts <= 0:
        return None
    return dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _extract_step(entry: Dict[str, Any], fallback_idx: int) -> int:
    for key in ("step", "global_step", "iter", "iteration", "epoch"):
        if key in entry:
            val = safe_float(entry[key])
            if val is not None and not math.isnan(val) and not math.isinf(val):
                return int(round(val))
    return fallback_idx


def _append_entry_metrics(
    entry: Dict[str, Any],
    step: int,
    series: Dict[str, List[Tuple[int, float]]],
    per_class: Dict[str, List[Tuple[int, float]]],
) -> None:
    for raw_k, raw_v in entry.items():
        k = str(raw_k).strip()
        if not k or k in DIAGNOSTIC_ONLY_KEYS:
            continue
        val = safe_float(raw_v)
        if val is None:
            continue
        if is_per_class_metric(k):
            per_class.setdefault(k, []).append((step, val))
        else:
            series.setdefault(k, []).append((step, val))


def _detect_best_pth(run_dir: Path) -> Optional[Dict[str, Any]]:
    candidates = sorted(run_dir.glob("best_*.pth")) + sorted(run_dir.glob("best_*.pt"))
    if not candidates:
        return None
    chosen = candidates[-1]
    step_match = re.search(r"(?:iter|step|epoch)[_-]?(\d+)", chosen.name, re.IGNORECASE)
    step = int(step_match.group(1)) if step_match else None
    return {"path": chosen.name, "step": step}


def _parse_hf_run(run_dir: Path, state_file: Path, tail_lines: int = 200) -> RunRecord:
    source_name = str(state_file.relative_to(run_dir))
    warnings = []
    try:
        data = json.loads(state_file.read_text(encoding="utf-8", errors="replace"))
        if not isinstance(data, dict) or not isinstance(data.get("log_history", []), list):
            raise ValueError("unexpected trainer state structure")
        step_max = int(data.get("global_step") or 0)
    except (OSError, ValueError, TypeError, OverflowError):
        # A local scan or remote snapshot can coincide with a trainer rewriting its state.
        warnings.append(f"{run_dir.name}/{source_name}: could not read trainer state.")
        data = {}
        step_max = 0
    series: Dict[str, List[Tuple[int, float]]] = {}
    per_class: Dict[str, List[Tuple[int, float]]] = {}
    max_steps = data.get("max_steps")
    if isinstance(max_steps, (int, float)) and max_steps > 0 and math.isfinite(max_steps):
        max_steps = int(max_steps)
    else:
        max_steps = None

    for idx, item in enumerate(data.get("log_history", []), start=1):
        if not isinstance(item, dict):
            continue
        step = _extract_step(item, idx)
        step_max = max(step_max, step)
        _append_entry_metrics(item, step, series, per_class)

    config = _load_run_configs(run_dir)
    if max_steps is None and isinstance(config.get("max_steps"), int):
        max_steps = config["max_steps"]

    best_ckpt = None
    if data.get("best_model_checkpoint"):
        ckpt_name = str(data["best_model_checkpoint"])
        step_match = re.search(r"checkpoint-(\d+)", ckpt_name)
        best_ckpt = {
            "path": ckpt_name,
            "step": int(step_match.group(1)) if step_match else None,
            "value": safe_float(data.get("best_metric")),
        }

    text_logs = _find_text_logs(run_dir)
    tail_errors = _scan_tail_errors(text_logs, tail_lines=tail_lines)
    mtime = _latest_mtime([state_file, *text_logs])
    rec = RunRecord(
        run_id=run_dir.name,
        path=str(run_dir),
        framework="hf",
        updated_ts=mtime,
        updated_at=_format_utc(mtime),
        step_max=step_max,
        max_steps=max_steps,
        series=series,
        per_class=per_class,
        config=config,
        best_checkpoint=best_ckpt,
        log_files=[source_name] + [p.name for p in text_logs],
        tail_errors=tail_errors,
        warnings=warnings,
    )
    return rec


def _parse_mmengine_run(run_dir: Path, scalar_file: Path, tail_lines: int = 200) -> RunRecord:
    series: Dict[str, List[Tuple[int, float]]] = {}
    per_class: Dict[str, List[Tuple[int, float]]] = {}
    step_max = 0

    for idx, raw_line in enumerate(
        scalar_file.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(item, dict):
            continue
        step = _extract_step(item, idx)
        step_max = max(step_max, step)
        _append_entry_metrics(item, step, series, per_class)

    config = _load_run_configs(run_dir)
    max_steps = None
    for k in ("max_iters", "max_steps", "train_cfg.max_iters"):
        if isinstance(config.get(k), int) and config[k] > 0:
            max_steps = config[k]
            break

    best_ckpt = _detect_best_pth(run_dir)
    text_logs = _find_text_logs(run_dir)
    tail_errors = _scan_tail_errors(text_logs, tail_lines=tail_lines)
    mtime = _latest_mtime([scalar_file, *text_logs])
    return RunRecord(
        run_id=run_dir.name,
        path=str(run_dir),
        framework="mmengine",
        updated_ts=mtime,
        updated_at=_format_utc(mtime),
        step_max=step_max,
        max_steps=max_steps,
        series=series,
        per_class=per_class,
        config=config,
        best_checkpoint=best_ckpt,
        log_files=[str(scalar_file.relative_to(run_dir))] + [p.name for p in text_logs],
        tail_errors=tail_errors,
    )


def _parse_csv_run(
    run_dir: Path, csv_file: Path, framework: str, tail_lines: int = 200
) -> RunRecord:
    series: Dict[str, List[Tuple[int, float]]] = {}
    per_class: Dict[str, List[Tuple[int, float]]] = {}
    step_max = 0

    with csv_file.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh)
        for idx, row in enumerate(reader, start=1):
            cleaned = {str(k).strip(): v for k, v in row.items() if k is not None}
            step = _extract_step(cleaned, idx)
            step_max = max(step_max, step)
            _append_entry_metrics(cleaned, step, series, per_class)

    config = _load_run_configs(run_dir)
    max_steps = None
    for k in ("epochs", "max_steps", "max_iters"):
        if isinstance(config.get(k), int) and config[k] > 0:
            max_steps = config[k]
            break

    best_ckpt = _detect_best_pth(run_dir)
    if best_ckpt is None and (run_dir / "weights" / "best.pt").is_file():
        best_ckpt = {"path": "weights/best.pt", "step": None}

    text_logs = _find_text_logs(run_dir)
    tail_errors = _scan_tail_errors(text_logs, tail_lines=tail_lines)
    mtime = _latest_mtime([csv_file, *text_logs])
    return RunRecord(
        run_id=run_dir.name,
        path=str(run_dir),
        framework=framework,
        updated_ts=mtime,
        updated_at=_format_utc(mtime),
        step_max=step_max,
        max_steps=max_steps,
        series=series,
        per_class=per_class,
        config=config,
        best_checkpoint=best_ckpt,
        log_files=[csv_file.name] + [p.name for p in text_logs],
        tail_errors=tail_errors,
    )


def _parse_jsonl_run(run_dir: Path, jsonl_file: Path, tail_lines: int = 200) -> RunRecord:
    rec = _parse_mmengine_run(run_dir, jsonl_file, tail_lines=tail_lines)
    rec.framework = "jsonl"
    return rec


def _parse_tb_run(run_dir: Path, tb_files: Sequence[Path], tail_lines: int = 200) -> RunRecord:
    series: Dict[str, List[Tuple[int, float]]] = {}
    per_class: Dict[str, List[Tuple[int, float]]] = {}
    step_max = 0

    for tf in sorted(tb_files):
        parsed = read_tfevents_scalars(tf)
        for tag, points in parsed.items():
            target = per_class if is_per_class_metric(tag) else series
            target.setdefault(tag, []).extend(points)
            for s, _ in points:
                step_max = max(step_max, s)

    for bucket in (series, per_class):
        for tag in list(bucket.keys()):
            bucket[tag].sort(key=lambda x: x[0])

    config = _load_run_configs(run_dir)
    max_steps = config.get("max_steps") if isinstance(config.get("max_steps"), int) else None
    text_logs = _find_text_logs(run_dir)
    tail_errors = _scan_tail_errors(text_logs, tail_lines=tail_lines)
    mtime = _latest_mtime([*tb_files, *text_logs])
    return RunRecord(
        run_id=run_dir.name,
        path=str(run_dir),
        framework="tb",
        updated_ts=mtime,
        updated_at=_format_utc(mtime),
        step_max=step_max,
        max_steps=max_steps,
        series=series,
        per_class=per_class,
        config=config,
        best_checkpoint=_detect_best_pth(run_dir),
        log_files=[p.name for p in sorted(tb_files)] + [p.name for p in text_logs],
        tail_errors=tail_errors,
    )


def _has_nan_or_inf(series: Dict[str, List[Tuple[int, float]]]) -> bool:
    for points in series.values():
        for _, v in points:
            if math.isnan(v) or math.isinf(v):
                return True
    return False


def _finalize_status(rec: RunRecord, stale_minutes: float = 30.0, now_ts: Optional[float] = None) -> None:
    if rec.tail_errors or _has_nan_or_inf(rec.series):
        rec.status = "crashed"
        return
    if rec.warnings and not rec.series and not rec.per_class:
        rec.status = "unknown"
        return
    if rec.max_steps and rec.step_max >= rec.max_steps:
        rec.status = "completed"
        return
    if now_ts is None:
        now_ts = dt.datetime.now(tz=dt.timezone.utc).timestamp()
    if rec.updated_ts > 0:
        age_min = (now_ts - rec.updated_ts) / 60.0
        if age_min <= stale_minutes:
            rec.status = "running"
        elif rec.max_steps and rec.step_max < rec.max_steps:
            rec.status = "stalled"
        else:
            rec.status = "completed"
    else:
        rec.status = "unknown"


def _detect_run_in_dir(
    run_dir: Path, framework_filter: str = "auto", tail_lines: int = 200
) -> Optional[RunRecord]:
    # 1. HuggingFace trainer_state.json
    if framework_filter in ("auto", "hf"):
        hf_state = run_dir / "trainer_state.json"
        if not hf_state.is_file():
            ckpts = list(run_dir.glob("checkpoint-*/trainer_state.json"))
            if ckpts:
                def checkpoint_order(state: Path) -> Tuple[int, str]:
                    suffix = state.parent.name.partition("-")[2]
                    return (int(suffix) if suffix.isdecimal() else -1, suffix)

                hf_state = max(ckpts, key=checkpoint_order)
        if hf_state.is_file():
            return _parse_hf_run(run_dir, hf_state, tail_lines=tail_lines)

    # 2. MMEngine / MMSegmentation vis_data/scalars.json or *.log.json
    if framework_filter in ("auto", "mmengine"):
        mm_scalars = run_dir / "vis_data" / "scalars.json"
        if not mm_scalars.is_file():
            log_jsons = sorted(run_dir.glob("*.log.json"))
            if log_jsons:
                mm_scalars = log_jsons[-1]
        if mm_scalars.is_file():
            return _parse_mmengine_run(run_dir, mm_scalars, tail_lines=tail_lines)

    # 3. Ultralytics results.csv
    if framework_filter in ("auto", "ultralytics"):
        ultra_csv = run_dir / "results.csv"
        if ultra_csv.is_file():
            return _parse_csv_run(run_dir, ultra_csv, "ultralytics", tail_lines=tail_lines)

    # 4. TensorBoard events.out.tfevents.*
    if framework_filter in ("auto", "tb"):
        tb_files = sorted(run_dir.glob("events.out.tfevents.*"))
        if tb_files:
            return _parse_tb_run(run_dir, tb_files, tail_lines=tail_lines)

    # 5. Generic JSONL / CSV
    if framework_filter in ("auto", "jsonl"):
        for fname in ("metrics.jsonl", "scalars.jsonl", "events.jsonl", "log.jsonl"):
            jf = run_dir / fname
            if jf.is_file():
                return _parse_jsonl_run(run_dir, jf, tail_lines=tail_lines)

    if framework_filter in ("auto", "csv"):
        for fname in ("metrics.csv", "log.csv", "history.csv"):
            cf = run_dir / fname
            if cf.is_file():
                return _parse_csv_run(run_dir, cf, "csv", tail_lines=tail_lines)

    return None


def discover_runs(
    roots: Sequence[str],
    framework: str = "auto",
    stale_minutes: float = 30.0,
    limit: Optional[int] = 50,
    tail_lines: int = 200,
    now_ts: Optional[float] = None,
) -> List[RunRecord]:
    """Recursively discover training runs without duplicating nested directories."""
    discovered: List[RunRecord] = []
    seen_dirs: set[Path] = set()

    for raw_root in roots:
        if limit is not None and len(discovered) >= limit:
            break
        root = Path(raw_root).expanduser().resolve()
        if not root.exists():
            continue
        if root.is_file():
            rec = _parse_single_file(root, framework=framework, tail_lines=tail_lines)
            if rec and Path(rec.path) not in seen_dirs:
                _finalize_status(rec, stale_minutes=stale_minutes, now_ts=now_ts)
                discovered.append(rec)
                seen_dirs.add(Path(rec.path))
            continue

        # Walk top-down so once a directory is recognized as a run, we don't descend into vis_data/checkpoints.
        for dirpath, dirnames, _filenames in os.walk(root):
            current = Path(dirpath)
            dirnames[:] = sorted(
                d for d in dirnames if not d.startswith(".") and d not in ("__pycache__", "node_modules")
            )
            if current in seen_dirs:
                dirnames[:] = []
                continue
            rec = _detect_run_in_dir(current, framework_filter=framework, tail_lines=tail_lines)
            if rec is not None:
                _finalize_status(rec, stale_minutes=stale_minutes, now_ts=now_ts)
                discovered.append(rec)
                seen_dirs.add(current)
                dirnames[:] = []  # stop descending inside an identified run
                if limit is not None and len(discovered) >= limit:
                    break

    # Two projects can use the same run name. Preserve both columns in keyed reports.
    by_name: Dict[str, List[RunRecord]] = {}
    for run in discovered:
        by_name.setdefault(run.run_id, []).append(run)
    for group in by_name.values():
        if len(group) > 1:
            try:
                parent = os.path.commonpath([Path(run.path).parent for run in group])
            except ValueError:  # Different Windows drives have no common parent.
                parent = None
            for run in group:
                run.run_id = str(Path(run.path).relative_to(parent)) if parent else run.path
    return discovered


def _parse_single_file(
    file_path: Path, framework: str = "auto", tail_lines: int = 200
) -> Optional[RunRecord]:
    name = file_path.name
    if name == "trainer_state.json" and framework in ("auto", "hf"):
        return _parse_hf_run(file_path.parent, file_path, tail_lines=tail_lines)
    if name.startswith("events.out.tfevents.") and framework in ("auto", "tb"):
        return _parse_tb_run(file_path.parent, [file_path], tail_lines=tail_lines)
    if name == "results.csv" and framework in ("auto", "ultralytics"):
        return _parse_csv_run(file_path.parent, file_path, "ultralytics", tail_lines=tail_lines)
    if name.endswith(".csv") and framework in ("auto", "csv"):
        return _parse_csv_run(file_path.parent, file_path, "csv", tail_lines=tail_lines)
    if (name.endswith(".jsonl") or name.endswith(".json")) and framework in ("auto", "jsonl", "mmengine"):
        return _parse_jsonl_run(file_path.parent, file_path, tail_lines=tail_lines)
    return None


def summarize_series(metric_name: str, points: List[Tuple[int, float]]) -> Dict[str, Any]:
    """Compress a metric time series into a low-token statistical summary."""
    if not points:
        return {}
    finite = [(s, v) for s, v in points if not math.isnan(v) and not math.isinf(v)]
    last_step, last_val = points[-1]
    lower_name = metric_name.lower()

    if lower_name.rsplit("/", 1)[-1] == "grad_norm":
        max_step, max_val = max(finite, key=lambda point: point[1], default=(None, None))
        return {
            "last": json_safe_number(last_val),
            "last_step": last_step,
            "max": json_safe_number(max_val) if max_val is not None else None,
            "max_step": max_step,
        }

    if lower_name in ("lr", "learning_rate") or lower_name.startswith("lr/"):
        out: Dict[str, Any] = {"last": json_safe_number(last_val)}
        if finite:
            out["max"] = json_safe_number(max(v for _, v in finite))
        return out

    if lower_name in ("memory", "mem", "gpu_mem"):
        out = {"last": json_safe_number(last_val)}
        if finite:
            out["max"] = json_safe_number(max(v for _, v in finite))
        return out

    mode = infer_metric_mode(metric_name)
    out = {"last": json_safe_number(last_val), "last_step": last_step}
    if finite:
        best_step, best_val = (
            min(finite, key=lambda x: x[1]) if mode == "min" else max(finite, key=lambda x: x[1])
        )
        out["best"] = json_safe_number(best_val)
        out["best_step"] = best_step
        out["trend"] = _compute_trend(finite, mode=mode)
    else:
        out["trend"] = "diverging"
    return out


def _compute_trend(finite: List[Tuple[int, float]], mode: str) -> str:
    if len(finite) < 3:
        return "insufficient"
    vals = [v for _, v in finite]
    recent = vals[-min(4, len(vals)) :]
    first_val = recent[0]
    last_val = recent[-1]
    span = max(recent) - min(recent)
    scale = max(abs(first_val), 1e-6)
    rel_change = (last_val - first_val) / scale

    if span / scale < 0.005:
        return "plateau"
    if mode == "min":
        if rel_change < -0.02:
            return "improving"
        if rel_change > 0.05:
            return "diverging"
    else:
        if rel_change > 0.01:
            return "improving"
        if rel_change < -0.02:
            return "diverging"
    return "volatile"


def select_compact_config(config: Dict[str, Any], max_keys: int = 12) -> Dict[str, Any]:
    priority = (
        "model",
        "model_name_or_path",
        "dataset",
        "lr",
        "learning_rate",
        "batch_size",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "lora_r",
        "lora_alpha",
        "crop_size",
        "data_aug",
        "bf16",
        "fp16",
        "max_grad_norm",
        "weight_decay",
        "git_commit",
    )
    compact: Dict[str, Any] = {}
    for k in priority:
        if k in config and len(compact) < max_keys:
            compact[k] = config[k]
    if len(compact) < max_keys:
        for k in sorted(config.keys()):
            if k not in compact and len(compact) < max_keys:
                compact[k] = config[k]
    return compact


def compute_config_diff(runs: Sequence[RunRecord]) -> Dict[str, Dict[str, Any]]:
    """Return only config keys whose values differ across the given runs."""
    all_keys = sorted({k for r in runs for k in r.config.keys()})
    diff: Dict[str, Dict[str, Any]] = {}
    for k in all_keys:
        vals = [r.config.get(k, "<unset>") for r in runs]
        serialized = {json.dumps(v, sort_keys=True, default=str) for v in vals}
        if len(serialized) > 1:
            diff[k] = {r.run_id: r.config.get(k, None) for r in runs}
    return diff


def align_git_provenance(
    runs: Sequence[RunRecord], repo_path: Optional[str] = None
) -> Dict[str, Any]:
    """Read-only alignment of run git commits with local git log / diff."""
    commits = {r.run_id: r.config.get("git_commit") for r in runs if r.config.get("git_commit")}
    provenance: Dict[str, Any] = {"run_commits": commits}
    if not repo_path:
        return provenance

    repo = Path(repo_path).expanduser().resolve()
    if not (repo / ".git").exists():
        return provenance

    def _run_git(args: List[str]) -> Optional[str]:
        try:
            proc = subprocess.run(
                ["git", "--no-optional-locks", "-c", "core.fsmonitor=false", "-C", str(repo), *args],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if proc.returncode == 0:
                return proc.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
        return None

    head_sha = _run_git(["rev-parse", "--short", "HEAD"])
    status = _run_git(["status", "--porcelain"])
    provenance["repo_head"] = head_sha
    provenance["repo_dirty"] = bool(status) if status is not None else None

    # If two distinct commits exist in the local repo, summarize changed files between them.
    unique_shas = list(dict.fromkeys(str(c) for c in commits.values() if c))
    if len(unique_shas) >= 2:
        # Pass only resolved object IDs to diff. Revisions come from the run configs.
        first, last = (
            _run_git(["rev-parse", "--verify", "--end-of-options", f"{rev}^{{commit}}"])
            for rev in (unique_shas[0], unique_shas[-1])
        )
        diff_stat = None
        if first and last:
            diff_stat = _run_git(["diff", "--no-ext-diff", "--no-textconv", "--name-status", first, last, "--"])
        if diff_stat:
            provenance["commit_diff_files"] = {
                "from": unique_shas[0],
                "to": unique_shas[-1],
                "files": diff_stat.splitlines()[:20],
            }
    return provenance


def serialize_run_summary(rec: RunRecord, include_per_class: bool = False) -> Dict[str, Any]:
    metrics = {k: summarize_series(k, pts) for k, pts in sorted(rec.series.items())}
    anomalies: List[str] = []
    if _has_nan_or_inf(rec.series):
        anomalies.append("NAN_OR_INF")
    if any(any(p.search(e) for p in OOM_PATTERNS) for e in rec.tail_errors):
        anomalies.append("CUDA_OOM")
    if rec.status == "stalled":
        anomalies.append("STALLED_LOG")

    payload: Dict[str, Any] = {
        "run_id": rec.run_id,
        "path": rec.path,
        "framework": rec.framework,
        "status": rec.status,
        "updated_at": rec.updated_at,
        "step_max": rec.step_max,
        "best_checkpoint": rec.best_checkpoint,
        "metrics": metrics,
        "config": select_compact_config(rec.config),
        "anomalies": anomalies,
    }
    if include_per_class and rec.per_class:
        payload["per_class"] = {
            k: summarize_series(k, pts) for k, pts in sorted(rec.per_class.items())
        }
    return payload


def make_envelope(
    tool: str,
    ok: bool,
    summary: str,
    data: Dict[str, Any],
    warnings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": tool,
        "ok": ok,
        "summary": summary,
        "data": data,
        "warnings": warnings or [],
    }


def dump_json(payload: Dict[str, Any], pretty: bool = False) -> str:
    def safe(value: Any) -> Any:
        if isinstance(value, float) and not math.isfinite(value):
            return json_safe_number(value)
        if isinstance(value, dict):
            return {k: safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [safe(v) for v in value]
        return value

    return json.dumps(
        safe(payload), indent=2 if pretty else None, separators=None if pretty else (",", ":"),
        ensure_ascii=False, allow_nan=False,
    )


class EnvelopeParser(argparse.ArgumentParser):
    """Keep CLI argument errors machine-readable while preserving standard --help."""

    def parse_args(self, args=None, namespace=None):
        self.output_options = namespace if namespace is not None else argparse.Namespace()
        return super().parse_args(args, self.output_options)

    def error(self, message: str) -> None:
        envelope = make_envelope(self.prog, False, message, {}, [])
        if getattr(self.output_options, "format", None) == "md":
            print(f"**{self.prog}** {envelope['summary']}")
        else:
            print(dump_json(envelope, pretty=getattr(self.output_options, "pretty", False)))
        self.exit(2)


def render_markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    def cell(value: Any) -> str:
        escaped = str(value).replace("\\", "\\\\").replace("|", "\\|")
        return "<br>".join(escaped.splitlines())

    lines = [
        "| " + " | ".join(cell(h) for h in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(cell(c) for c in row) + " |")
    return "\n".join(lines)
