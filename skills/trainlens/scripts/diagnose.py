#!/usr/bin/env python3
"""Diagnose training anomalies from logged metrics and bounded text-log tails."""

from __future__ import annotations

import heapq
import math
import re
import sys
from bisect import bisect_right
from operator import itemgetter
from typing import Iterable, Optional

sys.dont_write_bytecode = True
from _core import OOM_PATTERNS, RunRecord, discover_runs, dump_json, json_safe_number
from _core import EnvelopeParser, make_envelope, render_markdown_table


Points = list[tuple[int, float]]
Signal = tuple[int, str]
TRAIN_LOSS_KEYS = ("loss", "train/loss", "train_loss", "training_loss", "train/total_loss")
VALIDATION_LOSS_KEYS = ("eval_loss", "val_loss", "eval/loss", "val/loss", "validation/loss")
LOG_STEP = re.compile(r"\b(?:global_step|step|iter(?:ation)?)\s*[:=#]?\s*(\d+)\b", re.I)


def _gradient_spike(metric: str, points: Points, threshold: float) -> Optional[Signal]:
    # Two heaps keep the median of all preceding observations in O(log n) per update.
    lower: list[float] = []  # Negated values, including the lower middle value.
    upper: list[float] = []
    for step, value in sorted(points, key=itemgetter(0)):
        if not math.isfinite(value):
            return step, f"{metric}={json_safe_number(value)}"
        if value > threshold:
            return step, f"{metric}={value:g} exceeds {threshold:g}"
        if lower and value > 15:
            median = -lower[0]
            if len(lower) == len(upper):
                median = (median + upper[0]) / 2
            if value > 10 * median:
                return step, f"{metric}={value:g} exceeds 10x prior median {median:g}"
        if not lower or value <= -lower[0]:
            heapq.heappush(lower, -value)
        else:
            heapq.heappush(upper, value)
        if len(lower) > len(upper) + 1:
            heapq.heappush(upper, -heapq.heappop(lower))
        elif len(upper) > len(lower):
            heapq.heappush(lower, -heapq.heappop(upper))
    return None


def _overfitting(
    train_key: str, train: Points, val_key: str, validation: Points
) -> Optional[Signal]:
    """Find a post-best rebound accompanied by a training-loss decline over that interval."""
    train = [(step, value) for step, value in train if math.isfinite(value)]
    valid = sorted(((s, v) for s, v in validation if math.isfinite(v)), key=itemgetter(0))
    if len(train) < 2 or len(valid) < 2:
        return None
    best_step, best = min(valid, key=itemgetter(1))
    steps = [step for step, _ in train]
    # Use the training measurement at the best eval, or the nearest preceding one.
    start = max(0, bisect_right(steps, best_step) - 1)
    train_step, train_start = train[start]
    for step, value in valid:
        if step <= best_step or value <= best + abs(best) * 0.015:
            continue
        end = bisect_right(steps, step) - 1
        if end > start:
            end_step, train_end = train[end]
            if end_step > train_step and train_end < train_start:
                return step, (
                    f"{val_key} {best:g} (step {best_step}) -> {value:g}, "
                    f"{train_key} {train_start:g} (step {train_step}) -> "
                    f"{train_end:g} (step {end_step})"
                )
    return None


def diagnose_run(
    rec: RunRecord, plateau_window: int = 5, min_delta: float = 1e-4, grad_threshold: float = 50.0
) -> list[dict]:
    """Return at most one evidence-backed finding per code for a run."""
    findings: list[dict] = []

    def record(severity: str, code: str, signals: Iterable[Optional[Signal]]) -> None:
        earliest = min((s for s in signals if s is not None), default=None)
        if earliest is not None:
            step, evidence = earliest
            findings.append({
                "run_id": rec.run_id,
                "severity": severity,
                "code": code,
                "step": step,
                "evidence": evidence,
                "suggested_playbook": code,
            })

    record("critical", "NAN_LOSS", (
        (step, f"{key}={json_safe_number(value)}")
        for metrics in (rec.series, rec.per_class)
        for key, points in metrics.items() if "loss" in key.lower()
        for step, value in points if not math.isfinite(value)
    ))
    record("critical", "GRAD_EXPLOSION", (
        _gradient_spike(key, points, grad_threshold) for key, points in rec.series.items()
        if key.lower().split("/")[-1] == "grad_norm"
    ))
    for line in rec.tail_errors:
        if any(pattern.search(line) for pattern in OOM_PATTERNS):
            match = LOG_STEP.search(line)
            step = int(match.group(1)) if match else rec.step_max
            evidence = line if match else f"{line} (step is last logged)"
            record("critical", "CUDA_OOM", [(step, evidence)])
            break
    if rec.status == "stalled":
        evidence = f"Last log update {rec.updated_at}, progress {rec.step_max}/{rec.max_steps}"
        record("warning", "STALLED_LOG", [(rec.step_max, evidence)])

    keys = {key.lower(): key for key in rec.series}
    train_keys = [keys[name] for name in TRAIN_LOSS_KEYS if name in keys][:1]
    if not train_keys:
        train_keys = [
            key for name, key in keys.items() if name.startswith("train/") and "loss" in name
        ]
    rebound_signals = []
    plateau_signals = []
    for key in train_keys:
        train = sorted(rec.series[key], key=itemgetter(0))
        for val in VALIDATION_LOSS_KEYS:
            if val in keys:
                rebound_signals.append(_overfitting(key, train, keys[val], rec.series[keys[val]]))
        tail = train[-plateau_window:]
        if len(tail) == plateau_window and all(math.isfinite(value) for _, value in tail):
            values = [value for _, value in tail]
            spread = max(values) - min(values)
            if spread < min_delta:
                evidence = (
                    f"{key} range {spread:g} < {min_delta:g} across {plateau_window} points "
                    f"(steps {tail[0][0]} to {tail[-1][0]})"
                )
                plateau_signals.append((tail[-1][0], evidence))
    record("warning", "OVERFITTING", rebound_signals)
    record("warning", "LOSS_PLATEAU", plateau_signals)
    return findings


def _format_markdown(envelope: dict) -> str:
    summary = envelope["summary"]
    findings = envelope["data"]["findings"]
    if not findings:
        return summary
    fields = ("run_id", "severity", "code", "step", "evidence")
    rows = [[finding[key] for key in fields] for finding in findings]
    table = render_markdown_table(["Run", "Severity", "Code / playbook", "Step", "Evidence"], rows)
    return f"{summary}\n\n{table}"


def main(argv: Optional[list[str]] = None) -> int:
    parser = EnvelopeParser(prog="diagnose", description=__doc__)
    parser.add_argument("paths", nargs="+", help="Run directories, roots or metric files.")
    parser.add_argument("--stale-minutes", type=float, default=30, help="Unfinished run age limit.")
    parser.add_argument("--plateau-window", type=int, default=5, help="Recent loss window length.")
    parser.add_argument("--min-delta", type=float, default=1e-4, help="Minimum loss range.")
    parser.add_argument("--grad-threshold", type=float, default=50, help="Gradient norm limit.")
    parser.add_argument("--tail-lines", type=int, default=200, help="Lines per text-log tail.")
    parser.add_argument("--format", choices=("json", "md"), default="json")
    parser.add_argument("--pretty", action="store_true", help="Indent JSON output.")
    args = parser.parse_args(argv)
    if args.plateau_window < 2 or args.tail_lines < 1:
        parser.error("--plateau-window must be at least 2 and --tail-lines must be positive")
    for name in ("stale_minutes", "min_delta", "grad_threshold"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and nonnegative")

    runs = discover_runs(
        args.paths, stale_minutes=args.stale_minutes, tail_lines=args.tail_lines, limit=None
    )
    findings = []
    for run in runs:
        findings.extend(diagnose_run(run, args.plateau_window, args.min_delta, args.grad_threshold))
    summary = f"Diagnosed {len(runs)} run(s), found {len(findings)} issue(s)."
    warnings = [warning for run in runs for warning in run.warnings]
    envelope = make_envelope("diagnose", True, summary, {"findings": findings}, warnings)
    if args.format == "md":
        print(_format_markdown(envelope))
    else:
        print(dump_json(envelope, pretty=args.pretty))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
