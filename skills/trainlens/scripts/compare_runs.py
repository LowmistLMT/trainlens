#!/usr/bin/env python3
"""Compare local training runs using a single metric and optional ablation tables."""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from typing import Any, Optional, Sequence

sys.dont_write_bytecode = True

from _core import (  # noqa: E402
    EnvelopeParser,
    RunRecord,
    align_git_provenance,
    compute_config_diff,
    discover_runs,
    dump_json,
    infer_metric_mode,
    json_safe_number,
    make_envelope,
    render_markdown_table,
)

PREFERRED_METRICS = ("mIoU", "eval_loss", "val_loss", "metrics/mAP50-95(B)", "loss")


def _auto_metric(runs: Sequence[RunRecord]) -> Optional[str]:
    # A metric observed only as NaN/Inf cannot produce a ranking.
    coverage: Counter[str] = Counter()
    for run in runs:
        for name, points in run.series.items():
            if any(math.isfinite(value) for _, value in points):
                coverage[name] += 1
    for name in PREFERRED_METRICS:
        if name in coverage:
            return name
    # For custom logs, prefer the metric available across the most runs.
    return min(coverage, key=lambda name: (-coverage[name], name), default=None)


def _ranking(
    runs: Sequence[RunRecord], metric: Optional[str], mode: Optional[str], top_k: Optional[int]
) -> list[dict[str, Any]]:
    rows = []
    select_best = min if mode == "min" else max
    for run in runs:
        points = []
        if metric is not None:
            points = run.series.get(metric) or run.per_class.get(metric, [])
        best_step, best = select_best(
            ((step, value) for step, value in points if math.isfinite(value)),
            key=lambda point: point[1],
            default=(None, None),
        )
        rows.append({
            "rank": None,
            "run_id": run.run_id,
            "framework": run.framework,
            "status": run.status,
            "step_max": run.step_max,
            "best_value": best,
            "best_step": best_step,
            "last_value": json_safe_number(points[-1][1]) if points else None,
            "delta_vs_best": None,
            "best_checkpoint": run.best_checkpoint,
            "git_commit": run.config.get("git_commit"),
        })

    def sort_key(row: dict[str, Any]) -> tuple[bool, float]:
        best = row["best_value"]
        if best is None:
            return True, 0.0
        return False, best if mode == "min" else -best

    rows.sort(key=sort_key)
    rows = rows[:top_k]
    leader = rows[0]["best_value"] if rows else None
    previous = None
    rank = 0
    for position, row in enumerate(rows, start=1):
        value = row["best_value"]
        if value is not None:
            if value != previous:
                rank = position
            row["rank"] = rank
            row["delta_vs_best"] = json_safe_number(value - leader)
            row["best_value"] = json_safe_number(value)
            previous = value
    return rows


def _per_class_diff(runs: Sequence[RunRecord]) -> dict[str, dict[str, Any]]:
    """Compare the last observation of each class metric against the first input run."""
    result = {}
    for name in sorted({name for run in runs for name in run.per_class}):
        base_points = runs[0].per_class.get(name)
        base = base_points[-1][1] if base_points else None
        values = {}
        for run in runs:
            points = run.per_class.get(name)
            if points:
                value = points[-1][1]
                delta = None
                if base is not None and math.isfinite(base) and math.isfinite(value):
                    delta = json_safe_number(value - base)
                values[run.run_id] = {"value": json_safe_number(value), "delta_vs_base": delta}
        result[name] = values
    return result


def _display(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, (list, bool)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def _markdown(envelope: dict[str, Any], run_ids: Sequence[str]) -> str:
    data = envelope["data"]
    sections = [envelope["summary"]]
    if data["ranking"]:
        rows = []
        for entry in data["ranking"]:
            cells = [entry[key] for key in (
                "rank", "run_id", "framework", "status", "step_max", "best_value", "best_step",
                "last_value", "delta_vs_best",
            )]
            cells.extend([(entry["best_checkpoint"] or {}).get("path"), entry["git_commit"]])
            rows.append([_display(value) for value in cells])
        sections.append(render_markdown_table(
            ["Rank", "Run", "Framework", "Status", "Step", "Best", "Best step", "Last",
             "Δ vs best", "Checkpoint", "Commit"], rows,
        ))
    if "config_diff" in data:
        rows = [
            [key] + [_display(values[run_id]) for run_id in run_ids]
            for key, values in data["config_diff"].items()
        ]
        sections.append("## Config Diff")
        sections.append(render_markdown_table(["Key", *run_ids], rows) if rows
                        else "No differing hyperparameters.")
    if data.get("per_class_diff"):
        rows = []
        for key, values in data["per_class_diff"].items():
            cells = []
            for run_id in run_ids:
                item = values.get(run_id)
                if item is None:
                    cells.append("—")
                else:
                    cells.append(f"{_display(item['value'])} ({_display(item['delta_vs_base'])})")
            rows.append([key, *cells])
        sections.extend([
            f"## Per-class metrics (last value and Δ vs {data['baseline_run_id']})",
            render_markdown_table(["Metric", *run_ids], rows),
        ])
    git = data.get("git_provenance", {})
    if "repo_head" in git:
        state = {True: "dirty", False: "clean", None: "unknown"}[git["repo_dirty"]]
        sections.append(f"Git HEAD {_display(git['repo_head'])} ({state}).")
    if "commit_diff_files" in git:
        diff = git["commit_diff_files"]
        sections.append(render_markdown_table(
            [f"Changes {diff['from']}..{diff['to']}"], [[line] for line in diff["files"]],
        ))
    sections.extend(envelope["warnings"])
    return "\n\n".join(sections)


def build_comparison(
    runs: Sequence[RunRecord],
    metric: Optional[str] = None,
    mode: str = "auto",
    *,
    config_diff: bool = False,
    per_class: bool = False,
    repo: Optional[str] = None,
    top_k: Optional[int] = 10,
    include_git: Optional[bool] = None,
) -> dict[str, Any]:
    """Build comparison data for both terminal output and exported reports."""
    metric = metric or _auto_metric(runs)
    direction: Optional[str] = mode
    if mode == "auto":
        direction = infer_metric_mode(metric) if metric else None
    data: dict[str, Any] = {
        "primary_metric": metric,
        "mode": direction,
        "ranking": _ranking(runs, metric, direction, top_k),
    }
    if config_diff:
        diff = compute_config_diff(runs)
        diff.pop("git_commit", None)
        data["config_diff"] = diff
    if per_class:
        data["baseline_run_id"] = runs[0].run_id if runs else None
        data["per_class_diff"] = _per_class_diff(runs)
    if include_git is None:
        include_git = bool(repo) or any(run.config.get("git_commit") for run in runs)
    if include_git:
        data["git_provenance"] = align_git_provenance(runs, repo)
    return data


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = EnvelopeParser(prog="compare_runs", description=__doc__)
    parser.add_argument("paths", nargs="+", help="Run directories or experiment roots.")
    parser.add_argument("--metric", help="Metric to rank (default: auto-select).")
    parser.add_argument("--mode", choices=("auto", "max", "min"), default="auto")
    parser.add_argument("--config-diff", action="store_true", help="Show changed hyperparameters.")
    parser.add_argument("--per-class", action="store_true", help=(
        "Compare final class metrics to the first discovered run. Pass its path first to select it."
    ))
    parser.add_argument("--repo", help="Local Git repository for provenance.")
    parser.add_argument("--top-k", type=int, default=10, help="Maximum ranking rows (default: 10).")
    parser.add_argument("--format", choices=("json", "md"), default="json")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON.")
    args = parser.parse_args(argv)
    if args.top_k < 1:
        parser.error("--top-k must be positive")

    runs = discover_runs(args.paths, limit=None)
    data = build_comparison(
        runs, args.metric, args.mode, config_diff=args.config_diff, per_class=args.per_class,
        repo=args.repo, top_k=args.top_k,
    )
    metric = data["primary_metric"]

    warnings = [warning for run in runs for warning in run.warnings]
    if not runs:
        summary = "No supported training runs found under the provided path(s)."
    elif data["ranking"][0]["best_value"] is None:
        summary = f"Compared {len(runs)} run(s)."
        warnings.append(f"No finite observations for {metric or 'any metric'}.")
    else:
        summary = f"Compared {len(runs)} run(s) by {metric} ({data['mode']})."
    envelope = make_envelope("compare_runs", True, summary, data, warnings)
    if args.format == "md":
        print(_markdown(envelope, [run.run_id for run in runs]))
    else:
        print(dump_json(envelope, pretty=args.pretty))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
