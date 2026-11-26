#!/usr/bin/env python3
"""Scan local training directories and emit a low-token experiment ledger (read-only)."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional

sys.dont_write_bytecode = True
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _core import (  # noqa: E402
    EnvelopeParser,
    discover_runs,
    dump_json,
    make_envelope,
    render_markdown_table,
    serialize_run_summary,
)


def _format_markdown(runs_payload: List[dict], summary: str) -> str:
    if not runs_payload:
        return f"{summary}\n"
    rows = []
    for r in runs_payload:
        metrics_str = ", ".join(
            f"{k}={v.get('best', v.get('last'))}"
            for k, v in list(r["metrics"].items())[:4]
        )
        anomalies = ",".join(r["anomalies"]) if r["anomalies"] else "-"
        rows.append([r["run_id"], r["framework"], r["status"], r["step_max"], metrics_str, anomalies])
    table = render_markdown_table(
        ["Run", "Framework", "Status", "Step", "Key Metrics", "Anomalies"], rows
    )
    return f"{summary}\n\n{table}\n"


def build_parser() -> EnvelopeParser:
    parser = EnvelopeParser(
        prog="scan_runs",
        description="Scan local training runs into a compact, read-only experiment ledger."
    )
    parser.add_argument("paths", nargs="+", help="Experiment root directories or log files.")
    parser.add_argument(
        "--framework",
        choices=("auto", "hf", "mmengine", "ultralytics", "tb", "csv", "jsonl"),
        default="auto",
        help="Framework filter (default: auto).",
    )
    parser.add_argument(
        "--stale-minutes",
        type=float,
        default=30.0,
        help="Minutes since last log update before marking an unfinished run stalled.",
    )
    parser.add_argument(
        "--limit", type=int, default=20, help="Maximum number of runs to return (default: 20)."
    )
    parser.add_argument(
        "--per-class",
        action="store_true",
        help="Include per-class metrics (e.g. IoU.*) in scan output.",
    )
    parser.add_argument(
        "--format", choices=("json", "md"), default="json", help="Output format (default: json)."
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    runs = discover_runs(
        args.paths,
        framework=args.framework,
        stale_minutes=args.stale_minutes,
        limit=args.limit,
    )
    runs_payload = [serialize_run_summary(r, include_per_class=args.per_class) for r in runs]
    status_counts = Counter(r["status"] for r in runs_payload)
    counts_str = ", ".join(f"{cnt} {st}" for st, cnt in sorted(status_counts.items()))
    summary = (
        f"Scanned {len(runs_payload)} run(s) ({counts_str})."
        if runs_payload
        else "No supported training runs found under the provided path(s)."
    )

    warnings = [warning for run in runs for warning in run.warnings]
    envelope = make_envelope("scan_runs", True, summary, {"runs": runs_payload}, warnings)
    if args.format == "md":
        sys.stdout.write(_format_markdown(envelope["data"]["runs"], envelope["summary"]))
        if warnings:
            print("\n" + "\n".join(warnings))
    else:
        sys.stdout.write(dump_json(envelope, pretty=args.pretty) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
