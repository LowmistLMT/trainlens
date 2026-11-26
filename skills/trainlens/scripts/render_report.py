#!/usr/bin/env python3
"""Render a compact experiment report to stdout or an external Markdown/HTML file."""

from __future__ import annotations

import html
import json
import os
import re
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

sys.dont_write_bytecode = True
from _core import EnvelopeParser, RunRecord, discover_runs, dump_json, make_envelope, render_markdown_table  # noqa: E402
from compare_runs import build_comparison  # noqa: E402
from diagnose import diagnose_run  # noqa: E402

DEFAULT_TEMPLATE = Path(__file__).resolve().parents[1] / "assets" / "report-template.md"
SLOT = re.compile(r"{{\s*([a-zA-Z_]\w*)\s*}}")


def _literal(value: Any) -> str:
    """Keep log and config strings in a single Markdown cell or line."""
    if value is None:
        return "—"
    if isinstance(value, (list, bool)):
        value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    text = " ".join(str(value).splitlines())
    text = html.escape(text, quote=False)
    # Untrusted names and evidence cannot introduce Markdown images, links or code.
    return text.translate({ord(char): f"&#{ord(char)};" for char in "[]*\x60"})


def _table(headers, rows) -> str:
    return render_markdown_table(
        [_literal(value) for value in headers],
        [[_literal(value) for value in row] for row in rows],
    )


def _report_markdown(template: str, data: dict, warnings: Sequence[str]) -> str:
    ranking = data["ranking"]
    run_ids = data["runs"]
    counts = Counter(run["status"] for run in ranking)
    status = ", ".join(f"{n} {key}" for key, n in sorted(counts.items()))
    metric = _literal(data["primary_metric"])
    population = f"{len(ranking)} {'run' if len(ranking) == 1 else 'runs'} ({status})."
    overview = f"{population} Ranked by {metric} ({data['mode']})."
    if not ranking:
        overview = "No supported experiments found."
    elif not data["primary_metric"]:
        overview = f"{population} No finite metrics available."
    elif ranking[0]["best_value"] is None:
        overview = f"{population} No finite values for {metric}."
    if warnings:
        overview += "\n\nInput warnings\n\n" + "\n".join("- " + _literal(w) for w in warnings)

    sections = {
        "title": _literal(data["title"]),
        "generated_at": data["generated_at"],
        "overview": overview,
        "git_provenance": "",
        "ranking": "No runs to rank.",
        "config_diff": "No differing hyperparameters.",
        "per_class": "",
        "diagnostics": "No diagnostic findings in available observations.",
    }
    if ranking:
        sections["ranking"] = _table(
            ["Rank", "Run", "Framework", "Status", "Best", "Best step", "Last", "Δ vs best",
             "Best checkpoint"],
            [(r["rank"], r["run_id"], r["framework"], r["status"], r["best_value"], r["best_step"],
              r["last_value"], r["delta_vs_best"], (r["best_checkpoint"] or {}).get("path"))
             for r in ranking],
        )
    if data["config_diff"]:
        sections["config_diff"] = _table(
            ["Parameter", *run_ids],
            ([key] + [values[run_id] for run_id in run_ids]
             for key, values in data["config_diff"].items()),
        )
    if data.get("per_class_diff"):
        rows = []
        for name, values in data["per_class_diff"].items():
            row = [name]
            for run_id in run_ids:
                point = values.get(run_id)
                if point:
                    delta = point["delta_vs_base"]
                    row.append(f"{point['value']} ({delta if delta is not None else '—'})")
                else:
                    row.append(None)
            rows.append(row)
        sections["per_class"] = "\n\n".join([
            "## Per-class metrics",
            f"Last value and Δ relative to {_literal(data['baseline_run_id'])}.",
            _table(["Metric", *run_ids], rows),
        ])
    if "git_provenance" in data:
        git = data["git_provenance"]
        parts = ["## Git provenance"]
        if "repo_head" in git:
            state = {True: "dirty", False: "clean", None: "unknown"}[git["repo_dirty"]]
            parts.append(f"Repository HEAD {_literal(git['repo_head'])} ({state}).")
        else:
            parts.append("Repository status unavailable.")
        commits = git["run_commits"]
        if commits:
            parts.append(_table(["Run", "Recorded commit"], [(r, commits.get(r)) for r in run_ids]))
        else:
            parts.append("No recorded commits.")
        if "commit_diff_files" in git:
            diff = git["commit_diff_files"]
            parts.append(_table([f"Changes {diff['from']}..{diff['to']}"],
                                [[line] for line in diff["files"]]))
        sections["git_provenance"] = "\n\n".join(parts)
    if data["findings"]:
        keys = ("run_id", "severity", "code", "step", "evidence")
        sections["diagnostics"] = _table(
            ["Run", "Severity", "Finding / playbook", "Step", "Evidence"],
            ([finding[key] for key in keys] for finding in data["findings"]),
        )

    def replace(match: re.Match) -> str:
        name = match[1]
        if name not in sections:
            raise ValueError(f"Unknown report template field: {name}")
        return sections[name]

    return re.sub(r"\n{3,}", "\n\n", SLOT.sub(replace, template)).strip() + "\n"


def _html_document(markdown: str, title: str) -> str:
    """Render the report template's headings, paragraphs, lists and pipe tables as safe HTML."""
    def text(value: str) -> str:
        return html.escape(html.unescape(value), quote=True)

    def cells(line: str) -> list[str]:
        items = re.findall(r"((?:\\.|[^\\|])*)\|", line.strip()[1:])
        return [text(re.sub(r"\\([\\|])", r"\1", item.strip())) for item in items]

    body = []
    lines = markdown.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        index += 1
        if not line.strip():
            continue
        heading = re.fullmatch(r"(#{1,6})\s+(.+)", line)
        if heading:
            level = len(heading[1])
            body.append(f"<h{level}>{text(heading[2])}</h{level}>")
        elif (line.startswith("|") and index < len(lines)
              and re.fullmatch(r"[|\s:-]+", lines[index])):
            headers = "".join(f'<th scope="col">{cell}</th>' for cell in cells(line))
            index += 1  # The Markdown table's separator row.
            rows = []
            while index < len(lines) and lines[index].startswith("|"):
                row = "".join(f"<td>{cell}</td>" for cell in cells(lines[index]))
                rows.append(f"<tr>{row}</tr>")
                index += 1
            body.append(f'<div class="table"><table><thead><tr>{headers}</tr></thead>'
                        f'<tbody>{"".join(rows)}</tbody></table></div>')
        elif line.startswith(("- ", "* ")):
            items = [f"<li>{text(line[2:])}</li>"]
            while index < len(lines) and lines[index].startswith(("- ", "* ")):
                items.append(f"<li>{text(lines[index][2:])}</li>")
                index += 1
            body.append("<ul>" + "".join(items) + "</ul>")
        else:
            body.append(f"<p>{text(line)}</p>")
    return "\n".join([
        "<!doctype html>", '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        f"<title>{html.escape(title)}</title>",
        "<style>",
        "body{color:#1b2630;background:#fff;font:16px/1.5 system-ui,sans-serif;"
        "max-width:78rem;margin:2rem auto;padding:0 1.25rem}",
        "h1,h2{line-height:1.2}h2{margin-top:2rem}.table{overflow-x:auto}"
        "table{border-collapse:collapse;font-size:.9rem;font-variant-numeric:tabular-nums}"
        "th,td{text-align:left;vertical-align:top;padding:.4rem .65rem;"
        "border:1px solid #b7c0c7}th{background:#eef2f4}",
        "@media print{body{max-width:none;margin:0}.table{overflow:visible}tr{break-inside:avoid}}",
        "</style></head><body><main>", *body, "</main></body></html>", "",
    ])


def _output_path(raw: str, inputs: Sequence[str], runs: Sequence[RunRecord], template: Path) -> Path:
    target = Path(raw).expanduser().absolute()
    if target.suffix.lower() not in (".md", ".html"):
        raise ValueError("Report output must end in .md or .html.")
    if target.is_symlink():
        raise ValueError("A symbolic link cannot be used as the report output.")
    dest = target.resolve()
    if dest.exists() and not dest.is_file():
        raise ValueError("Report output must be a regular file.")
    if dest == template or (dest.exists() and dest.samefile(template)):
        raise ValueError("Report output would overwrite the input template.")

    roots = set()
    for item in [*inputs, *(r.path for r in runs)]:
        root = Path(item).expanduser().absolute()
        if root.is_file():
            root = root.parent
        roots.update((root, root.resolve()))
    if any(dest.is_relative_to(root) or target.is_relative_to(root) for root in roots):
        raise ValueError("Report output must be outside experiment inputs.")

    # An external file may still be an experiment input through an internal link.
    # Compare inodes only if the destination can have another hard link.
    dest_stat = dest.stat() if dest.exists() else None

    def is_source(entry: Path) -> bool:
        if entry.is_symlink() and Path(os.path.realpath(entry)) == dest:
            return True
        return bool(dest_stat and dest_stat.st_nlink > 1 and entry.is_file()
                    and os.path.samestat(entry.stat(), dest_stat))

    # The parser may have opened a log or checkpoint through a linked parent directory.
    for run in runs:
        sources = [Path(run.path, name) for name in run.log_files]
        if run.best_checkpoint and run.best_checkpoint.get("path"):
            sources.append(Path(run.path, run.best_checkpoint["path"]))
        for entry in sources:
            if Path(os.path.realpath(entry)) == dest or is_source(entry):
                raise ValueError("Report output is an experiment source file.")

    real_roots = {root.resolve() for root in roots}
    for root in real_roots:
        if any(parent in real_roots for parent in root.parents):
            continue
        for directory, _, files in os.walk(root):
            if any(is_source(Path(directory, name)) for name in files):
                raise ValueError("Report output is linked to an experiment input.")
    return dest


def _write_report(dest: Path, contents: str) -> None:
    # Stage outside the experiments and replace the directory entry in one operation.
    fd, name = tempfile.mkstemp(dir=dest.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as staged:
            staged.write(contents)
        os.replace(name, dest)
    finally:
        Path(name).unlink(missing_ok=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = EnvelopeParser(prog="render_report", description=__doc__)
    parser.add_argument("paths", nargs="+", help="Experiment roots, runs or metric files.")
    parser.add_argument("--metric", help="Primary comparison metric (default: auto-select).")
    parser.add_argument("--mode", choices=("auto", "max", "min"), default="auto")
    parser.add_argument("--per-class", action="store_true", help="Compare final class metrics.")
    parser.add_argument("--repo", help="Local repository for Git provenance.")
    parser.add_argument("--include-git", action="store_true", help="Include Git provenance.")
    parser.add_argument("--title", default="Training experiment report")
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE,
                        help="Markdown template with named {{fields}}.")
    parser.add_argument("--output", help="External .md or .html output path.")
    parser.add_argument("--format", choices=("json", "md"), help="Stdout format.")
    parser.add_argument("--pretty", action="store_true", help="Indent JSON output.")
    args = parser.parse_args(argv)
    output_format = args.format or ("json" if args.output else "md")
    try:
        template_path = args.template.expanduser().resolve()
        template = template_path.read_text(encoding="utf-8")
        runs = discover_runs(args.paths, limit=None)
        dest = _output_path(args.output, args.paths, runs, template_path) if args.output else None
        data = build_comparison(
            runs, args.metric, args.mode, config_diff=True, per_class=args.per_class,
            repo=args.repo, top_k=None, include_git=args.include_git,
        )
        if not args.include_git:
            for row in data["ranking"]:
                row.pop("git_commit", None)
        data.update(
            title=args.title,
            generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            run_count=len(runs), runs=[run.run_id for run in runs],
            findings=[finding for run in runs for finding in diagnose_run(run)],
        )
        warnings = [warning for run in runs for warning in run.warnings]
        report = _report_markdown(template, data, warnings)
        summary = f"Rendered {len(runs)} run(s) with {len(data['findings'])} finding(s)."
        if dest is not None:
            contents = report
            if dest.suffix.lower() == ".html":
                contents = _html_document(report, args.title)
            _write_report(dest, contents)
            data = {
                "output": str(dest), "run_count": len(runs), "finding_count": len(data["findings"]),
            }
            summary = f"Saved report to {dest}."
        envelope = make_envelope("render_report", True, summary, data, warnings)
    except (OSError, UnicodeError, ValueError, RuntimeError) as exc:
        envelope = make_envelope("render_report", False, str(exc), {}, [])
        print(dump_json(envelope, pretty=args.pretty) if output_format == "json"
              else f"Report error: {exc}")
        return 1

    if output_format == "json":
        print(dump_json(envelope, pretty=args.pretty))
    elif dest is not None:
        print(envelope["summary"])
    else:
        sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
