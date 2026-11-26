---
name: trainlens
description: >
  Zero-intrusion, read-only AI training experiment ledger, ablation root-cause
  analyst, and remote GPU inspector for Coding Agents. Use when the user asks to
  inspect, summarize, compare, or debug ML/LLM/CV training runs, build a compact
  experiment ledger, find the best checkpoint, compare mIoU/per-class IoU/loss/eval
  metrics across runs, diff nested configs or correlate regressions with git commits,
  diagnose NaN/Inf/grad explosion/loss plateau/overfitting/CUDA OOM/stalled runs,
  inspect remote GPU servers over SSH (nvidia-smi), or generate a reproducible
  experiment report from TensorBoard (tfevents), HuggingFace (trainer_state.json),
  MMEngine/MMSegmentation (.json), Ultralytics (results.csv), or CSV/JSONL logs.
  适用于训练日志分析、多实验对比、最佳 checkpoint 筛选、mIoU/Loss 掉点排查、NaN/梯度爆炸/OOM 诊断、远程 SSH GPU 巡检和实验复现报告生成。
---

# TrainLens

Deterministic, read-only experiment ledger and root-cause lens for AI training workflows.

## Hard Safety & Token Rules

1. **Read-only by construction**: Never kill, stop, or restart training processes (`kill`, `pkill`, `scancel`, `shutdown`), never delete checkpoints or logs, never modify training source code unless the user explicitly asks for a code patch, and never install packages on remote GPU servers.
2. **Never dump raw logs**: Do not `cat`, `head`, `tail`, or `grep` multi-megabyte training logs or binary `.tfevents` files directly into context. Always invoke the deterministic Python scripts in `scripts/` first and consume their compact JSON output.
3. **Zero external dependencies**: Run scripts with `python3 <skill_dir>/scripts/<script>.py ...`. All core scripts rely strictly on the Python standard library so they work in offline sandboxes and local terminals without `pip install`.

## Standard Operating Workflow

Choose the minimal sequence of scripts needed for the user's request:

### 1. Build the Experiment Ledger (`scan_runs.py`)
Scan a local experiment root (or `assets/demo-logs/` for an offline demo) to get a low-token overview of all runs, statuses, best checkpoints, key metrics, and anomaly flags:
```bash
python3 scripts/scan_runs.py <log_dir> --format json
```
Use `--framework auto|hf|mmengine|ultralytics|tb|csv|jsonl` only when overriding auto-detection.

### 2. Compare Runs, Diff Configs & Align Git Provenance (`compare_runs.py`)
When the user asks why one run beat another, which checkpoint to keep, which hyperparameters changed, or how per-class metrics (e.g., MMSegmentation IoU) shifted:
```bash
python3 scripts/compare_runs.py <run_a> <run_b> --metric <metric_name> --config-diff --per-class
```
Add `--repo <git_repo_path>` to inspect local HEAD and working-tree status alongside each run's recorded commit. When the first and last distinct recorded commits both resolve locally, TrainLens also lists up to 20 changed files between them. Always report:
- Ranked primary metric deltas and best step/checkpoint paths.
- Minimal `config_diff` (only keys that differ across runs).
- Per-class metric deltas when `--per-class` is relevant.

### 3. Diagnose Failures & Correlate with Source Code (`diagnose.py`)
When a run has NaN/Inf, gradient explosion, loss plateau, train/eval divergence (overfitting), stalled log updates, or CUDA OOM:
```bash
python3 scripts/diagnose.py <run_or_dir> --format json
```
Then close the root-cause loop with repository context:
- Inspect `git diff` or the commits aligned to the failing run to connect the exact failing `step` or regression to concrete config/code lines.
- Load `references/failure-playbooks.md` only when a finding emits a `suggested_playbook` code.
- Propose concrete, minimal next-round hyperparameter or code fix suggestions, and wait for user approval before editing any training code.

### 4. Inspect Remote GPU Servers (`ssh_fetch.py`)
When runs live on AutoDL, RunPod, Vast.ai, or a remote Linux server configured in `~/.ssh/config`:
```bash
python3 scripts/ssh_fetch.py --host <ssh_alias> --remote-dir <path> --gpu --logs
```
Report GPU utilization, VRAM headroom, idle-GPU burn warnings, and parsed remote run summaries without modifying anything on the remote host.
GPU inspection uses the host's `nvidia-smi`. Log inspection uses its existing `python3` and a temporary local snapshot. Pass `--cache-dir <local_dir>` to retain the snapshot. The default transfer limit is 2 MiB per file, with at most 200 trailing lines per text log.

### 5. Export Reproducible Report (`render_report.py`)
When the user asks for a PR summary, paper ablation table, or weekly lab update:
```bash
python3 scripts/render_report.py <run_a> <run_b> --metric <metric_name> --include-git --output <report.md>
```
Omit `--output` to preview Markdown on stdout. Choose a destination outside the experiment inputs and use a `.html` suffix for a standalone HTML report. Pass `--repo <git_repo_path>` with `--include-git` to include local repository status alongside the commits recorded by runs.

## On-Demand References

Read these files only when the current task needs them:
- `references/log-formats.md`: Supported framework file layouts, metric key normalization, and pure-stdlib `tfevents` decoding notes.
- `references/failure-playbooks.md`: Deterministic root-cause checklists mapped to `diagnose.py` finding codes (`NAN_LOSS`, `GRAD_EXPLOSION`, `LOSS_PLATEAU`, `OVERFITTING`, `STALLED_LOG`, `CUDA_OOM`).
