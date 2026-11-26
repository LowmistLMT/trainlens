# Training Failure Playbooks

Load this playbook only when `diagnose.py` flags a matching `suggested_playbook` code.

## `NAN_LOSS`
- **Signal**: Any `loss`, `train_loss`, or `eval_loss` value is `NaN`, `Inf`, or `-Inf`.
- **Root-cause checklist**:
  1. Check `config_diff` for learning rate spikes, disabled `max_grad_norm` / gradient clipping, or `fp16` without dynamic loss scaling (prefer `bf16` on Ampere+ GPUs).
  2. Inspect `git diff` for custom loss functions (missing `eps` in `log`/`sqrt`/division, unstable softmax, label index out of `ignore_index` bounds).
  3. Inspect data pipeline changes around the failing `step` (corrupted sample, empty mask in segmentation, zero-length prompt/completion in SFT).
- **Safe next-round suggestion**: Reduce LR by 2x–5x, enable gradient clipping (`max_grad_norm=1.0`), switch `fp16 -> bf16`, or resume from the last healthy checkpoint prior to the spike.

## `GRAD_EXPLOSION`
- **Signal**: `grad_norm` spikes above threshold (default `> 50.0` or `> 10x` running median) or becomes `NaN`/`Inf`.
- **Root-cause checklist**:
  1. Verify warmup ratio/steps (`warmup_ratio` or `warmup_iters`) and scheduler type.
  2. Check LoRA `lora_alpha / lora_r` scaling ratio or unscaled residual branches.
  3. Verify effective batch size (`per_device_batch_size * grad_accum * world_size`) did not shrink without scaling LR down.

## `CUDA_OOM`
- **Signal**: Tail of log contains `CUDA out of memory`, `torch.cuda.OutOfMemoryError`, `CUBLAS_STATUS_ALLOC_FAILED`, or `NCCL` timeout/watchdog termination.
- **Root-cause checklist**:
  1. Check `config_diff` for increases in `batch_size`, `max_seq_len`, `crop_size`, or newly disabled `gradient_checkpointing`.
  2. Check whether OOM happened at the first validation step (validation batch size or full-resolution sliding-window inference without `torch.no_grad()`).
  3. On multi-GPU runs, check if `nvidia-smi` shows a lingering zombie process occupying VRAM on another rank.
- **Safe next-round suggestion**: Halve `per_device_batch_size` and double `gradient_accumulation_steps` to preserve effective batch size, enable gradient checkpointing, or use `bf16`/8-bit optimizer.

## `LOSS_PLATEAU`
- **Signal**: Primary loss or eval metric fails to improve by at least `--min-delta` over the last `--plateau-window` logged points.
- **Root-cause checklist**:
  1. Check if `lr` already decayed to near `0` too early (mismatched `max_steps` / `max_iters`).
  2. Check if backbone or adapter parameters were accidentally frozen (`requires_grad=False`).
  3. Compare train loss vs. eval metric: if train loss is still falling while eval plateaus, see `OVERFITTING`.

## `OVERFITTING`
- **Signal**: Training loss decreases meaningfully while validation loss increases (or validation mIoU/accuracy drops) across recent evaluations.
- **Root-cause checklist**:
  1. Check `config_diff` for reduced `weight_decay`, lower dropout, removed data augmentation, or higher LoRA rank / epoch count.
  2. Point the user directly to `best_checkpoint` from `scan_runs.py` / `compare_runs.py` so they do not waste compute training further past peak generalization.

## `STALLED_LOG`
- **Signal**: Run is neither marked completed nor crashed, but the latest log file has not been modified for longer than `--stale-minutes`.
- **Root-cause checklist**:
  1. On remote GPUs, run `ssh_fetch.py --gpu` to verify whether GPU utilization dropped to `0%` while VRAM remains allocated (classic DataLoader / NCCL deadlock).
  2. Check if the process is stuck in slow evaluation, checkpoint upload, or disk-full I/O stall.
