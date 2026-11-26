# Supported Training Log Formats

TrainLens prioritizes deterministic, structured files written by mainstream training frameworks.

## 1. HuggingFace / TRL / LLaMA-Factory (`hf`)
- **Primary metric file**: `trainer_state.json` (in run root or latest `checkpoint-*` subdirectory).
  - Reads `log_history` (`step`, `epoch`, `loss`, `eval_loss`, `learning_rate`, `grad_norm`, `eval_*`).
  - Extracts `best_metric` and `best_model_checkpoint` when present.
- **Config files**: `training_args.json`, `config.json`, `adapter_config.json`, or `train_args.yaml` / `args.json`.
- **Optional tail-only text log**: `train.log` or `trainer_log.txt` (only last N lines inspected by `diagnose.py` for OOM/traceback signals).

## 2. MMEngine / MMSegmentation / MMDetection (`mmengine`)
- **Primary metric file**: `vis_data/scalars.json` (JSON Lines, one record per train or val step) or `*.log.json`.
  - Train entries contain `step`/`iter`, `loss`, `lr`, `grad_norm`, `memory`, `time`.
  - Validation entries contain `mIoU`, `mAcc`, `aAcc`, and per-class keys such as `IoU.road`, `IoU.car`, `Acc.road` (normalized into `per_class` maps).
- **Config files** include `config.json`, `meta.json`, and `config.yaml`. Local scans flatten nested JSON keys and read simple YAML settings.
- **Checkpoints**: Detects `best_*.pth` filenames to infer best checkpoint path and step.

## 3. Ultralytics YOLO (`ultralytics`)
- **Primary metric file**: `results.csv` (column whitespace trimmed; maps `epoch`, `train/box_loss`, `metrics/mAP50(B)`, `metrics/mAP50-95(B)`, `lr/pg0`, etc.).
- **Config files**: `args.yaml` (parsed via safe lightweight YAML subset parser without PyYAML).

## 4. TensorBoard (`tb`)
- **Primary metric file**: `events.out.tfevents.*`.
- Parsed with `scripts/_tfevents.py` using pure Python standard library (`struct` + TFRecord record framing + minimal Protobuf wire-format decoding for `Event.step`, `Event.wall_time`, `Summary.Value.tag`, and `simple_value` / float tensor).
- Requires zero `tensorflow` or `tensorboard` installation.

## 5. Generic CSV / JSONL (`csv`, `jsonl`)
- **Primary metric files**: `metrics.csv`, `log.csv`, `history.csv`, `metrics.jsonl`, `scalars.jsonl`, `events.jsonl`.
- Step column auto-detected from `step`, `global_step`, `iter`, `iteration`, or `epoch` (falling back to 1-based row index).

## Metric Normalization Rules
- Lower-is-better metrics (auto `mode=min`): `loss`, `train_loss`, `eval_loss`, `val_loss`, `perplexity`, `ppl`, `wer`, `cer`, `mae`, `rmse`, `mse`, plus any metric ending in `_loss` or `/loss`.
- Higher-is-better metrics (auto `mode=max`): `mIoU`, `mAcc`, `aAcc`, `mAP`, `accuracy`, `acc`, `f1`, `auc`, `bleu`, `rouge`, `reward`, `pass@1`, plus `IoU.*` and `metrics/mAP*`.
