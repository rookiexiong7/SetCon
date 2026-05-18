#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python"
fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

MODEL="${MODEL:-saved_models/qwen3_8b_0410_descv7_1query_video_semantic_merge_desc_full_run2_30930_bf16}"
if [[ "$MODEL" != /* ]]; then
  MODEL="$ROOT/$MODEL"
fi
MODEL_NAME="$(basename "$MODEL")"

IMAGE_OUTPUT_DIR="${IMAGE_OUTPUT_DIR:-work_dirs/evaluation/ours}"
IMAGE_GPUS="${IMAGE_GPUS:-4}"
IMAGE_EVAL_WORKERS="${IMAGE_EVAL_WORKERS:-16}"
CONFIDENCE="${CONFIDENCE:-0.7}"

GREFCOCO_ANN="${GREFCOCO_ANN:-setcon_training_datasets/image_eval/grefcoco/grefcoco_val.jsonl}"
MUSE_ANN="${MUSE_ANN:-setcon_training_datasets/image_eval/muse/muse_val.jsonl}"
REFCOCO_ANN="${REFCOCO_ANN:-setcon_training_datasets/image_eval/refcoco/refcoco_val.jsonl}"

VIDEO_OUTPUT_BASE="${VIDEO_OUTPUT_BASE:-work_dirs/evaluation_video/$MODEL_NAME}"
VIDEO_SHARDS="${VIDEO_SHARDS:-4}"
VIDEO_GPU_IDS="${VIDEO_GPU_IDS:-0,1,2,3}"
VIDEO_SCORE_WORKERS="${VIDEO_SCORE_WORKERS:-16}"

DAVIS_META="${DAVIS_META:-/inspire/hdd/global_user/zhangzhixiong-253108100067/dataset/ReferVOS/Ref-DAVIS/Ref-DAVIS/meta_expressions/valid/meta_expressions.json}"
DAVIS_FRAMES="${DAVIS_FRAMES:-/inspire/hdd/global_user/zhangzhixiong-253108100067/dataset/ReferVOS/Ref-DAVIS/Ref-DAVIS/valid/JPEGImages}"
DAVIS_GT="${DAVIS_GT:-/inspire/hdd/global_user/zhangzhixiong-253108100067/dataset/ReferVOS/Ref-DAVIS/Ref-DAVIS/valid/Annotations_exp}"

SECVOS_META="${SECVOS_META:-/inspire/hdd/global_user/zhangzhixiong-253108100067/dataset/ReferVOS/SeCVOS/meta_expressions.json}"
SECVOS_FRAMES="${SECVOS_FRAMES:-/inspire/hdd/global_user/zhangzhixiong-253108100067/dataset/ReferVOS/SeCVOS/JPEGImages}"
SECVOS_GT="${SECVOS_GT:-/inspire/hdd/global_user/zhangzhixiong-253108100067/dataset/ReferVOS/SeCVOS/Annotations}"

usage() {
  cat <<'EOF'
Usage:
  bash tools/eval_30930_bf16_benchmarks.sh [target]

Targets:
  all            Run image + video inference + video score. Default.
  image          Run grefcoco, muse, and refcoco val.
  grefcoco       Run grefcoco val only.
  muse           Run muse val only.
  refcoco        Run refcoco val only.
  video          Run DAVIS and SeCVOS inference + score.
  davis          Run DAVIS inference + score.
  secvos         Run SeCVOS inference + score.
  video_infer    Run DAVIS and SeCVOS inference only.
  video_score    Score existing DAVIS and SeCVOS predictions only.
  davis_infer    Run DAVIS inference only.
  davis_score    Score existing DAVIS predictions only.
  secvos_infer   Run SeCVOS inference only.
  secvos_score   Score existing SeCVOS predictions only.

Useful env overrides:
  MODEL=/path/to/model
  IMAGE_GPUS=4
  VIDEO_SHARDS=4 VIDEO_GPU_IDS=0,1,2,3
  VIDEO_OUTPUT_BASE=work_dirs/evaluation_video/custom_name
EOF
}

log() {
  printf '[%(%F %T)T] %s\n' -1 "$*"
}

require_model() {
  if [[ ! -d "$MODEL" ]]; then
    echo "Model directory not found: $MODEL" >&2
    exit 1
  fi
}

run_image_one() {
  local dataset="$1"
  local ann_file="$2"

  log "image eval: $dataset"
  if [[ "$IMAGE_GPUS" -gt 1 ]]; then
    bash projects/setcon/evaluation/launch_dist.sh \
      projects/setcon/evaluation/image_eval.py \
      "$MODEL" \
      "$IMAGE_GPUS" \
      --ann_file "$ann_file" \
      --dataset "$dataset" \
      --output_dir "$IMAGE_OUTPUT_DIR" \
      --confidence "$CONFIDENCE" \
      --eval-num-workers "$IMAGE_EVAL_WORKERS"
  else
    "$PYTHON_BIN" projects/setcon/evaluation/image_eval.py \
      "$MODEL" \
      --ann_file "$ann_file" \
      --dataset "$dataset" \
      --output_dir "$IMAGE_OUTPUT_DIR" \
      --confidence "$CONFIDENCE" \
      --eval-num-workers "$IMAGE_EVAL_WORKERS"
  fi
}

run_image_all() {
  run_image_one grefcoco "$GREFCOCO_ANN"
  run_image_one muse "$MUSE_ANN"
  run_image_one refcoco "$REFCOCO_ANN"
}

gpu_id_for_shard() {
  local shard_id="$1"
  local gpu_ids_csv="$VIDEO_GPU_IDS"
  local old_ifs="$IFS"
  local gpu_ids=()
  IFS=',' read -r -a gpu_ids <<< "$gpu_ids_csv"
  IFS="$old_ifs"

  if [[ "${#gpu_ids[@]}" -eq 0 ]]; then
    echo "0"
    return
  fi
  echo "${gpu_ids[$((shard_id % ${#gpu_ids[@]}))]}"
}

run_video_infer_one() {
  local name="$1"
  local meta_json="$2"
  local frame_root="$3"
  local output_root="$VIDEO_OUTPUT_BASE/$name/results"
  local log_dir="$VIDEO_OUTPUT_BASE/$name/logs"

  mkdir -p "$output_root" "$log_dir"
  log "video inference: $name, shards=$VIDEO_SHARDS, output=$output_root"

  local pids=()
  local shard_id
  for ((shard_id = 0; shard_id < VIDEO_SHARDS; shard_id++)); do
    local gpu_id
    gpu_id="$(gpu_id_for_shard "$shard_id")"
    local log_file="$log_dir/shard_${shard_id}.log"
    log "start $name shard $shard_id on gpu $gpu_id, log=$log_file"
    "$PYTHON_BIN" projects/setcon/evaluation/video_eval.py \
      --model-path "$MODEL" \
      --meta-json "$meta_json" \
      --frame-root "$frame_root" \
      --output-root "$output_root" \
      --num-shards "$VIDEO_SHARDS" \
      --shard-id "$shard_id" \
      --gpu-id "$gpu_id" \
      > "$log_file" 2>&1 &
    pids+=("$!")
  done

  local status=0
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done

  if [[ "$status" -ne 0 ]]; then
    echo "Video inference failed for $name. Check logs in $log_dir" >&2
    exit "$status"
  fi
  log "video inference done: $name"
}

run_davis_infer() {
  run_video_infer_one davis "$DAVIS_META" "$DAVIS_FRAMES"
}

run_secvos_infer() {
  run_video_infer_one secvos "$SECVOS_META" "$SECVOS_FRAMES"
}

run_video_score_one() {
  local name="$1"
  local gt_root="$2"
  local pred_root="$VIDEO_OUTPUT_BASE/$name/results"

  log "video score: $name"
  "$PYTHON_BIN" projects/setcon/evaluation/video_score.py \
    --gt-root "$gt_root" \
    --pred-root "$pred_root" \
    --num-processes "$VIDEO_SCORE_WORKERS"
}

run_davis_score() {
  run_video_score_one davis "$DAVIS_GT"
}

run_secvos_score() {
  run_video_score_one secvos "$SECVOS_GT"
}

run_video_infer_all() {
  run_davis_infer
  run_secvos_infer
}

run_video_score_all() {
  run_davis_score
  run_secvos_score
}

run_video_all() {
  run_davis_infer
  run_davis_score
  run_secvos_infer
  run_secvos_score
}

main() {
  local target="${1:-all}"
  require_model

  case "$target" in
    all)
      run_image_all
      run_video_all
      ;;
    image)
      run_image_all
      ;;
    grefcoco)
      run_image_one grefcoco "$GREFCOCO_ANN"
      ;;
    muse)
      run_image_one muse "$MUSE_ANN"
      ;;
    refcoco)
      run_image_one refcoco "$REFCOCO_ANN"
      ;;
    video)
      run_video_all
      ;;
    davis)
      run_davis_infer
      run_davis_score
      ;;
    secvos)
      run_secvos_infer
      run_secvos_score
      ;;
    video_infer)
      run_video_infer_all
      ;;
    video_score)
      run_video_score_all
      ;;
    davis_infer)
      run_davis_infer
      ;;
    davis_score)
      run_davis_score
      ;;
    secvos_infer)
      run_secvos_infer
      ;;
    secvos_score)
      run_secvos_score
      ;;
    -h|--help|help)
      usage
      ;;
    *)
      usage
      echo "Unknown target: $target" >&2
      exit 2
      ;;
  esac
}

main "$@"
