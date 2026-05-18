#!/usr/bin/env python3
"""Run SetCon image inference and score the dumped results."""

from __future__ import annotations

import argparse
import copy
import datetime
import json
import os
import shutil
import sys
from pathlib import Path

import torch
import tqdm
from transformers import AutoProcessor, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from projects.setcon.evaluation.image_utils import (
    load_image,
    load_image_records,
    mask_to_rle,
    prepare_for_dump,
    serialize_result_for_dump,
)
from projects.setcon.evaluation.image_metrics import METRIC_CHOICES, evaluate_records
from projects.setcon.evaluation.utils import (
    _init_dist_pytorch,
    collect_results_cpu,
    get_dist_info,
    get_rank,
)


def parse_args():
    parser = argparse.ArgumentParser(description="SetCon image evaluation")
    parser.add_argument("model_path", help="HF model path.")
    parser.add_argument("--ann_file", required=True, help="Image annotation file or directory.")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["grefcoco", "muse", "refcoco"],
        help="Dataset family; selects the primary metric: grefcoco=merged, muse=hungarian, refcoco=single.",
    )
    parser.add_argument("--image_root", default=None, help="Optional root for relative image paths.")
    parser.add_argument("--output_dir", default="work_dirs/evaluation/ours", help="Output directory.")
    parser.add_argument("--confidence", type=float, default=0.7, help="Confidence threshold.")
    parser.add_argument("--metric", choices=METRIC_CHOICES, default="auto", help="Metric to compute. auto uses the dataset primary metric.")
    parser.add_argument("--eval-num-workers", type=int, default=16, help="Offline metric worker threads.")
    parser.add_argument("--launcher", choices=["none", "pytorch"], default="none", help="Job launcher.")
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


def _load_model(model_path, device, confidence=0.5):
    from projects.setcon.hf.models_qwen3vl.modeling_setcon_qwen import SetConChatModelQwen

    model_path_obj = Path(model_path)
    has_bin_weights = any(model_path_obj.glob("pytorch_model*.bin"))
    wrapper = SetConChatModelQwen.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
        use_safetensors=not has_bin_weights,
    )
    wrapper.grounding_encoder.sam3_model.to(device)
    wrapper.grounding_encoder.device = device
    wrapper.grounding_encoder.set_confidence_threshold(confidence)
    wrapper.to(device)
    wrapper.eval()
    return wrapper


def _prediction_token_item(target_pred):
    raw_mask = target_pred.get("masks")
    raw_box = target_pred.get("boxes")
    raw_logit = target_pred.get("scores")
    if raw_mask is None or len(raw_mask) == 0:
        return {"masks": [], "boxes": None, "logits": None}

    return {
        "masks": mask_to_rle(raw_mask),
        "boxes": raw_box.tolist() if raw_box is not None else None,
        "logits": raw_logit.tolist() if raw_logit is not None else None,
    }


def _predict_one_sample(model, tokenizer, processor, data_batch, sample_idx):
    text = data_batch.get("text", "")
    image = data_batch.pop("image")
    prediction = dict(copy.deepcopy(data_batch))

    print(f"Processing sample {sample_idx}, text: {text}")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        pred = model.predict_forward(image=image, text=text, tokenizer=tokenizer, processor=processor)

    pred_dict = pred.get("prediction_dict") or []
    pred_text = pred.get("prediction", "")
    print(f"Pred text: {pred_text}")

    prediction_masks = []
    for target_pred in pred_dict:
        if isinstance(target_pred, dict):
            prediction_masks.append(_prediction_token_item(target_pred))

    prediction.update(
        {
            "prediction_masks": prediction_masks,
            "prediction_text": pred_text,
        }
    )
    return prediction


def main():
    args = parse_args()

    if args.launcher != "none":
        local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
        torch.cuda.set_device(local_rank)
        _init_dist_pytorch("nccl", timeout=datetime.timedelta(minutes=30))
        rank, world_size = get_dist_info()
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    device = torch.device(f"cuda:{local_rank}")
    model = _load_model(args.model_path, device=device, confidence=args.confidence)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    records = load_image_records(args.ann_file, image_root=args.image_root)
    local_records = [record for index, record in enumerate(records) if index % world_size == rank]

    results = []
    for sample_idx, record in enumerate(tqdm.tqdm(local_records)):
        data_batch = dict(record)
        data_batch["image"] = load_image(data_batch["image_path"])
        results.append(_predict_one_sample(model, tokenizer, processor, data_batch, sample_idx))

    model_name = Path(args.model_path.rstrip("/")).name
    ann_name = Path(args.ann_file.rstrip("/")).stem
    output_dir = Path(args.output_dir) / model_name / ann_name
    output_dir.mkdir(parents=True, exist_ok=True)

    tmpdir = f"./dist_eval_{ann_name}_{model_name.replace('/', '')}"
    results = collect_results_cpu(results, len(records), tmpdir=tmpdir)

    if get_rank() != 0:
        return

    if os.path.exists(tmpdir):
        shutil.rmtree(tmpdir, ignore_errors=True)

    results_path = output_dir / "results.json"
    with results_path.open("w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(serialize_result_for_dump(res), ensure_ascii=False) + "\n")
    print(f"Results saved to {output_dir}")

    metrics = evaluate_records(
        results,
        dataset=args.dataset,
        num_workers=args.eval_num_workers,
        conf_threshold=args.confidence,
        metric=args.metric,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(prepare_for_dump(metrics), f, ensure_ascii=False, indent=2)
    with (output_dir / "metrics.txt").open("w", encoding="utf-8") as f:
        f.write(json.dumps(prepare_for_dump(metrics), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
