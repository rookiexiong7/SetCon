#!/usr/bin/env python3
from __future__ import annotations

import argparse
import colorsys
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from transformers import AutoProcessor, AutoTokenizer


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.setcon.evaluation.image_utils import (  # noqa: E402
    build_question,
    mask_to_rle,
    prepare_for_dump,
)
from projects.setcon.hf.models_qwen3vl.modeling_setcon_qwen import (  # noqa: E402
    SetConChatModelQwen,
)

@dataclass
class MaskItem:
    token_idx: int
    label: str
    mask: np.ndarray
    score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SetCon single-image inference.")
    parser.add_argument("--image-path", required=True, help="Image path, relative to repo root or absolute.")
    parser.add_argument("--query-text", required=True, help="Text query for segmentation.")
    parser.add_argument("--model-path", required=True, help="HF model path.")
    parser.add_argument("--output-dir", default="work_dirs/image_inference", help="Output directory.")
    parser.add_argument("--device", default="cuda:0", help="Torch device, for example cuda:0 or cpu.")
    parser.add_argument("--confidence", type=float, default=0.7, help="Mask confidence threshold.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Overlay alpha.")
    return parser.parse_args()


def repo_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else ROOT / path


def load_model(model_path: Path, device: torch.device, confidence: float):
    has_bin_weights = model_path.exists() and any(model_path.glob("pytorch_model*.bin"))
    model = SetConChatModelQwen.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
        use_safetensors=not has_bin_weights,
    )
    model.grounding_encoder.sam3_model.to(device)
    model.grounding_encoder.device = device
    model.grounding_encoder.set_confidence_threshold(confidence)
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
    return model, tokenizer, processor


def prediction_to_masks(prediction: dict[str, Any], confidence: float) -> tuple[str, list[MaskItem], dict[str, Any]]:
    def clean_text(text: str) -> str:
        return (text or "").replace("<|im_end|>", "").replace("<|end|>", "").strip()

    def ref_labels(text: str, count: int) -> list[str]:
        labels = [m.strip() for m in re.findall(r"<ref>\s*(.*?)\s*</ref>", text or "") if m.strip()]
        if len(labels) < count:
            labels.extend([f"token_{idx}" for idx in range(len(labels), count)])
        return labels

    def to_numpy(value: Any) -> np.ndarray | None:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
            if value.dtype == torch.bfloat16:
                value = value.float()
            return value.numpy()
        if isinstance(value, np.ndarray):
            return value
        return None

    def score_list(value: Any) -> list[float]:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().tolist()
        elif isinstance(value, np.ndarray):
            value = value.tolist()
        return [float(v) for v in value] if isinstance(value, list) else []

    def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
        inter = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        return float(inter / union) if union else 0.0

    def nms(items: Sequence[MaskItem], iou_thr: float = 0.75) -> list[MaskItem]:
        keep: list[MaskItem] = []
        for item in sorted(items, key=lambda x: x.score, reverse=True):
            if all(mask_iou(item.mask, kept.mask) <= iou_thr for kept in keep):
                keep.append(item)
        return keep

    def assign(cost: np.ndarray) -> list[tuple[int, int]]:
        try:
            from scipy.optimize import linear_sum_assignment

            rows, cols = linear_sum_assignment(cost)
            return list(zip(rows.tolist(), cols.tolist()))
        except Exception:
            pairs = [(float(cost[i, j]), i, j) for i in range(cost.shape[0]) for j in range(cost.shape[1])]
            pairs.sort(key=lambda x: x[0])
            used_rows: set[int] = set()
            used_cols: set[int] = set()
            out: list[tuple[int, int]] = []
            for _, row, col in pairs:
                if row in used_rows or col in used_cols:
                    continue
                used_rows.add(row)
                used_cols.add(col)
                out.append((row, col))
            return out

    prediction_text = clean_text(str(prediction.get("prediction", "")))
    pred_dict = prediction.get("prediction_dict", [])
    if not isinstance(pred_dict, list):
        pred_dict = []

    labels = ref_labels(prediction_text, len(pred_dict))
    token_items: list[list[MaskItem]] = []
    raw_counts: list[int] = []
    filtered_counts: list[int] = []
    raw_scores: list[list[float]] = []

    for token_idx, item in enumerate(pred_dict):
        masks = to_numpy(item.get("masks")) if isinstance(item, dict) else None
        scores = score_list(item.get("scores", item.get("logits", []))) if isinstance(item, dict) else []
        raw_scores.append(scores)
        if masks is None or masks.size == 0:
            raw_counts.append(0)
            filtered_counts.append(0)
            token_items.append([])
            continue
        if masks.ndim == 2:
            masks = masks[None, ...]

        raw_counts.append(int(masks.shape[0]))
        cur_items: list[MaskItem] = []
        for mask_idx, mask in enumerate(masks):
            score = scores[mask_idx] if mask_idx < len(scores) else 1.0
            if score >= confidence:
                cur_items.append(
                    MaskItem(
                        token_idx=token_idx,
                        label=labels[token_idx],
                        mask=np.asarray(mask) > 0,
                        score=float(score),
                    )
                )
        filtered_counts.append(len(cur_items))
        token_items.append(cur_items)

    global_items = token_items[0] if token_items else []
    if not global_items:
        final_items = [item for items in token_items for item in items]
    else:
        sub_items = nms([item for items in token_items[1:] for item in items])
        if not sub_items:
            final_items = global_items
        else:
            cost = np.ones((len(global_items), len(sub_items)), dtype=np.float32)
            iou_mat = np.zeros_like(cost)
            for row, global_item in enumerate(global_items):
                for col, sub_item in enumerate(sub_items):
                    iou = mask_iou(global_item.mask, sub_item.mask)
                    iou_mat[row, col] = iou
                    cost[row, col] = 1.0 - iou

            matched = {row: col for row, col in assign(cost) if iou_mat[row, col] >= 0.15}
            final_items = []
            for row, global_item in enumerate(global_items):
                if row in matched:
                    sub_item = sub_items[matched[row]]
                    final_items.append(
                        MaskItem(
                            token_idx=sub_item.token_idx,
                            label=sub_item.label,
                            mask=global_item.mask,
                            score=global_item.score,
                        )
                    )
                else:
                    final_items.append(global_item)

    stats = {
        "raw_token_mask_counts": raw_counts,
        "filtered_token_mask_counts": filtered_counts,
        "raw_token_scores": raw_scores,
    }
    return prediction_text, final_items, stats


def render_overlay(image: Image.Image, masks: Sequence[MaskItem], alpha: float) -> Image.Image:
    def color(token_idx: int, instance_idx: int) -> np.ndarray:
        hue = (0.03 + token_idx * 0.6180339887498949 + instance_idx * 0.10) % 1.0
        rgb = colorsys.hsv_to_rgb(hue, 0.55, 0.92)
        return np.array([int(c * 255) for c in rgb], dtype=np.float32)

    def centroid(mask: np.ndarray) -> tuple[int, int] | None:
        ys, xs = np.where(mask)
        return (int(xs.mean()), int(ys.mean())) if ys.size else None

    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    out = base.copy()
    token_seen: dict[int, int] = {}
    labels_to_draw: list[tuple[str, tuple[int, int], tuple[int, int, int]]] = []

    for display_idx, item in enumerate(masks, start=1):
        mask = np.asarray(item.mask) > 0
        if mask.ndim != 2 or mask.shape != out.shape[:2]:
            continue

        instance_idx = token_seen.get(item.token_idx, 0)
        token_seen[item.token_idx] = instance_idx + 1
        mask_color = color(item.token_idx, instance_idx)
        out[mask] = (1.0 - alpha) * out[mask] + alpha * mask_color

        edge = np.zeros_like(mask, dtype=bool)
        edge[:-1, :] |= mask[:-1, :] != mask[1:, :]
        edge[:, :-1] |= mask[:, :-1] != mask[:, 1:]
        out[edge] = mask_color

        center = centroid(mask)
        if center is not None:
            labels_to_draw.append((str(display_idx), center, tuple(int(c) for c in mask_color.tolist())))

    result = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(result)
    font = ImageFont.load_default()
    for label, (x, y), label_color in labels_to_draw:
        square = 26
        x0 = max(x - square // 2, 0)
        y0 = max(y - square // 2, 0)
        x1 = min(x0 + square, result.width - 1)
        y1 = min(y0 + square, result.height - 1)
        fill_color = tuple(int(0.85 * c + 0.15 * 255) for c in label_color)
        text_color = (0, 0, 0) if sum(fill_color) > 430 else (255, 255, 255)
        draw.rectangle((x0, y0, x1, y1), fill=fill_color)
        draw.rectangle((x0, y0, x1, y1), outline=label_color, width=1)
        bbox = draw.textbbox((0, 0), label, font=font)
        text_x = x0 + ((x1 - x0) - (bbox[2] - bbox[0])) // 2
        text_y = y0 + ((y1 - y0) - (bbox[3] - bbox[1])) // 2
        draw.text((text_x, text_y), label, fill=text_color, font=font)
    return result


def save_results(
    output_dir: Path,
    image_path: Path,
    overlay: Image.Image,
    payload: dict[str, Any],
) -> tuple[Path, Path]:
    case_dir = output_dir / image_path.stem
    case_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = case_dir / "overlay.png"
    result_path = case_dir / "result.json"
    overlay.save(overlay_path)
    result_path.write_text(json.dumps(prepare_for_dump(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return overlay_path, result_path


@torch.inference_mode()
def run_inference(
    model,
    tokenizer,
    processor,
    image: Image.Image,
    query_text: str,
    confidence: float,
    alpha: float,
    device: torch.device,
) -> tuple[Image.Image, str, list[MaskItem], dict[str, Any]]:
    prompt_text = build_question(query_text)
    model.grounding_encoder.set_confidence_threshold(confidence)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        prediction = model.predict_forward(
            image=image,
            text=prompt_text,
            tokenizer=tokenizer,
            processor=processor,
        )
    prediction_text, masks, stats = prediction_to_masks(prediction, confidence=confidence)
    overlay = render_overlay(image, masks, alpha=alpha)
    stats["prompt_text"] = prompt_text
    return overlay, prediction_text, masks, stats


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_grad_enabled(False)

    image_path = repo_path(args.image_path)
    model_path = repo_path(args.model_path)
    output_dir = repo_path(args.output_dir)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    long_side = max(image.width, image.height)
    if long_side > 1024:
        scale = 1024 / long_side
        new_size = (round(image.width * scale), round(image.height * scale))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    model, tokenizer, processor = load_model(model_path, device=device, confidence=args.confidence)
    overlay, prediction_text, masks, stats = run_inference(
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        image=image,
        query_text=args.query_text,
        confidence=args.confidence,
        alpha=args.alpha,
        device=device,
    )

    payload = {
        "image_path": args.image_path,
        "model_path": args.model_path,
        "output_dir": args.output_dir,
        "query_text": args.query_text,
        "prompt_text": stats["prompt_text"],
        "prediction_text": prediction_text,
        "confidence": args.confidence,
        "alpha": args.alpha,
        "mask_count": len(masks),
        "raw_token_mask_counts": stats["raw_token_mask_counts"],
        "filtered_token_mask_counts": stats["filtered_token_mask_counts"],
        "raw_token_scores": stats["raw_token_scores"],
        "masks": [
            {
                "id": idx,
                "token_id": item.token_idx,
                "ref_label": item.label,
                "score": item.score,
                "rle": mask_to_rle(item.mask)[0],
            }
            for idx, item in enumerate(masks, start=1)
        ],
    }
    overlay_path, result_path = save_results(output_dir, image_path, overlay, payload)

    print("prediction_text:", prediction_text)
    print("raw_token_mask_counts:", stats["raw_token_mask_counts"])
    print("filtered_token_mask_counts:", stats["filtered_token_mask_counts"])
    print("mask_count:", len(masks))
    print("overlay_path:", overlay_path)
    print("result_path:", result_path)


if __name__ == "__main__":
    main()

"""
python demo.py --image-path assets/animals.jpg --query-text "If a zookeeper wants to identify the main herbivores currently feeding in this scene, which objects should be detected?" --model-path path/to/model
python demo.py --image-path assets/room.jpg --query-text "If someone wanted to pull down all the curtains in the room, which ones would they have to address and can you describe them?" --model-path path/to/model
"""