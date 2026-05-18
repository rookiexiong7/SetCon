#!/usr/bin/env python3
"""Offline metrics for SetCon image evaluation results."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from pycocotools import mask as mask_utils
from tqdm import tqdm

from third_parts.sam3.model.box_ops import box_xyxy_to_cxcywh, masks_to_boxes
from third_parts.sam3.train.data.collator import packed_to_padded_naive
from third_parts.sam3.train.matcher import BinaryHungarianMatcherV2

_THREAD_LOCAL = threading.local()
PRIMARY_METRIC_BY_DATASET = {
    "grefcoco": "merged_masks",
    "muse": "hungarian_matching",
    "refcoco": "single_object",
}
METRIC_NAMES = ("hungarian_matching", "merged_masks", "single_object")
METRIC_CHOICES = ("auto", "all", *METRIC_NAMES)


def decode_rle(rle: Dict[str, Any]) -> np.ndarray:
    if not isinstance(rle, dict) or "size" not in rle or "counts" not in rle:
        raise ValueError(f"Invalid RLE mask: {type(rle)}")

    rle_obj = dict(rle)
    counts = rle_obj["counts"]
    if isinstance(counts, str):
        rle_obj["counts"] = counts.encode("utf-8")
    elif not isinstance(counts, bytes):
        raise TypeError(f"Unsupported RLE counts type: {type(counts)}")

    mask = mask_utils.decode(rle_obj)
    if mask.ndim == 3:
        mask = np.any(mask > 0, axis=2)
    return (mask > 0).astype(np.uint8)


def iter_records(results_path: str | Path) -> Iterable[Dict[str, Any]]:
    path = Path(results_path)
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Result row must be a JSON object at {path}:{line_no}")
            yield record


def count_records(results_path: str | Path) -> int:
    with Path(results_path).open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _is_rle(value: Any) -> bool:
    return isinstance(value, dict) and "size" in value and "counts" in value


def extract_target_masks(record: Dict[str, Any]) -> Optional[np.ndarray]:
    gt_rles = record.get("masks")
    if not isinstance(gt_rles, list):
        return None

    gt_rles = [mask for mask in gt_rles if _is_rle(mask)]
    if not gt_rles:
        return None
    return np.stack([decode_rle(rle) for rle in gt_rles], axis=0)


def extract_all_pred_masks_and_logits(
    record: Dict[str, Any],
    conf_threshold: float = float("-inf"),
    max_masks_per_token: int = 9,
) -> Tuple[Optional[np.ndarray], List[float]]:
    token_predictions = record.get("prediction_masks")
    if not isinstance(token_predictions, list) or not token_predictions:
        return None, []

    pred_masks: List[np.ndarray] = []
    pred_logits: List[float] = []
    token_prediction = token_predictions[0]
    if not isinstance(token_prediction, dict):
        return None, []

    masks = token_prediction.get("masks")
    logits = token_prediction.get("logits")
    if not isinstance(masks, list):
        return None, []

    for idx, rle in enumerate(masks[:max_masks_per_token]):
        if not _is_rle(rle):
            continue
        if isinstance(logits, Sequence) and not isinstance(logits, (str, bytes)) and idx < len(logits):
            logit = float(logits[idx])
        else:
            logit = 0.0
        if logit < conf_threshold:
            continue
        pred_masks.append(decode_rle(rle))
        pred_logits.append(logit)

    if not pred_masks:
        return None, []
    return np.stack(pred_masks, axis=0), pred_logits


def _pair_iou(pred: Optional[np.ndarray], gt: Optional[np.ndarray]) -> Tuple[float, float, float]:
    if pred is None and gt is None:
        return 0.0, 0.0, 0.0
    if pred is None:
        union = float((gt > 0).sum())
        return 0.0, union, 0.0
    if gt is None:
        union = float((pred > 0).sum())
        return 0.0, union, 0.0

    inter = float(np.logical_and(pred > 0, gt > 0).sum())
    union = float(np.logical_or(pred > 0, gt > 0).sum())
    iou = 0.0 if union == 0 else inter / union
    return inter, union, iou


class HungarianEvaluator:
    def __init__(self) -> None:
        self.matcher = BinaryHungarianMatcherV2(
            focal=True,
            cost_class=2.0,
            cost_bbox=5.0,
            cost_giou=2.0,
            alpha=0.25,
            gamma=2,
            stable=False,
        )

    def _prepare_matcher_inputs(
        self,
        pred_masks: np.ndarray,
        pred_logits: Sequence[float],
        target_masks: np.ndarray,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        img_h, img_w = target_masks.shape[1], target_masks.shape[2]

        pred_masks_t = torch.from_numpy(pred_masks).float()
        target_masks_t = torch.from_numpy(target_masks).float()
        pred_logits_t = torch.tensor(pred_logits, dtype=torch.float32).unsqueeze(-1)

        pred_boxes_xyxy = masks_to_boxes(pred_masks_t)
        target_boxes_xyxy = masks_to_boxes(target_masks_t)
        scale = torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)

        pred_boxes_cxcywh = box_xyxy_to_cxcywh(pred_boxes_xyxy / scale)
        target_boxes_cxcywh = box_xyxy_to_cxcywh(target_boxes_xyxy / scale)

        outputs = {
            "pred_logits": pred_logits_t.unsqueeze(0),
            "pred_boxes": pred_boxes_cxcywh.unsqueeze(0),
        }
        targets = {
            "num_boxes": torch.tensor([len(target_masks)], dtype=torch.long),
            "boxes": target_boxes_cxcywh,
        }
        targets["boxes_padded"] = packed_to_padded_naive(
            target_boxes_cxcywh.view(-1, 4), targets["num_boxes"]
        )
        return outputs, targets

    def evaluate_sample(
        self,
        pred_masks: Optional[np.ndarray],
        pred_logits: Sequence[float],
        target_masks: Optional[np.ndarray],
        iou_threshold: float = 0.5,
    ) -> Tuple[float, float, float, int, int, int, int]:
        num_gt = 0 if target_masks is None else int(target_masks.shape[0])
        num_pred = 0 if pred_masks is None else int(pred_masks.shape[0])
        pair_count = max(num_gt, num_pred)
        if pair_count == 0:
            return 0.0, 0.0, 0.0, 0, 0, 0, 0

        src_idx = np.array([], dtype=np.int64)
        tgt_idx = np.array([], dtype=np.int64)
        if num_gt > 0 and num_pred > 0:
            outputs, targets = self._prepare_matcher_inputs(pred_masks, pred_logits, target_masks)
            with torch.no_grad():
                if num_gt == 1 and num_pred == 1:
                    src_idx = np.array([0], dtype=np.int64)
                    tgt_idx = np.array([0], dtype=np.int64)
                else:
                    indices = self.matcher(outputs, targets)
                    if indices:
                        _, src, tgt = indices
                        src_idx = src.detach().cpu().numpy() if torch.is_tensor(src) else np.asarray(src)
                        if tgt is None:
                            # BinaryHungarianMatcherV2 omits target indices when
                            # every target is matched and src is already sorted
                            # by target id.
                            tgt_idx = np.arange(len(src_idx), dtype=np.int64)
                        else:
                            tgt_idx = tgt.detach().cpu().numpy() if torch.is_tensor(tgt) else np.asarray(tgt)

        matched_pred = set(src_idx.tolist())
        matched_gt = set(tgt_idx.tolist())
        inter_sum = 0.0
        union_sum = 0.0
        giou_sum = 0.0
        tp_at_thresh = 0

        for pred_i, gt_i in zip(src_idx.tolist(), tgt_idx.tolist()):
            inter, union, iou = _pair_iou(pred_masks[pred_i], target_masks[gt_i])
            inter_sum += inter
            union_sum += union
            giou_sum += iou
            tp_at_thresh += int(iou >= iou_threshold)

        for gt_i in range(num_gt):
            if gt_i not in matched_gt:
                inter, union, iou = _pair_iou(None, target_masks[gt_i])
                inter_sum += inter
                union_sum += union
                giou_sum += iou

        for pred_i in range(num_pred):
            if pred_i not in matched_pred:
                inter, union, iou = _pair_iou(pred_masks[pred_i], None)
                inter_sum += inter
                union_sum += union
                giou_sum += iou

        fp_at_thresh = max(num_pred - tp_at_thresh, 0)
        fn_at_thresh = max(num_gt - tp_at_thresh, 0)
        return inter_sum, union_sum, giou_sum, pair_count, tp_at_thresh, fp_at_thresh, fn_at_thresh


def _merged_metrics_one_sample(
    gt_merged: Optional[np.ndarray],
    pred_merged: Optional[np.ndarray],
) -> Tuple[int, int, float]:
    gt_empty = gt_merged is None or int(gt_merged.sum()) == 0
    pred_empty = pred_merged is None or int(pred_merged.sum()) == 0
    if gt_empty and pred_empty:
        return 0, 0, 1.0

    if gt_merged is None:
        gt_merged = np.zeros_like(pred_merged, dtype=np.uint8)
    if pred_merged is None:
        pred_merged = np.zeros_like(gt_merged, dtype=np.uint8)
    if gt_merged.shape != pred_merged.shape:
        raise ValueError(f"Mask shape mismatch: gt={gt_merged.shape}, pred={pred_merged.shape}")

    inter = int(np.logical_and(gt_merged > 0, pred_merged > 0).sum())
    union = int(np.logical_or(gt_merged > 0, pred_merged > 0).sum())
    iou = 0.0 if union == 0 else float(inter) / float(union)
    return inter, union, iou


def _single_object_one_sample(
    target_masks: Optional[np.ndarray],
    pred_masks: Optional[np.ndarray],
    pred_logits: Sequence[float],
) -> Optional[Tuple[float, float, float]]:
    if target_masks is None or int(target_masks.shape[0]) != 1:
        return None
    if pred_masks is None or int(pred_masks.shape[0]) == 0:
        return _pair_iou(None, target_masks[0])

    if len(pred_logits) >= int(pred_masks.shape[0]):
        top_idx = int(np.argmax(np.asarray(pred_logits[: pred_masks.shape[0]], dtype=np.float32)))
    else:
        top_idx = 0
    return _pair_iou(pred_masks[top_idx], target_masks[0])


def _get_thread_hungarian_evaluator() -> HungarianEvaluator:
    evaluator = getattr(_THREAD_LOCAL, "hungarian_evaluator", None)
    if evaluator is None:
        evaluator = HungarianEvaluator()
        _THREAD_LOCAL.hungarian_evaluator = evaluator
    return evaluator


def evaluate_one_record(
    rec: Dict[str, Any],
    conf_threshold: float = float("-inf"),
    metrics: Sequence[str] = METRIC_NAMES,
) -> Dict[str, float]:
    target_masks = extract_target_masks(rec)
    totals = _empty_totals()
    totals["sample_count"] = 1.0
    pred_cache: Dict[float, Tuple[Optional[np.ndarray], List[float]]] = {}

    def get_predictions(threshold: float) -> Tuple[Optional[np.ndarray], List[float]]:
        if threshold not in pred_cache:
            pred_cache[threshold] = extract_all_pred_masks_and_logits(
                rec,
                conf_threshold=threshold,
            )
        return pred_cache[threshold]

    if "hungarian_matching" in metrics:
        pred_masks, pred_logits = get_predictions(conf_threshold)
        inter, union, giou_sum, pair_count, tp_05, fp_05, fn_05 = (
            _get_thread_hungarian_evaluator().evaluate_sample(
                pred_masks,
                pred_logits,
                target_masks,
                iou_threshold=0.5,
            )
        )
        totals["hungarian_inter"] = inter
        totals["hungarian_union"] = union
        totals["hungarian_giou_sum"] = giou_sum
        totals["hungarian_pair_count"] = float(pair_count)
        totals["hungarian_tp_05"] = float(tp_05)
        totals["hungarian_fp_05"] = float(fp_05)
        totals["hungarian_fn_05"] = float(fn_05)

    if "merged_masks" in metrics:
        pred_masks, _ = get_predictions(conf_threshold)
        gt_merged = np.any(target_masks > 0, axis=0).astype(np.uint8) if target_masks is not None else None
        pred_merged = np.any(pred_masks > 0, axis=0).astype(np.uint8) if pred_masks is not None else None
        merged_inter, merged_union, merged_iou = _merged_metrics_one_sample(gt_merged, pred_merged)
        totals["merged_inter"] = float(merged_inter)
        totals["merged_union"] = float(merged_union)
        totals["merged_giou_sum"] = merged_iou

    if "single_object" in metrics:
        pred_masks, pred_logits = get_predictions(conf_threshold)
        single_obj_metrics = _single_object_one_sample(target_masks, pred_masks, pred_logits)
        if single_obj_metrics is not None:
            single_obj_inter, single_obj_union, single_obj_iou = single_obj_metrics
            totals["single_obj_samples"] = 1.0
            totals["single_obj_inter"] = single_obj_inter
            totals["single_obj_union"] = single_obj_union
            totals["single_obj_giou_sum"] = single_obj_iou

    return totals


def _empty_totals() -> Dict[str, float]:
    return {
        "sample_count": 0.0,
        "hungarian_inter": 0.0,
        "hungarian_union": 0.0,
        "hungarian_giou_sum": 0.0,
        "hungarian_pair_count": 0.0,
        "hungarian_tp_05": 0.0,
        "hungarian_fp_05": 0.0,
        "hungarian_fn_05": 0.0,
        "merged_inter": 0.0,
        "merged_union": 0.0,
        "merged_giou_sum": 0.0,
        "single_obj_samples": 0.0,
        "single_obj_inter": 0.0,
        "single_obj_union": 0.0,
        "single_obj_giou_sum": 0.0,
    }


def _primary_metric_name(dataset: str) -> str:
    if dataset not in PRIMARY_METRIC_BY_DATASET:
        valid = ", ".join(sorted(PRIMARY_METRIC_BY_DATASET))
        raise ValueError(f"Unsupported dataset: {dataset}. Valid choices: {valid}")
    return PRIMARY_METRIC_BY_DATASET[dataset]


def _metrics_to_compute(dataset: str, metric: str = "auto") -> Tuple[str, ...]:
    if metric == "auto":
        return (_primary_metric_name(dataset),)
    if metric == "all":
        return METRIC_NAMES
    if metric in METRIC_NAMES:
        return (metric,)
    valid = ", ".join(METRIC_CHOICES)
    raise ValueError(f"Unsupported metric: {metric}. Valid choices: {valid}")


def summarize_totals(
    totals: Dict[str, float],
    conf_threshold: float,
    dataset: str,
    computed_metrics: Sequence[str],
) -> Dict[str, Any]:
    sample_count = totals["sample_count"]
    dataset_primary_metric_name = _primary_metric_name(dataset)

    hungarian_ciou = totals["hungarian_inter"] / (totals["hungarian_union"] + 1e-10)
    hungarian_giou = totals["hungarian_giou_sum"] / max(totals["hungarian_pair_count"], 1.0)
    hungarian_precision_05 = totals["hungarian_tp_05"] / max(
        totals["hungarian_tp_05"] + totals["hungarian_fp_05"], 1.0
    )
    hungarian_recall_05 = totals["hungarian_tp_05"] / max(
        totals["hungarian_tp_05"] + totals["hungarian_fn_05"], 1.0
    )
    hungarian_f1_05 = (
        2.0
        * hungarian_precision_05
        * hungarian_recall_05
        / max(hungarian_precision_05 + hungarian_recall_05, 1e-10)
    )

    merged_ciou = totals["merged_inter"] / max(totals["merged_union"], 1.0)
    merged_giou = totals["merged_giou_sum"] / max(sample_count, 1.0)
    single_obj_ciou = totals["single_obj_inter"] / (totals["single_obj_union"] + 1e-10)
    single_obj_giou = totals["single_obj_giou_sum"] / max(totals["single_obj_samples"], 1.0)

    metrics = {
        "total_samples": int(sample_count),
        "dataset": dataset,
        "dataset_primary_metric": dataset_primary_metric_name,
        "computed_metrics": list(computed_metrics),
    }

    if "hungarian_matching" in computed_metrics:
        metrics["hungarian_matching"] = {
            "prediction_rule": "first token item, Hungarian matching, pad unmatched masks as empty",
            "confidence_threshold": conf_threshold,
            "cIoU": 100.0 * hungarian_ciou,
            "gIoU": 100.0 * hungarian_giou,
            "F1@0.5": 100.0 * hungarian_f1_05,
            "precision@0.5": 100.0 * hungarian_precision_05,
            "recall@0.5": 100.0 * hungarian_recall_05,
            "tp@0.5": int(totals["hungarian_tp_05"]),
            "fp@0.5": int(totals["hungarian_fp_05"]),
            "fn@0.5": int(totals["hungarian_fn_05"]),
            "total_pairs": int(totals["hungarian_pair_count"]),
            "total_intersection": totals["hungarian_inter"],
            "total_union": totals["hungarian_union"],
        }
    if "merged_masks" in computed_metrics:
        metrics["merged_masks"] = {
            "prediction_rule": "merge masks from the first token item",
            "confidence_threshold": conf_threshold,
            "cIoU": 100.0 * merged_ciou,
            "gIoU": 100.0 * merged_giou,
            "total_intersection": totals["merged_inter"],
            "total_union": totals["merged_union"],
        }
    if "single_object" in computed_metrics:
        metrics["single_object"] = {
            "prediction_rule": "highest-confidence predicted mask, only samples with one GT mask",
            "confidence_threshold": conf_threshold,
            "cIoU": 100.0 * single_obj_ciou,
            "gIoU": 100.0 * single_obj_giou,
            "total_samples": int(totals["single_obj_samples"]),
            "total_intersection": totals["single_obj_inter"],
            "total_union": totals["single_obj_union"],
        }

    return metrics


def evaluate_records(
    records: Iterable[Dict[str, Any]],
    dataset: str,
    num_workers: int = 1,
    conf_threshold: float = float("-inf"),
    metric: str = "auto",
    show_progress: bool = False,
    total: Optional[int] = None,
) -> Dict[str, Any]:
    computed_metrics = _metrics_to_compute(dataset, metric)
    totals = _empty_totals()
    evaluate = partial(
        evaluate_one_record,
        conf_threshold=conf_threshold,
        metrics=computed_metrics,
    )
    num_workers = max(int(num_workers), 1)

    if num_workers == 1:
        parts = map(evaluate, records)
        executor = None
    else:
        executor = ThreadPoolExecutor(max_workers=num_workers)
        parts = executor.map(evaluate, records)

    if show_progress:
        parts = tqdm(parts, total=total, desc="Evaluating", unit="sample")

    try:
        for part in parts:
            for key in totals:
                totals[key] += part[key]
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    return summarize_totals(
        totals,
        conf_threshold=conf_threshold,
        dataset=dataset,
        computed_metrics=computed_metrics,
    )


def evaluate_results(
    results_path: str | Path,
    dataset: str,
    num_workers: int = 1,
    conf_threshold: float = float("-inf"),
    metric: str = "auto",
    show_progress: bool = True,
) -> Dict[str, Any]:
    return evaluate_records(
        iter_records(results_path),
        dataset=dataset,
        num_workers=num_workers,
        conf_threshold=conf_threshold,
        metric=metric,
        show_progress=show_progress,
        total=count_records(results_path),
    )
