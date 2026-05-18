#!/usr/bin/env python3
"""Offline metrics for video mask folders.

Supported folder layouts:

1. DAVIS-like label maps:
   root/video_id/frame.png

2. One binary-mask folder per expression/object:
   root/video_id/expr_id/frame.png
"""

from __future__ import annotations

import math
import os
import time
from collections import defaultdict
from multiprocessing import Pool
from os import path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
import tqdm
from PIL import Image
from skimage.morphology import disk


def _load_mask(mask_path: str) -> np.ndarray:
    return np.array(Image.open(mask_path))


def _spatial_shape(mask: np.ndarray) -> Tuple[int, int]:
    if mask.ndim < 2:
        raise ValueError(f"Mask must be at least 2D, got shape={mask.shape}")
    return int(mask.shape[0]), int(mask.shape[1])


class VideoEvaluator:
    def __init__(self, gt_root: str, pred_root: str, skip_first_and_last: bool = False) -> None:
        self.gt_root = gt_root
        self.pred_root = pred_root
        self.skip_first_and_last = skip_first_and_last

    def __call__(self, video_id: str) -> Tuple[str, Dict[str, float], Dict[str, float]]:
        to_evaluate, is_binary_folder_format = self.scan_video_folder(video_id)

        eval_results = []
        for frames, object_id, gt_path, pred_path in to_evaluate:
            if self.skip_first_and_last:
                frames = frames[1:-1]

            evaluator = Evaluator(name=video_id, obj_id=object_id)
            for frame_name in frames:
                gt_array, pred_array = self.get_gt_and_pred(
                    gt_path, pred_path, frame_name, is_binary_folder_format
                )
                evaluator.feed_frame(mask=pred_array, gt=gt_array)

            iou, boundary_f = evaluator.conclude()
            eval_results.append((object_id, iou, boundary_f))

        if is_binary_folder_format:
            return video_id, *self.consolidate(eval_results)

        assert len(eval_results) == 1
        return video_id, eval_results[0][1], eval_results[0][2]

    def get_gt_and_pred(
        self,
        gt_path: str,
        pred_path: str,
        frame_name: str,
        is_binary_folder_format: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        gt_mask_path = path.join(gt_path, frame_name)
        pred_mask_path = path.join(pred_path, frame_name)
        if not path.exists(pred_mask_path):
            raise FileNotFoundError(f"{pred_mask_path} not found")

        gt_array = _load_mask(gt_mask_path)
        pred_array = _load_mask(pred_mask_path)
        gt_h, gt_w = _spatial_shape(gt_array)
        pred_h, pred_w = _spatial_shape(pred_array)
        if (gt_h, gt_w) != (pred_h, pred_w):
            print(f"[WARN] resize prediction to GT shape: {gt_mask_path}, {pred_mask_path}")
            pred_array = cv2.resize(
                pred_array,
                (gt_w, gt_h),
                interpolation=cv2.INTER_NEAREST,
            )

        if is_binary_folder_format:
            if len(np.unique(gt_array)) > 2:
                raise ValueError(
                    f"Expected a binary GT mask in {gt_mask_path}, "
                    "because folder layout is root/video_id/expr_id/frame.png."
                )
            if len(np.unique(pred_array)) > 2:
                raise ValueError(
                    f"Expected a binary predicted mask in {pred_mask_path}, "
                    "because folder layout is root/video_id/expr_id/frame.png."
                )
            gt_array = gt_array > 0
            pred_array = pred_array > 0

        return gt_array, pred_array

    def scan_video_folder(self, video_id: str) -> Tuple[List[Tuple[List[str], str | None, str, str]], bool]:
        vid_gt_path = path.join(self.gt_root, video_id)
        vid_pred_path = path.join(self.pred_root, video_id)
        entries = sorted(os.listdir(vid_gt_path))
        png_entries = [name for name in entries if name.endswith(".png")]

        if entries and len(png_entries) == len(entries):
            return [(png_entries, None, vid_gt_path, vid_pred_path)], False

        to_evaluate = []
        for object_id in entries:
            obj_gt_path = path.join(vid_gt_path, object_id)
            obj_pred_path = path.join(vid_pred_path, object_id)
            if not path.isdir(obj_gt_path):
                continue
            frames = sorted(name for name in os.listdir(obj_gt_path) if name.endswith(".png"))
            to_evaluate.append((frames, object_id, obj_gt_path, obj_pred_path))
        return to_evaluate, True

    def consolidate(self, eval_results) -> Tuple[Dict[str, float], Dict[str, float]]:
        iou_output = {}
        boundary_f_output = {}
        for object_id, iou, boundary_f in eval_results:
            assert len(iou) == 1
            key = next(iter(iou))
            iou_output[object_id] = iou[key]
            boundary_f_output[object_id] = boundary_f[key]
        return iou_output, boundary_f_output


def _seg2bmap(seg: np.ndarray, width: int | None = None, height: int | None = None) -> np.ndarray:
    seg = seg.astype(bool)
    assert np.atleast_3d(seg).shape[2] == 1

    src_h, src_w = seg.shape[:2]
    width = src_w if width is None else width
    height = src_h if height is None else height

    src_ratio = float(src_w) / float(src_h)
    dst_ratio = float(width) / float(height)
    if width > src_w or height > src_h or abs(dst_ratio - src_ratio) > 0.01:
        raise ValueError(f"Cannot convert {src_w}x{src_h} seg to {width}x{height} bmap.")

    east = np.zeros_like(seg)
    south = np.zeros_like(seg)
    south_east = np.zeros_like(seg)

    east[:, :-1] = seg[:, 1:]
    south[:-1, :] = seg[1:, :]
    south_east[:-1, :-1] = seg[1:, 1:]

    boundary = (seg ^ east) | (seg ^ south) | (seg ^ south_east)
    boundary[-1, :] = seg[-1, :] ^ east[-1, :]
    boundary[:, -1] = seg[:, -1] ^ south[:, -1]
    boundary[-1, -1] = 0

    if src_w == width and src_h == height:
        return boundary

    boundary_map = np.zeros((height, width), dtype=boundary.dtype)
    for x in range(src_w):
        for y in range(src_h):
            if boundary[y, x]:
                dst_y = 1 + math.floor((y - 1) + height / src_h)
                dst_x = 1 + math.floor((x - 1) + width / src_w)
                boundary_map[dst_y, dst_x] = 1
    return boundary_map


def get_iou(intersection: float, pixel_sum: float) -> float:
    if intersection == pixel_sum:
        assert intersection == 0
        return 1.0
    return float(intersection) / float(pixel_sum - intersection)


class Evaluator:
    def __init__(self, boundary: float = 0.008, name: str | None = None, obj_id: str | None = None):
        self.boundary = boundary
        self.name = name
        self.obj_id = obj_id
        self.objects_in_gt = set()
        self.objects_in_masks = set()
        self.object_iou = defaultdict(list)
        self.boundary_f = defaultdict(list)
        self.no_object_iou = []
        self.no_object_boundary_f = []
        self.n_frames = 0

    def feed_frame(self, mask: np.ndarray, gt: np.ndarray) -> None:
        self.n_frames += 1

        gt_objects = np.unique(gt)
        gt_objects = gt_objects[gt_objects != 0].tolist()
        mask_objects = np.unique(mask)
        mask_objects = mask_objects[mask_objects != 0].tolist()

        if len(gt_objects) == 0:
            if len(mask_objects) == 0:
                self.no_object_iou.append(1.0)
                self.no_object_boundary_f.append(1.0)
            else:
                self.no_object_iou.append(0.0)
                self.no_object_boundary_f.append(0.0)

        self.objects_in_gt.update(gt_objects)
        self.objects_in_masks.update(mask_objects)
        all_objects = self.objects_in_gt.union(self.objects_in_masks)

        bound_pix = int(np.ceil(self.boundary * np.linalg.norm(mask.shape)))
        boundary_disk = disk(bound_pix)

        for object_id in all_objects:
            obj_mask = mask == object_id
            obj_gt = gt == object_id

            self.object_iou[object_id].append(
                get_iou((obj_mask * obj_gt).sum(), obj_mask.sum() + obj_gt.sum())
            )

            mask_boundary = _seg2bmap(obj_mask)
            gt_boundary = _seg2bmap(obj_gt)
            mask_dilated = cv2.dilate(mask_boundary.astype(np.uint8), boundary_disk)
            gt_dilated = cv2.dilate(gt_boundary.astype(np.uint8), boundary_disk)

            gt_match = gt_boundary * mask_dilated
            fg_match = mask_boundary * gt_dilated

            n_fg = np.sum(mask_boundary)
            n_gt = np.sum(gt_boundary)
            if n_fg == 0 and n_gt > 0:
                precision = 1
                recall = 0
            elif n_fg > 0 and n_gt == 0:
                precision = 0
                recall = 1
            elif n_fg == 0 and n_gt == 0:
                precision = 1
                recall = 1
            else:
                precision = np.sum(fg_match) / float(n_fg)
                recall = np.sum(gt_match) / float(n_gt)

            if precision + recall == 0:
                boundary_f = 0
            else:
                boundary_f = 2 * precision * recall / (precision + recall)
            self.boundary_f[object_id].append(boundary_f)

    def conclude(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        all_iou = {}
        all_boundary_f = {}

        if len(self.objects_in_gt) == 0:
            all_iou[0] = np.mean(self.no_object_iou) * 100
            all_boundary_f[0] = np.mean(self.no_object_boundary_f) * 100
            return all_iou, all_boundary_f

        for object_id in self.objects_in_gt:
            n_record_frames = len(self.object_iou[object_id])
            if n_record_frames < self.n_frames:
                n_pad = self.n_frames - n_record_frames
                self.object_iou[object_id] = self.object_iou[object_id] + [1.0] * n_pad
                self.boundary_f[object_id] = self.boundary_f[object_id] + [1.0] * n_pad

            all_iou[object_id] = np.mean(self.object_iou[object_id]) * 100
            all_boundary_f[object_id] = np.mean(self.boundary_f[object_id]) * 100

        return all_iou, all_boundary_f


def _video_dirs(root: str) -> List[str]:
    return sorted(name for name in os.listdir(root) if path.isdir(path.join(root, name)))


def _maybe_annotations_root(root: str, mask_root: str) -> Tuple[str, List[str]]:
    entries = os.listdir(root)
    mask_entries = os.listdir(mask_root)
    videos = _video_dirs(root)
    if len(entries) == len(mask_entries) or "Annotations" not in entries:
        return root, videos

    annotations_root = path.join(root, "Annotations")
    annotation_entries = os.listdir(annotations_root)
    if annotation_entries and ".png" not in annotation_entries[0]:
        return annotations_root, _video_dirs(annotations_root)
    return root, videos


def _evaluate_dataset(
    gt_root: str,
    mask_root: str,
    strict: bool,
    pool: Pool,
    verbose: bool,
    skip_first_and_last: bool,
) -> Iterable[Tuple[str, Dict[str, float], Dict[str, float]]]:
    gt_root, gt_videos = _maybe_annotations_root(gt_root, mask_root)
    mask_videos = _video_dirs(mask_root)

    if not strict:
        videos = sorted(set(gt_videos) & set(mask_videos))
    else:
        gt_extras = set(gt_videos) - set(mask_videos)
        mask_extras = set(mask_videos) - set(gt_videos)
        if gt_extras:
            print(f"Videos that are in {gt_root} but not in {mask_root}: {gt_extras}")
        if mask_extras:
            print(f"Videos that are in {mask_root} but not in {gt_root}: {mask_extras}")
        if gt_extras or mask_extras:
            raise SystemExit("Validation failed. Exiting.")
        videos = gt_videos

    if verbose:
        print(f"In dataset {gt_root}, we are evaluating on {len(videos)} videos: {videos}")

    evaluator = VideoEvaluator(gt_root, mask_root, skip_first_and_last=skip_first_and_last)
    if verbose:
        return tqdm.tqdm(pool.imap(evaluator, videos), total=len(videos))
    return pool.map(evaluator, videos)


def _format_results(
    object_metrics: Dict[str, Tuple[Dict[str, float], Dict[str, float]]],
    global_jf: float,
    global_j: float,
    global_f: float,
) -> str:
    max_len = max(*[len(name) for name in object_metrics.keys()], len("Global score"))
    out = f'{"sequence":<{max_len}},{"obj":>3}, {"J&F":>4}, {"J":>4}, {"F":>4}\n'
    out += f'{"Global score":<{max_len}},{"":>3}, {global_jf:.1f}, {global_j:.1f}, {global_f:.1f}\n'
    for name, (iou, boundary_f) in object_metrics.items():
        for object_id in iou.keys():
            j = iou[object_id]
            f = boundary_f[object_id]
            jf = (j + f) / 2
            out += f"{name:<{max_len}},{str(object_id):>3}, {jf:>4.1f}, {j:>4.1f}, {f:>4.1f}\n"
    return out


def benchmark(
    gt_roots: List[str],
    mask_roots: List[str],
    strict: bool = True,
    num_processes: int | None = None,
    *,
    verbose: bool = True,
    skip_first_and_last: bool = False,
):
    assert len(gt_roots) == len(mask_roots)
    single_dataset = len(gt_roots) == 1

    if verbose:
        if skip_first_and_last:
            print("Skipping first and last frame for video evaluation.")
        else:
            print("Evaluating all frames for video evaluation.")

    pool = Pool(num_processes)
    start = time.time()
    async_results = []
    dataset_results = []

    success = False
    try:
        for gt_root, mask_root in zip(gt_roots, mask_roots):
            if single_dataset:
                dataset_results.append(
                    _evaluate_dataset(
                        gt_root,
                        mask_root,
                        strict,
                        pool,
                        verbose,
                        skip_first_and_last,
                    )
                )
            else:
                gt_root, gt_videos = _maybe_annotations_root(gt_root, mask_root)
                mask_videos = _video_dirs(mask_root)
                if not strict:
                    videos = sorted(set(gt_videos) & set(mask_videos))
                else:
                    gt_extras = set(gt_videos) - set(mask_videos)
                    mask_extras = set(mask_videos) - set(gt_videos)
                    if gt_extras or mask_extras:
                        if gt_extras:
                            print(f"Videos that are in {gt_root} but not in {mask_root}: {gt_extras}")
                        if mask_extras:
                            print(f"Videos that are in {mask_root} but not in {gt_root}: {mask_extras}")
                        raise SystemExit("Validation failed. Exiting.")
                    videos = gt_videos
                if verbose:
                    print(f"In dataset {gt_root}, we are evaluating on {len(videos)} videos: {videos}")
                async_results.append(
                    pool.map_async(
                        VideoEvaluator(
                            gt_root,
                            mask_root,
                            skip_first_and_last=skip_first_and_last,
                        ),
                        videos,
                    )
                )
        pool.close()
        all_global_jf, all_global_j, all_global_f = [], [], []
        all_object_metrics = []
        for dataset_idx, mask_root in enumerate(mask_roots):
            if single_dataset:
                results = dataset_results[dataset_idx]
            else:
                results = async_results[dataset_idx].get()

            all_iou = []
            all_boundary_f = []
            object_metrics = {}
            for name, iou, boundary_f in results:
                all_iou.extend(list(iou.values()))
                all_boundary_f.extend(list(boundary_f.values()))
                object_metrics[name] = (iou, boundary_f)

            global_j = np.array(all_iou).mean()
            global_f = np.array(all_boundary_f).mean()
            global_jf = (global_j + global_f) / 2
            out_string = _format_results(object_metrics, global_jf, global_j, global_f)

            if verbose:
                print(out_string.replace(",", " "), end="")
                print("\nSummary:")
                print(f"Global score: J&F: {global_jf:.1f} J: {global_j:.1f} F: {global_f:.1f}")
                print(f"Time taken: {time.time() - start:.2f}s")

            result_path = path.join(mask_root, "results.csv")
            print(f"Saving the results to {result_path}")
            with open(result_path, "w", encoding="utf-8") as f:
                f.write(out_string)

            all_global_jf.append(global_jf)
            all_global_j.append(global_j)
            all_global_f.append(global_f)
            all_object_metrics.append(object_metrics)

        success = True
        return all_global_jf, all_global_j, all_global_f, all_object_metrics
    finally:
        if not success:
            pool.terminate()
        pool.join()
