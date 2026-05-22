#!/usr/bin/env python3
"""Run SetCon video inference and save PNG masks."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SetCon video inference.")
    parser.add_argument("--model-path", required=True, help="SetCon model path.")
    parser.add_argument("--sam3-ckpt", required=True, help="SAM3 checkpoint.")
    parser.add_argument("--meta-json", required=True, help="Benchmark meta_expressions json.")
    parser.add_argument("--frame-root", required=True, help="Root that contains video frame folders.")
    parser.add_argument("--output-root", required=True, help="Output root: output_root/video_id/expr_id/frame.png.")
    parser.add_argument("--start-frame-index", type=int, default=0)
    parser.add_argument(
        "--propagation-direction",
        choices=["forward", "backward", "both"],
        default="forward",
    )
    parser.add_argument("--max-frame-num-to-track", type=int, default=None)
    parser.add_argument(
        "--mask-save-mode",
        choices=["merge", "instance"],
        default="merge",
        help="merge saves one binary union mask; instance saves one palette label per tracked object.",
    )
    parser.add_argument("--mask-value", type=int, default=255, help="Foreground value used by merge mode.")
    parser.add_argument("--frame-name-template", default="{:05d}")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--failed-tasks-file", default=None)
    return parser.parse_args()


def _model_family(model_path: str | Path) -> str:
    config_path = Path(model_path) / "config.json"
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    architectures = set(config.get("architectures") or [])
    if "SetConChatModelQwen" in architectures or "text_config" in config:
        return "qwen3vl"
    if "SetConChatModel" in architectures or "llm_config" in config:
        return "internvl"
    raise ValueError(f"Unsupported SetCon model config: {config_path}")


def _expression_text(expr_item: Any) -> str:
    if isinstance(expr_item, dict):
        return (
            expr_item.get("exp")
            or expr_item.get("exp_text")
            or expr_item.get("expression")
            or expr_item.get("text")
            or ""
        )
    return str(expr_item)


def _iter_expressions(video_item: Dict[str, Any]) -> Iterable[Tuple[str, str]]:
    expressions = video_item.get("expressions", {})
    if isinstance(expressions, dict):
        items = sorted(
            expressions.items(),
            key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else str(kv[0]),
        )
        for expr_id, expr_item in items:
            text = _expression_text(expr_item)
            if text:
                yield str(expr_id), text
        return

    if isinstance(expressions, list):
        for index, expr_item in enumerate(expressions):
            expr_id = expr_item.get("exp_id", index) if isinstance(expr_item, dict) else index
            text = _expression_text(expr_item)
            if text:
                yield str(expr_id), text


def load_tasks(meta_json: str | Path) -> List[Dict[str, Any]]:
    with Path(meta_json).open("r", encoding="utf-8") as f:
        meta = json.load(f)
    videos = meta.get("videos")
    if not isinstance(videos, dict):
        raise ValueError(f"Invalid meta json, missing videos: {meta_json}")

    tasks: List[Dict[str, Any]] = []
    for video_id in sorted(videos):
        video_item = videos[video_id]
        frames = video_item.get("frames", [])
        for expr_id, text in _iter_expressions(video_item):
            tasks.append(
                {
                    "video_id": str(video_id),
                    "expr_id": str(expr_id),
                    "text": text,
                    "num_frames": len(frames) if isinstance(frames, list) else 0,
                }
            )
    return tasks


def is_finished(output_dir: Path, expected_frames: int) -> bool:
    done_file = output_dir / ".done"
    if done_file.exists():
        return True
    if not output_dir.exists() or expected_frames <= 0:
        return False
    return len(list(output_dir.glob("*.png"))) >= expected_frames


def _list_frame_stems(resource_path: Path) -> List[str]:
    if not resource_path.exists() or not resource_path.is_dir():
        return []
    image_files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
        image_files.extend(resource_path.glob(ext))
    return [path.stem for path in sorted(image_files)]


def _infer_frame_shape(resource_path: Path) -> Tuple[int, int] | None:
    if not resource_path.exists() or not resource_path.is_dir():
        return None
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
        for path in sorted(resource_path.glob(ext)):
            with Image.open(path) as img:
                w, h = img.size
                return h, w
    return None


def _frame_name_from_index(frame_index: int, frame_stems: List[str], frame_name_template: str) -> str:
    if 0 <= frame_index < len(frame_stems):
        return frame_stems[frame_index]
    return frame_name_template.format(frame_index)


def _mask_palette() -> List[int]:
    palette: List[int] = [0, 0, 0]
    for label in range(1, 256):
        value = label
        r = g = b = 0
        for bit in range(8):
            r |= ((value >> 0) & 1) << (7 - bit)
            g |= ((value >> 1) & 1) << (7 - bit)
            b |= ((value >> 2) & 1) << (7 - bit)
            value >>= 3
        palette.extend([r, g, b])
    return palette


def _obj_label(mask_index: int, obj_ids: np.ndarray) -> int:
    if mask_index < len(obj_ids):
        obj_id = int(obj_ids[mask_index])
        if 0 < obj_id < 256:
            return obj_id
    return min(mask_index + 1, 255)


def _compose_frame_mask(outputs: Dict[str, Any], mask_save_mode: str) -> np.ndarray | None:
    raw_masks = np.asarray(outputs.get("out_binary_masks", []))
    if raw_masks.size == 0:
        return None
    if raw_masks.ndim == 2:
        raw_masks = raw_masks[None, ...]

    if mask_save_mode == "merge":
        union_mask = None
        for raw_mask in raw_masks:
            mask = (np.asarray(raw_mask) > 0).astype(np.uint8)
            union_mask = mask if union_mask is None else np.maximum(union_mask, mask)
        return union_mask

    obj_ids = np.asarray(outputs.get("out_obj_ids", []), dtype=np.int64).reshape(-1)
    instance_mask = np.zeros(raw_masks.shape[-2:], dtype=np.uint8)
    for mask_index, raw_mask in enumerate(raw_masks):
        mask = np.asarray(raw_mask) > 0
        instance_mask[mask] = _obj_label(mask_index, obj_ids)
    return instance_mask


def _save_frame_mask(mask: np.ndarray, save_path: Path, mask_save_mode: str, mask_value: int) -> None:
    if mask_save_mode == "merge":
        binary_mask = (np.asarray(mask) > 0).astype(np.uint8)
        Image.fromarray((binary_mask * mask_value).astype(np.uint8), mode="L").save(save_path)
        return

    image = Image.fromarray(np.asarray(mask).astype(np.uint8), mode="P")
    image.putpalette(_mask_palette())
    image.save(save_path)


def run_single_task(
    predictor,
    resource_path: Path,
    text: str,
    output_mask_dir: Path,
    start_frame_index: int,
    propagation_direction: str,
    max_frame_num_to_track: int | None,
    mask_save_mode: str,
    mask_value: int,
    frame_name_template: str,
) -> None:
    session = predictor.handle_request(
        {
            "type": "start_session",
            "resource_path": str(resource_path),
        }
    )
    session_id = session["session_id"]
    try:
        predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": start_frame_index,
                "text": text,
            }
        )

        frame_stems = _list_frame_stems(resource_path)
        frame_shape = _infer_frame_shape(resource_path)
        frame_masks: Dict[int, np.ndarray | None] = {}
        frame_indices = []

        stream_req = {
            "type": "propagate_in_video",
            "session_id": session_id,
            "propagation_direction": propagation_direction,
            "start_frame_index": start_frame_index,
            "max_frame_num_to_track": max_frame_num_to_track,
        }
        for item in predictor.handle_stream_request(stream_req):
            frame_index = item["frame_index"]
            outputs = item["outputs"]
            frame_indices.append(frame_index)
            if outputs is None:
                frame_masks[frame_index] = None
                continue

            frame_mask = _compose_frame_mask(outputs, mask_save_mode)
            frame_masks[frame_index] = frame_mask
            if frame_mask is not None:
                frame_shape = frame_mask.shape

        if frame_shape is None:
            raise RuntimeError(f"Unable to infer frame shape from {resource_path}")

        output_mask_dir.mkdir(parents=True, exist_ok=True)
        mask_value = max(0, min(255, int(mask_value)))
        for frame_index in sorted(set(frame_indices)):
            frame_name = _frame_name_from_index(frame_index, frame_stems, frame_name_template)
            mask = frame_masks.get(frame_index)
            if mask is None:
                mask = np.zeros(frame_shape, dtype=np.uint8)
            _save_frame_mask(mask, output_mask_dir / f"{frame_name}.png", mask_save_mode, mask_value)
    finally:
        predictor.handle_request({"type": "close_session", "session_id": session_id})


def main() -> None:
    args = parse_args()
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError("--shard-id must satisfy 0 <= shard_id < num_shards")
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    failed_tasks_file = (
        Path(args.failed_tasks_file)
        if args.failed_tasks_file
        else output_root.parent / "logs" / f"shard_{args.shard_id}_failed_tasks.jsonl"
    )
    failed_tasks_file.parent.mkdir(parents=True, exist_ok=True)

    tasks = load_tasks(args.meta_json)
    local_tasks = [task for index, task in enumerate(tasks) if index % args.num_shards == args.shard_id]
    print(
        f"[shard {args.shard_id}/{args.num_shards}] total_tasks={len(tasks)} "
        f"local_tasks={len(local_tasks)} meta_json={args.meta_json}",
        flush=True,
    )

    if args.dry_run:
        for index, task in enumerate(local_tasks, 1):
            print(
                f"[dry-run] {index}/{len(local_tasks)} "
                f"{task['video_id']}/{task['expr_id']} text={task['text']}",
                flush=True,
            )
        return

    if _model_family(args.model_path) == "internvl":
        from projects.setcon.hf.models.video_predictor import build_setcon_video_predictor
    else:
        from projects.setcon.hf.models_qwen3vl.video_predictor import build_setcon_video_predictor
    
    predictor = build_setcon_video_predictor(
        bpe_path="third_parts/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        model_name_or_path=args.model_path,
        sam3_checkpoint_path=args.sam3_ckpt,
        device="cuda",
        torch_dtype=torch.bfloat16,
    )

    resume = not args.no_resume
    failed_count = 0
    try:
        for index, task in enumerate(local_tasks, 1):
            video_id = task["video_id"]
            expr_id = task["expr_id"]
            output_mask_dir = output_root / video_id / expr_id
            done_file = output_mask_dir / ".done"

            if resume and is_finished(output_mask_dir, task["num_frames"]):
                print(f"[skip] {index}/{len(local_tasks)} {video_id}/{expr_id}", flush=True)
                continue
            if done_file.exists():
                done_file.unlink()

            print(f"[run] {index}/{len(local_tasks)} {video_id}/{expr_id}", flush=True)
            try:
                run_single_task(
                    predictor=predictor,
                    resource_path=Path(args.frame_root) / video_id,
                    text=task["text"],
                    output_mask_dir=output_mask_dir,
                    start_frame_index=args.start_frame_index,
                    propagation_direction=args.propagation_direction,
                    max_frame_num_to_track=args.max_frame_num_to_track,
                    mask_save_mode=args.mask_save_mode,
                    mask_value=args.mask_value,
                    frame_name_template=args.frame_name_template,
                )
                done_file.write_text("ok\n", encoding="utf-8")
            except Exception as exc:
                failed_count += 1
                error = {
                    "task_index": index,
                    "video_id": video_id,
                    "expr_id": expr_id,
                    "text": task["text"],
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                with failed_tasks_file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(error, ensure_ascii=False) + "\n")
                print(f"[fail] {video_id}/{expr_id}: {exc}", flush=True)
                if args.fail_fast:
                    raise
    finally:
        predictor.shutdown()

    if failed_count:
        print(f"finished with {failed_count} failures, see {failed_tasks_file}", flush=True)
    print(f"done: {output_root}", flush=True)


if __name__ == "__main__":
    main()
