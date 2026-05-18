"""Small helpers shared by image evaluation scripts."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image, ImageOps
from pycocotools import mask as mask_utils


def mask_to_rle(mask: Any):
    array = mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else mask
    array = np.asarray(array)
    if array.ndim == 2:
        array = array[None, ...]
    rles = []
    for m in array:
        rle = mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))
        rle["counts"] = rle["counts"].decode()
        rles.append(rle)
    return rles


def prepare_for_dump(data: Any):
    if isinstance(data, np.ndarray):
        return data.tolist()
    if isinstance(data, bytes):
        return data.decode("utf-8")
    if isinstance(data, (np.integer, np.floating)):
        return data.item()
    if isinstance(data, torch.Tensor):
        return prepare_for_dump(data.detach().cpu().numpy())
    if isinstance(data, list):
        return [prepare_for_dump(item) for item in data]
    if isinstance(data, dict):
        return {k: prepare_for_dump(v) for k, v in data.items()}
    return data


def serialize_result_for_dump(result: dict):
    serializable = copy.deepcopy(result)
    return prepare_for_dump(serializable)


def normalize_image_roots(image_root: Optional[Sequence[str] | str]) -> List[str]:
    if image_root is None:
        return []
    if isinstance(image_root, str):
        candidates = image_root.split(os.pathsep)
    else:
        candidates = list(image_root)
    return [str(Path(root).expanduser()) for root in candidates if str(root).strip()]


def resolve_ann_files(ann_file: str) -> List[Path]:
    path = Path(ann_file).expanduser()
    if not path.exists() and path.suffix == "":
        path = path.with_suffix(".jsonl")
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(path.glob("*.jsonl"))
        if files:
            return files
        image_files = sorted((path / "image").glob("*.jsonl"))
        if image_files:
            return image_files
    raise FileNotFoundError(f"Could not find image annotation file(s): {ann_file}")


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid annotation row at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Annotation row must be an object at {path}:{line_no}")
            yield row


def resolve_image_path(image_path: str, image_roots: Sequence[str]) -> str:
    raw = Path(image_path).expanduser()
    if raw.is_absolute() or raw.exists():
        return str(raw)
    for root in image_roots:
        candidate = Path(root) / image_path
        if candidate.exists():
            return str(candidate)
    if image_roots:
        return str(Path(image_roots[0]) / image_path)
    return image_path


def is_rle(mask: Any) -> bool:
    return isinstance(mask, dict) and "size" in mask and "counts" in mask


def build_question(text: str) -> str:
    text = text[:-1] if text.endswith(".") else text
    return f'Given the question or description "{text}", Can you segment the object(s) it refers to in this image?'


def load_image_records(ann_file: str, image_root: str | None = None) -> List[Dict[str, Any]]:
    image_roots = normalize_image_roots(image_root)
    records: List[Dict[str, Any]] = []
    for ann_path in resolve_ann_files(ann_file):
        for row in iter_jsonl(ann_path):
            index = len(records)
            image = str(row.get("image", ""))
            text = str(row.get("text", ""))
            masks = row.get("masks")
            records.append(
                {
                    "image": image,
                    "image_path": resolve_image_path(image, image_roots),
                    "text": build_question(text),
                    "raw_text": text,
                    "masks": [mask for mask in masks if is_rle(mask)] if isinstance(masks, list) else [],
                    "segments": row.get("segments", []),
                    "img_id": str(row.get("id", index)),
                    "source_ann_file": str(ann_path),
                }
            )
    if not records:
        raise ValueError(f"No samples found in {ann_file}")
    return records


def load_image(image_path: str) -> Image.Image:
    image = Image.open(image_path)
    return ImageOps.exif_transpose(image).convert("RGB")
