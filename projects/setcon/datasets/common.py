import copy
import json
import os.path as osp

import mmengine

def load_jsonl(path):
    """Load one JSON object per non-empty line."""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f'Invalid JSONL at {path}:{line_no}: {e}') from e
    return data


def join_ref_names(names):
    """Format names as <ref> tags with natural-language joining."""
    segs = [f"<ref>{name}</ref>" for name in names if isinstance(name, str) and name.strip()]
    if not segs:
        return ""
    if len(segs) == 1:
        return segs[0]
    if len(segs) == 2:
        return f"{segs[0]} and {segs[1]}"
    return f"{', '.join(segs[:-1])} and {segs[-1]}"


def build_ref_answer(first_seg_name, class_names):
    """Mirror the answer templates from sec2_image_seg/process_muse_json.py."""
    primary = f"<ref>{first_seg_name}</ref>" if first_seg_name else ""
    class_part = join_ref_names(class_names)

    if primary and class_part:
        templates = [
            "It's {primary} (includes {class_part}).",
            "The target is {primary} (includes {class_part}).",
            "{primary} (includes {class_part}).",
        ]
        selector = (sum(ord(ch) for ch in primary) + len(class_part)) % len(templates)
        return templates[selector].format(primary=primary, class_part=class_part)
    if primary:
        return f"It's {primary}."
    if class_part:
        return f"It includes {class_part}."
    return ""


def build_ref_answer_from_segments(segments):
    """Build the training answer from JSONL segments."""
    labels = [
        str(segment.get('label', '')).strip()
        for segment in segments
        if isinstance(segment, dict) and str(segment.get('label', '')).strip()
    ]
    if not labels:
        return ""
    return build_ref_answer(labels[0], labels[1:])


def build_mask_data_from_segments(item):
    """Convert JSONL masks/segments into the internal ref-indexed mask dict."""
    all_masks = item['masks']
    mask_data = {}
    for idx, segment in enumerate(item['segments']):
        label = str(segment.get('label', '')).strip()
        mask_ids = segment.get('mask_ids', [])
        mask_data[str(idx)] = {
            'masks': [all_masks[mask_id] for mask_id in mask_ids],
            'desc': label,
        }
    return mask_data


def initialize_simple_data_list(dataset, data_root, ann_file, data_prefix=None, split='train'):
    """Initialize fields previously provided by mmengine BaseDataset."""
    dataset.data_root = data_root or ''
    dataset.ann_file = ann_file
    dataset.data_prefix = copy.copy(data_prefix or {})
    dataset.split = split

    if dataset.ann_file and not mmengine.is_abs(dataset.ann_file):
        dataset.ann_file = osp.join(dataset.data_root, dataset.ann_file)

    for key, prefix in dataset.data_prefix.items():
        if prefix and not mmengine.is_abs(prefix):
            dataset.data_prefix[key] = osp.join(dataset.data_root, prefix)

    dataset.data_list = dataset.load_data_list()
