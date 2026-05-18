"""
SetCon image segmentation dataset for setcon_training_datasets/image JSONL.

Expected JSONL fields:
  image: path relative to image_root, or an absolute path
  text: referring question or description
  masks: list of RLE or polygon masks
  segments: list of {"label": str, "mask_ids": [int, ...]}

Each segment becomes one <ref> span in the generated answer. The referenced
masks are decoded as separate objects and are not merged.
"""
import copy
import os.path as osp
import random
from typing import Literal, List

import torch
import numpy as np
from pycocotools import mask as mask_utils

from .common import (
    build_mask_data_from_segments,
    build_ref_answer_from_segments,
    initialize_simple_data_list,
    load_jsonl,
)
from projects.setcon.datasets.base import SetConBaseDataset

SEG_QUESTIONS_MULTI = [
    "Given the question or description \"{class_name}\", Can you segment the object(s) it refers to in this image?",
    "Given the question or description \"{class_name}\", Please segment the object(s) it refers to in this image.",
    "For the question or description \"{class_name}\", Segment the object(s) it refers to in this image.",
    "Analyze the question or description \"{class_name}\", and segment the object(s) it refers to in this image.",
    "In the question or description \"{class_name}\", segment the object(s) it refers to in this image.",
]

class SetConMultiSegDataset(SetConBaseDataset):
    """
    Image segmentation dataset backed by SetCon JSONL annotations.

    The loader normalizes each row into the internal structure consumed by
    _parse_annotations: absolute image path, ref-indexed mask groups, the
    original text prompt, and a generated <ref> answer.
    """

    def __init__(self,
                 data_root,
                 ann_file=None,
                 special_tokens=None,
                 prompt_template=None,
                 extra_image_processor=None,
                 image_root=None,
                 tokenizer=None,
                 max_length=2048,
                 single_image_mode=False,
                 arch_type: Literal['intern_vl', 'qwen'] = 'intern_vl',
                 preprocessor=None,
                 repeats: int = 1,
                 name: str = 'MultiSegDataset',
                 **kwargs):
        self.image_root = image_root

        initialize_simple_data_list(
            self,
            data_root=data_root,
            ann_file=ann_file,
            split=kwargs.pop('split', 'train'),
        )

        # Shared tokenizer and image-processing setup.
        SetConBaseDataset.__init__(self,
            tokenizer=tokenizer,
            prompt_template=prompt_template,
            max_length=max_length,
            special_tokens=special_tokens,
            arch_type=arch_type,
            preprocessor=preprocessor,
            extra_image_processor=extra_image_processor,
            repeats=repeats,
            name=name
        )
        
        # Image dataset state.
        self.begin_str = f'<image>\n'
        self.single_image_mode = single_image_mode

    def _join_image_path(self, image_path: str) -> str:
        if osp.isabs(image_path):
            return image_path
        if self.image_root:
            return osp.join(self.image_root, image_path)
        return image_path

    def load_data_list(self) -> List[dict]:
        """Load JSONL annotations in setcon_training_datasets/image format."""
        data_list = []

        for item in load_jsonl(self.ann_file):
            segments = item['segments']
            data_info = {
                'img_path': self._join_image_path(item['image']),
                'mask': build_mask_data_from_segments(item),
                'text': item['text'],
                'answer': build_ref_answer_from_segments(segments),
            }
            data_list.append(data_info)

        if len(data_list) == 0:
            raise ValueError(f'No sample in split.')

        return data_list

    @property
    def modality_length(self):
        return [self._get_modality_length_default(100) for _ in range(len(self))]

    def _decode_mask_group(self, mask_group, height, width):
        """Decode one segment into a tensor shaped (1, num_objects, H, W)."""
        masks = []
        for obj_mask in mask_group:
            if isinstance(obj_mask, dict):  # RLE
                if isinstance(obj_mask['counts'], str):
                    obj_mask['counts'] = obj_mask['counts'].encode()
                binary_mask = mask_utils.decode(obj_mask).astype(np.uint8)
            elif isinstance(obj_mask, list):  # polygon
                binary_mask = np.zeros((height, width), dtype=np.uint8)
                for seg in obj_mask:
                    rles = mask_utils.frPyObjects([seg], height, width)
                    m = mask_utils.decode(rles).astype(np.uint8)
                    binary_mask += m.squeeze()
            else:
                binary_mask = np.zeros((height, width), dtype=np.uint8)
            if binary_mask.sum() == 0:  # skip empty masks
                continue
            masks.append(binary_mask)
        
        if len(masks) == 0:
            masks_tensor = torch.zeros((0, height, width), dtype=torch.uint8)
        else:
            masks_tensor = torch.stack([torch.from_numpy(m) for m in masks], dim=0)
        return masks_tensor.unsqueeze(0)

    def _parse_annotations(self, ann_info):
        """Decode normalized JSONL metadata and build one training example."""
        image_path = ann_info['img_path']
        image = self._read_image(image_path)
        if image is None:
            return None
        width, height = image.size

        mask_data = ann_info['mask']
        phrase = ann_info['text']
        answer = ann_info.get('answer', '')
        
        phrase = phrase.strip()
        masks = []
        
        if not isinstance(mask_data, dict):
            raise ValueError(f"Invalid mask data format: {type(mask_data)}")

        for idx in range(len(mask_data)):
            key = str(idx)
            if key not in mask_data:
                continue
            decoded_masks = self._decode_mask_group(mask_data[key]["masks"], height, width)
            masks.append(decoded_masks)

        has_positive_mask = any(mask.numel() > 0 and mask.sum().item() > 0 for mask in masks)
        
        if not has_positive_mask and random.random() < 0.5:
            print(f"[INFO] No positive mask found for {image_path}, original answer: {answer}")
            answer = "It's <ref>no target</ref> (includes <ref>none</ref>)."
        conversation = []
        question = random.choice(SEG_QUESTIONS_MULTI).format(class_name=phrase)
        conversation.append({'from': 'human', 'value': self.begin_str + question})
        
        if answer:
            conversation.append({'from': 'gpt', 'value': answer})
        else:
            raise ValueError(f"No answer found for {image_path}")

        ann_info.update({
            'masks': masks,
            'conversations': conversation,
            'image': image_path,
            'pil_image': image
        })
        return ann_info

    def prepare_data(self, index):
        data_dict = copy.deepcopy(self.data_list[index])
        data_dict = self._parse_annotations(data_dict)
        if data_dict is None:
            return None

        out_data_dict = {'masks': data_dict['masks']}

        image_data = self._process_single_image(data_dict['pil_image'], self.single_image_mode)
        out_data_dict.update(image_data)

        image_token_str = self._create_image_token_string(image_data['num_image_tokens'])
        conversation = self._process_conversations_for_encoding(data_dict['conversations'], image_token_str)
        token_dict = self.get_inputid_labels(conversation)
        out_data_dict.update(token_dict)

        return out_data_dict

    def real_len(self):
        return len(self.data_list)
