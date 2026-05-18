"""
SetCon video segmentation dataset for setcon_training_datasets/video JSONL.

Expected JSONL fields:
  video_id: directory key under frame_root
  frames: frame filenames relative to frame_root/video_id
  text: referring description
  masks: object tracks, each as a list of per-frame masks
  segments: list of {"label": str, "mask_ids": [int, ...]}

Each segment becomes one <ref> span in the generated answer. During sampling,
selected frame masks are decoded for the grounding branch.
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
    load_jsonl,
)
from projects.setcon.datasets.base import SetConBaseDataset


# Question templates for video referring segmentation
VIDEO_SEG_QUESTIONS = [
    "Given the video context frames and the description \"{class_name}\", please segment all the objects matching the description.",
    "Based on the preceding video frames and the description \"{class_name}\", segment the referred objects.",
    "These are frames from a video. According to the description \"{class_name}\", segment the matching objects.",
    "Given the video frames, find and segment the objects described as \"{class_name}\".",
    "Using the video context and the description \"{class_name}\", please segment the corresponding objects.",
]


class SetConVideoSegDataset(SetConBaseDataset):
    """
    Video segmentation dataset backed by SetCon JSONL annotations.

    The loader resolves frame paths with frame_root/video_id/frame, converts
    segments into ref-indexed mask groups, and generates <ref> answers with the
    same template used by the image pipeline.
    """

    def __init__(self,
                 json_file,
                 prompt_template=None,
                 tokenizer=None,
                 max_length=2048,
                 special_tokens=None,
                 arch_type: Literal['intern_vl', 'qwen'] = 'intern_vl',
                 preprocessor=None,
                 extra_image_processor=None,
                 num_sample_frames: int = 8,
                 frame_root=None,
                 use_separate_grounding_sampling: bool = True,
                 repeats: float = 1.0,
                 name: str = 'VideoSegDataset',
                 **kwargs):

        # Shared tokenizer and image-processing setup.
        super().__init__(
            tokenizer=tokenizer,
            prompt_template=prompt_template,
            max_length=max_length,
            special_tokens=special_tokens,
            arch_type=arch_type,
            preprocessor=preprocessor,
            extra_image_processor=extra_image_processor,
            repeats=repeats,
            name=name,
            **kwargs,
        )

        # Video dataset state.
        self.num_sample_frames = num_sample_frames
        self.frame_root = frame_root
        self.use_separate_grounding_sampling = use_separate_grounding_sampling
        self.begin_str = '<image>\n'

        # Load JSONL metadata.
        assert json_file is not None and tokenizer is not None
        self.data_list = self._build_data_list(load_jsonl(json_file))

    def _join_frame_path(self, video_id: str, frame_path: str) -> str:
        if osp.isabs(frame_path):
            return frame_path
        if self.frame_root:
            return osp.join(self.frame_root, video_id, frame_path)
        return osp.join(video_id, frame_path)

    def _build_data_list(self, annotations) -> List[dict]:
        """Build data list from setcon_training_datasets/video JSONL annotations."""
        data_list = []
        for item in annotations:
            video_id = item.get('video_id', '')
            segments = item['segments']
            data_info = {
                'image': [self._join_frame_path(video_id, frame) for frame in item['frames']],
                'mask': build_mask_data_from_segments(item),
                'text': item['text'],
                'answer': build_ref_answer_from_segments(segments),
                'video_id': video_id,
            }
            data_list.append(data_info)

        if len(data_list) == 0:
            raise ValueError('No samples found in the JSONL file.')

        return data_list

    def real_len(self):
        return len(self.data_list)

    @property
    def modality_length(self):
        return [self._get_modality_length_default(10000) for _ in range(len(self))]

    def _sample_frames(self, num_total_frames: int):
        if num_total_frames <= 0:
            raise ValueError('Video has no frames.')

        if num_total_frames >= self.num_sample_frames:
            sampled_indices = random.sample(range(num_total_frames), self.num_sample_frames)
        else:
            sampled_indices = list(range(num_total_frames))

        return sorted(sampled_indices)

    def _decode_selected_frame_masks(
        self,
        mask_data: dict,
        selected_frame_indices: List[int],
        height: int,
        width: int,
    ):
        """
        Decode masks for all selected frames.

        Args:
            mask_data: Internal ref-indexed mask dict built from segments.
            selected_frame_indices: Indices of selected frames in temporal order
            height: Image height
            width: Image width

        Returns:
            List of mask tensors (one per token group), each tensor shape (F_sel, N_obj, H, W)
        """
        masks_out = []
        if not isinstance(mask_data, dict) or len(mask_data) == 0:
            masks_out.append(torch.zeros((len(selected_frame_indices), 0, height, width), dtype=torch.uint8))
            return masks_out

        token_keys = sorted(mask_data.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))
        num_sel = len(selected_frame_indices)

        for key in token_keys:
            group = mask_data.get(key, {})
            obj_masks_list = group.get("masks", [])  # List[List[rle]] - [num_objects][num_frames]
            num_obj = len(obj_masks_list)
            if num_obj == 0:
                masks_out.append(torch.zeros((num_sel, 0, height, width), dtype=torch.uint8))
                continue

            # Build [num_obj, num_sel, H, W], then permute -> [num_sel, num_obj, H, W]
            obj_frame_tensors = []
            for obj_frame_masks in obj_masks_list:
                per_obj_frames = []
                for frame_idx in selected_frame_indices:
                    binary_mask = np.zeros((height, width), dtype=np.uint8)
                    if frame_idx < len(obj_frame_masks):
                        rle = obj_frame_masks[frame_idx]
                        if rle is not None:
                            if isinstance(rle, dict):
                                if isinstance(rle['counts'], str):
                                    rle['counts'] = rle['counts'].encode()
                                binary_mask = mask_utils.decode(rle).astype(np.uint8)
                            elif isinstance(rle, list):
                                for seg in rle:
                                    rles = mask_utils.frPyObjects([seg], height, width)
                                    m = mask_utils.decode(rles).astype(np.uint8)
                                    binary_mask += m.squeeze()
                    per_obj_frames.append(torch.from_numpy((binary_mask > 0).astype(np.uint8)))
                obj_frame_tensors.append(torch.stack(per_obj_frames, dim=0))  # [num_sel, H, W]

            masks_tensor = torch.stack(obj_frame_tensors, dim=0).permute(1, 0, 2, 3).contiguous()  # [num_sel, num_obj, H, W]
            masks_out.append(masks_tensor)
        return masks_out

    def prepare_data(self, index):
        """Prepare data for a given index."""
        index = index % self.real_len()
        ann_info = copy.deepcopy(self.data_list[index])

        all_frame_paths = ann_info['image']
        mask_data = ann_info['mask']
        text = ann_info['text']
        answer = ann_info.get('answer', '')

        num_total_frames = len(all_frame_paths)

        # Sample frame indices for the LLM and grounding branches.
        sampled_indices = self._sample_frames(num_total_frames)
        if self.use_separate_grounding_sampling:
            separate_grounding_indices = self._sample_frames(num_total_frames)
            total_n = len(sampled_indices)
            n_from_sampled = total_n // 2
            n_from_separate = total_n - n_from_sampled
            grounding_sampled_indices = (
                random.sample(sampled_indices, min(n_from_sampled, len(sampled_indices)))
                + random.sample(
                    separate_grounding_indices,
                    min(n_from_separate, len(separate_grounding_indices)),
                )
            )
            random.shuffle(grounding_sampled_indices)
        else:
            grounding_sampled_indices = sampled_indices

        # Load selected frame images for the LLM branch.
        images = []
        for fidx in sampled_indices:
            img = self._read_image(all_frame_paths[fidx])
            if img is None:
                return None
            images.append(img)

        # Load selected frame images for the grounding branch.
        grounding_images = []
        for fidx in grounding_sampled_indices:
            img = self._read_image(all_frame_paths[fidx])
            if img is None:
                return None
            grounding_images.append(img)

        # Frames in a video share one size; use the last grounding frame.
        target_image = grounding_images[-1]
        width, height = target_image.size

        # Decode masks for the same frames used by g_pixel_values.
        masks = self._decode_selected_frame_masks(mask_data, grounding_sampled_indices, height, width)
        has_positive_mask = any(mask.numel() > 0 and mask.sum().item() > 0 for mask in masks)
        if not has_positive_mask and random.random() < 0.5:
            answer = '<ref>no target</ref>'

        phrase = text.strip()
        if phrase and phrase[-1] == '.':
            phrase = phrase[:-1]

        question = random.choice(VIDEO_SEG_QUESTIONS).format(class_name=phrase)
        conversation = [
            {'from': 'human', 'value': self.begin_str + question},
            {'from': 'gpt', 'value': answer},
        ]

        out_data_dict = {}
        out_data_dict['masks'] = masks

        try:
            image_data = self._process_multiple_images(images)
            grounding_data = self._process_multiple_images(grounding_images)
            out_data_dict.update(image_data)
            if 'g_pixel_values' in grounding_data:
                out_data_dict['g_pixel_values'] = grounding_data['g_pixel_values']

            num_frames = len(images)
            image_token_str = self._create_token_string(
                image_data['num_image_tokens'], num_frames
            )

            conversations = self._process_conversations_for_encoding(
                conversation, image_token_str, is_video=True
            )

            if self.arch_type == 'qwen' and 'num_frame_tokens' in image_data and num_frames > 1:
                conversations = self._expand_video_tokens(
                    conversations,
                    image_data['num_frame_tokens'],
                    image_data['num_image_tokens'],
                )

            token_dict = self.get_inputid_labels(conversations)
            out_data_dict.update(token_dict)

        except Exception as e:
            print(f'Error processing video frames: {e}', flush=True)
            return None

        out_data_dict['type'] = 'video'
        return out_data_dict

