import math
from typing import Any, Dict, List, Optional, Tuple
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn


class _BackboneProxy:
    """Proxy SAM3 backbone calls and cache the latest text prompt for the Qwen detector."""

    def __init__(self, adapter: "SetConQwenVideoDetectorAdapter", backbone: nn.Module):
        self._adapter = adapter
        self._backbone = backbone

    def forward_text(self, text_batch, device=None):
        self._adapter._last_text_batch = list(text_batch)
        return self._backbone.forward_text(text_batch, device=device)

    def __getattr__(self, name):
        return getattr(self._backbone, name)


class SetConQwenVideoDetectorAdapter(nn.Module):
    """
    Keep SAM3 detector interface unchanged, but swap detection outputs with SetCon-Qwen model outputs.

    The SAM3 feature detector is still used to produce tracker backbone features required by SAM3 tracker.
    """

    def __init__(
        self,
        feature_detector: nn.Module,
        setcon_model: nn.Module,
        processor: Any,
        tokenizer: Any,
        image_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        image_std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    ):
        super().__init__()
        self.feature_detector = feature_detector
        self.setcon_model = setcon_model
        self.processor = processor
        self.tokenizer = tokenizer
        self.image_mean = image_mean
        self.image_std = image_std
        self._last_text_batch: List[str] = []

        # Keep these attributes so existing SAM3 inference/compile logic can run unchanged.
        self.backbone = _BackboneProxy(self, self.feature_detector.backbone)
        self.transformer = self.feature_detector.transformer
        self.segmentation_head = self.feature_detector.segmentation_head
        self.rank = getattr(self.feature_detector, "rank", 0)
        self.world_size = getattr(self.feature_detector, "world_size", 1)

        # Cache Video feature
        self._cached_video_uid: Optional[int] = None
        self._cached_total_frames: int = -1
        self._cached_text_prompt: str = ""
        self._cached_sampled_indices: List[int] = []
        self._cached_hidden_states_per_segment: List[List[torch.Tensor]] = []

    def _tensor_to_pil(self, img_chw: torch.Tensor) -> Image.Image:
        img = img_chw.detach().float().cpu()
        if img.dim() != 3 or img.size(0) != 3:
            raise ValueError(f"Expected CHW image tensor with 3 channels, got shape={tuple(img.shape)}")
        mean = torch.tensor(self.image_mean, dtype=img.dtype).view(3, 1, 1)
        std = torch.tensor(self.image_std, dtype=img.dtype).view(3, 1, 1)
        img = img * std + mean
        img = (img.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
        img = img.permute(1, 2, 0).numpy()
        return Image.fromarray(img, mode="RGB")

    def _preprocessed_frame(self, img_chw: torch.Tensor) -> torch.Tensor:
        img = img_chw.detach().float()
        mean = torch.tensor(self.image_mean, dtype=img.dtype, device=img.device).view(3, 1, 1)
        std = torch.tensor(self.image_std, dtype=img.dtype, device=img.device).view(3, 1, 1)
        img = img * std + mean                              # denorm to [0,1]
        img = (img.clamp(0.0, 1.0) * 255.0).to(torch.uint8)  # uint8 truncation (== PIL)
        img = img.float() / 255.0                            # ToDtype(float32, scale)
        img = (img - mean) / std                             # Normalize(0.5, 0.5)
        return img

    def _normalize_xyxy(self, boxes: np.ndarray, w: int, h: int) -> np.ndarray:
        boxes = boxes.astype(np.float32).reshape(-1, 4)
        # If likely cxcywh, convert to xyxy.
        if np.any(boxes[:, 2] < boxes[:, 0]) or np.any(boxes[:, 3] < boxes[:, 1]):
            cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            x1 = cx - bw / 2.0
            y1 = cy - bh / 2.0
            x2 = cx + bw / 2.0
            y2 = cy + bh / 2.0
            boxes = np.stack([x1, y1, x2, y2], axis=-1)
        if np.max(np.abs(boxes)) > 1.5:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] / max(float(w), 1.0)
            boxes[:, [1, 3]] = boxes[:, [1, 3]] / max(float(h), 1.0)
        return np.clip(boxes, 0.0, 1.0)

    @staticmethod
    def _flatten_scores(scores: Any) -> np.ndarray:
        if scores is None:
            return np.zeros((0,), dtype=np.float32)
        if torch.is_tensor(scores):
            scores = scores.detach().float().cpu().numpy()
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        return scores

    @staticmethod
    def _to_numpy(x: Any) -> Optional[np.ndarray]:
        if x is None:
            return None
        if torch.is_tensor(x):
            return x.detach().float().cpu().numpy()
        return np.asarray(x)

    def _collect_setcon_dets(
        self,
        frame_tensor: torch.Tensor,
        frame_hw: Tuple[int, int],
        seg_hidden_states: Optional[List[torch.Tensor]],
        text_prompt: str,
        target_h: int,
        target_w: int,
        device: torch.device,
    ):
        if not text_prompt or text_prompt in {"<text placeholder>", "visual"}:
            return (
                torch.zeros((0, 4), device=device),
                torch.zeros((0, target_h, target_w), device=device),
                torch.zeros((0,), device=device),
            )

        if seg_hidden_states is None or len(seg_hidden_states) == 0:
            return (
                torch.zeros((0, 4), device=device),
                torch.zeros((0, target_h, target_w), device=device),
                torch.zeros((0,), device=device),
            )

        frame_h, frame_w = frame_hw
        pred_list = self.setcon_model.decode_hidden_states(
            preprocessed_image=frame_tensor,
            original_size=(frame_h, frame_w),
            seg_hidden_states=seg_hidden_states,
        )


        all_boxes, all_masks, all_scores = [], [], []
        img_w, img_h = frame_w, frame_h
        for pred in pred_list:
            score_np = self._flatten_scores(pred.get("scores"))
            box_np = self._to_numpy(pred.get("boxes"))
            mask_np = self._to_numpy(pred.get("masks"))
            if score_np.size == 0 or box_np is None or mask_np is None:
                continue
            box_np = self._normalize_xyxy(box_np, img_w, img_h)
            box_np = box_np[0:1]
            score_np = score_np[0:1]
            mask_np = np.asarray(mask_np)
            if mask_np.ndim == 4:
                mask_np = mask_np[0, 0]
            elif mask_np.ndim == 3:
                mask_np = mask_np[0]
            elif mask_np.ndim != 2:
                continue
            mask_t = torch.as_tensor(mask_np, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            mask_t = F.interpolate(mask_t, size=(target_h, target_w), mode="bilinear", align_corners=False)
            all_masks.append(mask_t.squeeze(0).squeeze(0))
            all_boxes.append(torch.as_tensor(box_np[0], device=device, dtype=torch.float32))
            all_scores.append(torch.as_tensor(score_np[0], device=device, dtype=torch.float32))

        if len(all_scores) == 0:
            return (
                torch.zeros((0, 4), device=device),
                torch.zeros((0, target_h, target_w), device=device),
                torch.zeros((0,), device=device),
            )
        return torch.stack(all_boxes, dim=0), torch.stack(all_masks, dim=0), torch.stack(all_scores, dim=0)

    @staticmethod
    def _sample_key_indices(total_frames: int, num_segments: int = 8) -> List[int]:
        if total_frames <= 0:
            return [0]
        if total_frames < num_segments:
            return list(range(total_frames))
        sampled = np.linspace(0, total_frames - 1, num=num_segments, dtype=np.float32)
        sampled = np.rint(sampled).astype(np.int32).tolist()
        sampled[0] = 0
        sampled[-1] = total_frames - 1
        return sampled

    @staticmethod
    def _frame_to_segment(frame_idx: int, sampled_indices: List[int]) -> int:
        if len(sampled_indices) <= 1:
            return 0
        if sampled_indices[0] == sampled_indices[-1]:
            return 0
        midpoints = [int((sampled_indices[i] + sampled_indices[i + 1]) // 2) for i in range(len(sampled_indices) - 1)]
        segment_idx = int(np.searchsorted(np.asarray(midpoints, dtype=np.int32), frame_idx, side="right"))
        return max(0, min(len(sampled_indices) - 1, segment_idx))

    def _ensure_video_cache(self, img_batch: torch.Tensor, text_prompt: str):
        total_frames = int(img_batch.shape[0])
        video_uid = int(img_batch.data_ptr()) if torch.is_tensor(img_batch) else id(img_batch)
        need_refresh = (
            self._cached_video_uid != video_uid
            or self._cached_total_frames != total_frames
            or self._cached_text_prompt != text_prompt
            or len(self._cached_hidden_states_per_segment) != len(self._cached_sampled_indices)
        )
        if not need_refresh:
            return

        sampled_indices = self._sample_key_indices(total_frames, num_segments=8)
        num_segments = len(sampled_indices)

        sampled_frames = [self._tensor_to_pil(img_batch[idx]) for idx in sampled_indices]
        prompt = "Given the video context frames and the description \"{}\", please segment all the objects matching the description.".format(text_prompt)
        hs_result = self.setcon_model.infer_hidden_states(
            video=sampled_frames,
            text=prompt,
            tokenizer=self.tokenizer,
            processor=self.processor,
        )
        print("[INFO] Text Prompt: ", text_prompt)
        print("[INFO] Pred Text: ", hs_result.get("prediction", ""))
    
        shared = hs_result.get("seg_hidden_states", []) or []
        per_frame_hidden_states = [shared for _ in range(num_segments)]

        self._cached_video_uid = video_uid
        self._cached_total_frames = total_frames
        self._cached_text_prompt = text_prompt
        self._cached_sampled_indices = sampled_indices
        self._cached_hidden_states_per_segment = per_frame_hidden_states

    def forward_video_grounding_multigpu(self, **kwargs):
        # SetCon overwrites SAM3's own detection outputs (pred_logits/pred_boxes_xyxy/pred_masks) below,
        # only its vision backbone features are needed for the tracker.
        kwargs.setdefault("backbone_only", True)
        sam3_out, aux = self.feature_detector.forward_video_grounding_multigpu(**kwargs)
        # 2) Replace detection outputs with SetCon-Qwen outputs.
        text_prompt = self._last_text_batch[0] if len(self._last_text_batch) > 0 else ""

        img_batch = kwargs["backbone_out"]["img_batch_all_stages"]
        total_frames = int(img_batch.shape[0])
        
        frame_idx = int(kwargs["frame_idx"])
        frame_idx = max(0, min(total_frames - 1, frame_idx))
        current_frame = self._preprocessed_frame(img_batch[frame_idx])
        frame_hw = (int(img_batch.shape[-2]), int(img_batch.shape[-1]))

        self._ensure_video_cache(img_batch=img_batch, text_prompt=text_prompt)
        segment_idx = self._frame_to_segment(frame_idx, self._cached_sampled_indices)
        seg_hidden_states = self._cached_hidden_states_per_segment[segment_idx]

        pred_masks_ref = sam3_out["pred_masks"]  # [B, Q, H, W]
        bsz, num_queries, mask_h, mask_w = pred_masks_ref.shape
        device = pred_masks_ref.device
        boxes, masks, scores = self._collect_setcon_dets(
            frame_tensor=current_frame,
            frame_hw=frame_hw,
            seg_hidden_states=seg_hidden_states,
            text_prompt=text_prompt,
            target_h=mask_h,
            target_w=mask_w,
            device=device,
        )
        det_n = min(num_queries, int(scores.numel()))

        pred_logits = torch.full((bsz, num_queries, 1), -20.0, device=device, dtype=pred_masks_ref.dtype)
        pred_boxes_xyxy = torch.zeros((bsz, num_queries, 4), device=device, dtype=pred_masks_ref.dtype)
        pred_masks = torch.full((bsz, num_queries, mask_h, mask_w), -10.0, device=device, dtype=pred_masks_ref.dtype)
        if det_n > 0:
            probs = scores[:det_n].clamp(1e-4, 1 - 1e-4)
            logits = torch.log(probs / (1.0 - probs))
            pred_logits[0, :det_n, 0] = logits.to(dtype=pred_logits.dtype)
            pred_boxes_xyxy[0, :det_n] = boxes[:det_n].to(dtype=pred_boxes_xyxy.dtype)
            pred_masks[0, :det_n] = masks[:det_n].to(dtype=pred_masks.dtype)

        sam3_out["pred_logits"] = pred_logits
        sam3_out["pred_boxes_xyxy"] = pred_boxes_xyxy
        sam3_out["pred_masks"] = pred_masks
        return sam3_out, aux

