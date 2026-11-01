from typing import Dict

import numpy as np
import torch
from PIL import Image
from mmengine.model import BaseModule
from torchvision.transforms import v2

from transformers import GroundingDinoForObjectDetection

from third_parts.sam3.model import box_ops

from projects.setcon.models.groundingdino_train import (
    GD_DEFAULT,
    build_text_tensors,
    encode_vision,
    resize_and_pad_batch,
    run_grounding,
)


class GroundingDinoRunner(BaseModule):
    """Inference-time grounding encoder: Grounding DINO produces language-
    conditioned boxes, then SAM3's interactive predictor (`predict_inst`) turns
    each box into a mask (box-prompt segmentation). Mirrors SAM3Runner's
    interface (set_image / set_text_prompt / set_confidence_threshold) and uses
    the same GD submodule name (`gd`) as GroundingDinoTrainRunner so a trained
    checkpoint loads directly. The SAM3 model here is frozen and only used for
    box->mask; it is not part of the trained checkpoint.
    """

    def __init__(
        self,
        pretrained_path: str = GD_DEFAULT,
        image_mean=(0.485, 0.456, 0.406),
        image_std=(0.229, 0.224, 0.225),
        shortest_edge: int = 800,
        longest_edge: int = 1333,
        size_divisor: int = 32,
        confidence_threshold: float = 0.5,
        # SAM3 (for box->mask)
        sam3_bpe_path: str = "third_parts/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        sam3_checkpoint_path: str = "pretrained/sam3/sam3.pt",
        sam3_load_from_HF: bool = False,
        sam3_resolution: int = 1008,
    ):
        super().__init__(init_cfg=None)

        self.gd = GroundingDinoForObjectDetection.from_pretrained(pretrained_path)
        cfg = self.gd.config
        self.hidden_dim = cfg.d_model
        self.confidence_threshold = confidence_threshold

        # GD-native preprocessing (aspect-ratio preserving + pad); resize is
        # per-image so the transform here is normalize-only.
        self.shortest_edge = shortest_edge
        self.longest_edge = longest_edge
        self.size_divisor = size_divisor
        self.normalize = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=list(image_mean), std=list(image_std)),
        ])

        # Frozen SAM3, only for box-prompt -> mask.
        import third_parts.sam3  # noqa: F401
        from third_parts.sam3.model_builder import build_sam3_image_model
        self.sam3_model = build_sam3_image_model(
            bpe_path=sam3_bpe_path, checkpoint_path=sam3_checkpoint_path,
            load_from_HF=sam3_load_from_HF, device='cpu', eval_mode=True,
            enable_inst_interactivity=True,
        )
        self.sam3_model.requires_grad_(False)
        self.sam3_resolution = sam3_resolution
        self.sam3_transform = v2.Compose([
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(size=(sam3_resolution, sam3_resolution)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    @property
    def device(self):
        return next(self.gd.parameters()).device

    @torch.inference_mode()
    def set_confidence_threshold(self, threshold: float, state=None):
        self.confidence_threshold = threshold
        if state is not None and "gd_state" in state:
            return self._forward_grounding(state)
        return state

    @torch.inference_mode()
    def set_image(self, image, state=None):
        if state is None:
            state = {}
        if isinstance(image, Image.Image):
            width, height = image.size
        elif isinstance(image, torch.Tensor):
            height, width = image.shape[-2:]
        elif isinstance(image, np.ndarray):
            height, width = image.shape[:2]
        else:
            raise ValueError("Image must be a PIL image or a tensor")

        state["original_height"] = height
        state["original_width"] = width

        img = v2.functional.to_image(image)

        # Grounding DINO vision encode (for boxes): GD-native aspect-ratio
        # resize + pad, with a real pixel_mask marking the padded region.
        gd_pixels, gd_mask = resize_and_pad_batch(
            [img], self.normalize, self.device,
            self.shortest_edge, self.longest_edge, self.size_divisor,
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            state["gd_state"] = encode_vision(self.gd, gd_pixels, gd_mask)

        # SAM3 backbone encode (for box->mask via predict_inst).
        sam_pixels = self.sam3_transform(img).unsqueeze(0).to(self.device)
        backbone_out = self.sam3_model.backbone.forward_image(sam_pixels)
        if "sam2_backbone_out" in backbone_out:
            sam2_bb = backbone_out["sam2_backbone_out"]
            dec = self.sam3_model.inst_interactive_predictor.model.sam_mask_decoder
            sam2_bb["backbone_fpn"][0] = dec.conv_s0(sam2_bb["backbone_fpn"][0])
            sam2_bb["backbone_fpn"][1] = dec.conv_s1(sam2_bb["backbone_fpn"][1])
        state["backbone_out"] = backbone_out
        return state

    @torch.inference_mode()
    def _forward_grounding(self, state: Dict):
        text = state["text"]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred_logits, pred_boxes = run_grounding(self.gd, state["gd_state"], text)

        probs = pred_logits.float().sigmoid()[0, :, 0]  # (Q,), single prompt, bs==1
        boxes_cxcywh = pred_boxes.float()[0]             # (Q, 4) normalized

        keep = probs > self.confidence_threshold
        probs = probs[keep]
        boxes_cxcywh = boxes_cxcywh[keep]

        img_h, img_w = state["original_height"], state["original_width"]
        if probs.shape[0] == 0:
            state["masks"] = np.zeros((0, img_h, img_w), dtype=bool)
            state["boxes"] = np.zeros((0, 4), dtype=np.float32)
            state["scores"] = np.zeros((0,), dtype=np.float32)
            return state

        scale = torch.tensor([img_w, img_h, img_w, img_h], device=self.device, dtype=torch.float32)
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes_cxcywh) * scale[None, :]  # pixel xyxy

        order = torch.argsort(probs, descending=True)
        probs = probs[order]
        boxes_xyxy = boxes_xyxy[order]

        # Box-prompt each detection through SAM3 to obtain a mask.
        masks, _, _ = self.sam3_model.predict_inst(
            state,
            box=boxes_xyxy.cpu().numpy(),
            multimask_output=False,
        )
        masks = np.asarray(masks)
        if masks.ndim == 4:      # (N, 1, H, W)
            masks = masks[:, 0]
        masks_bin = masks > 0.0  # predict_inst returns thresholded masks by default

        state["scores"] = probs.cpu().numpy()
        state["boxes"] = boxes_xyxy.cpu().numpy()
        state["masks"] = masks_bin.astype(bool)
        return state

    @torch.inference_mode()
    def set_text_prompt(self, language_embd, state: Dict):
        if not isinstance(language_embd, list):
            raise ValueError("language_embd must be a list")
        if "gd_state" not in state:
            raise ValueError("You must call set_image before set_text_prompt")

        device = state["gd_state"]["source_flatten"].device
        dtype = state["gd_state"]["source_flatten"].dtype
        state["text"] = build_text_tensors(language_embd, device, dtype, self.hidden_dim)
        return self._forward_grounding(state)

    def forward(self, batch):
        raise NotImplementedError
