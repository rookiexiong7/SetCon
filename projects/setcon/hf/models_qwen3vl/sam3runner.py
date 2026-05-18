import copy
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from mmengine.model import BaseModule
from torchvision.transforms import v2

from third_parts.sam3.model import box_ops
from third_parts.sam3.model.data_misc import FindStage, interpolate

class SAM3Runner(BaseModule):

    def __init__(
        self,
        # Load SAM3 model
        bpe_path: str = "third_parts/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        checkpoint_path: str = None,
        load_from_HF: bool = False,
        eval_mode: bool = True,
        # Detection params
        resolution: int = 1008,
        confidence_threshold: float = 0.5,
    ):
        super().__init__(init_cfg=None)

        import third_parts.sam3     # noqa: F401
        from third_parts.sam3.model_builder import build_sam3_image_model

        # Build base model
        self.sam3_model = build_sam3_image_model(
            bpe_path=bpe_path, checkpoint_path=checkpoint_path, 
            load_from_HF=load_from_HF, device='cpu', eval_mode=eval_mode,
        )
        
        self.device = self.sam3_model.device
        self.hidden_dim = self.sam3_model.backbone.language_backbone.d_model

        self.resolution = resolution
        self.confidence_threshold = confidence_threshold
        
        self.transform = v2.Compose([
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(size=(self.resolution, self.resolution)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    @torch.inference_mode()
    def set_confidence_threshold(self, threshold: float, state=None):
        """Sets the confidence threshold for the masks"""
        self.confidence_threshold = threshold
        if state is not None and "boxes" in state:
            # we need to filter the boxes again
            # In principle we could do this more efficiently since we would only need
            # to rerun the heads. But this is simpler and not too inefficient
            return self._forward_grounding(state)
        return state

    @torch.inference_mode()
    def set_image(self, image, state=None):
        """Sets the image on which we want to do predictions."""
        if state is None:
            state = {}

        if isinstance(image, Image.Image):
            width, height = image.size
        elif isinstance(image, (torch.Tensor, np.ndarray)):
            height, width = image.shape[-2:]
        else:
            raise ValueError("Image must be a PIL image or a tensor")

        image = v2.functional.to_image(image).to(self.device)
        image = self.transform(image).unsqueeze(0)

        state["original_height"] = height
        state["original_width"] = width
        state["backbone_out"] = self.sam3_model.backbone.forward_image(image)
        inst_interactivity_en = self.sam3_model.inst_interactive_predictor is not None
        if inst_interactivity_en and "sam2_backbone_out" in state["backbone_out"]:
            sam2_backbone_out = state["backbone_out"]["sam2_backbone_out"]
            sam2_backbone_out["backbone_fpn"][0] = (
                self.sam3_model.inst_interactive_predictor.model.sam_mask_decoder.conv_s0(
                    sam2_backbone_out["backbone_fpn"][0]
                )
            )
            sam2_backbone_out["backbone_fpn"][1] = (
                self.sam3_model.inst_interactive_predictor.model.sam_mask_decoder.conv_s1(
                    sam2_backbone_out["backbone_fpn"][1]
                )
            )
        return state
    
    @torch.inference_mode()
    def _forward_grounding(self, state: Dict, prompt_mode: str = "text_only"):
        find_stage = FindStage(
            img_ids=torch.tensor([0], device=self.device, dtype=torch.long),
            text_ids=torch.tensor([0], device=self.device, dtype=torch.long),
            input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
            input_points=None, input_points_mask=None,
        )
        
        outputs = self.sam3_model.forward_grounding(
            backbone_out=state["backbone_out"], find_input=find_stage,
            geometric_prompt=state["geometric_prompt"], find_target=None,
        )

        out_bbox = outputs["pred_boxes"]
        out_logits = outputs["pred_logits"]
        out_masks = outputs["pred_masks"]
        out_probs = (out_logits.sigmoid() * outputs["presence_logit_dec"].sigmoid().unsqueeze(1)).squeeze(-1)

        keep = out_probs > self.confidence_threshold
        out_probs, out_masks, out_bbox = out_probs[keep], out_masks[keep], out_bbox[keep]

        img_h, img_w = state["original_height"], state["original_width"]
        scale_fct = torch.tensor([img_w, img_h, img_w, img_h]).to(self.device)
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox) * scale_fct[None, :]
        out_masks = interpolate(out_masks.unsqueeze(1), (img_h, img_w), mode="bilinear", align_corners=False).sigmoid()

        state["masks_logits"] = out_masks
        state["masks"] = out_masks > 0.5
        state["boxes"] = boxes
        state["scores"] = out_probs

        if len(state["scores"]) > 0:
            score_indices = sorted(range(len(state["scores"])), key=lambda k: state["scores"][k], reverse=True)
            state["scores"] = np.array([state["scores"][i].cpu().item() for i in score_indices])
            state["boxes"] = np.stack([state["boxes"][i].cpu().numpy() for i in score_indices], axis=0)
            state["masks"] = np.stack([state["masks"][i].squeeze().cpu().numpy() for i in score_indices], axis=0)
        
        return state
    
    @torch.inference_mode()
    def set_text_prompt(self, language_embd, state: Dict):
        """
        Set text prompt for grounding.
        
        Args:
            language_embd: List[Tensor[label_len, D]]: new format (multi-token per segment)
            state: SAM3 state dict
        """
        if isinstance(language_embd, list):
            # New format: List[Tensor[label_len, hidden_dim]]
            # Pad to [max_label_len, B, hidden_dim]
            B = len(language_embd)
            if B == 0:
                device = state["backbone_out"]["vision_features"].device
                dtype = state["backbone_out"]["vision_features"].dtype
                hidden_dim = self.hidden_dim
                language_features = torch.zeros(1, 0, hidden_dim, device=device, dtype=dtype)
                language_mask = torch.zeros(0, 1, device=device, dtype=torch.bool)
            else:
                label_lengths = [tokens.shape[0] for tokens in language_embd]
                max_label_len = max(label_lengths)
                hidden_dim = language_embd[0].shape[-1]
                device = language_embd[0].device
                dtype = language_embd[0].dtype
                
                language_features = torch.zeros(max_label_len, B, hidden_dim, device=device, dtype=dtype)
                language_mask = torch.ones(B, max_label_len, device=device, dtype=torch.bool)
                
                for seg_idx, (tokens, length) in enumerate(zip(language_embd, label_lengths)):
                    language_features[:length, seg_idx, :] = tokens
                    language_mask[seg_idx, :length] = False  # False = valid
            
            text_output = {
                "language_features": language_features,
                "language_mask": language_mask
            }
        else:
            raise ValueError("language_embd must be a list")
        
        state["backbone_out"].update(text_output)
        
        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.sam3_model._get_dummy_prompt()
        return self._forward_grounding(state)

    @torch.inference_mode()
    def set_text_prompt_sam3(self, prompt: str, state: Dict):
        if "backbone_out" not in state:
            raise ValueError("You must call set_image before set_text_prompt")

        text_outputs = self.sam3_model.backbone.forward_text([prompt], device=self.device)
        # will erase the previous text prompt if any
        state["backbone_out"].update(text_outputs)
        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.sam3_model._get_dummy_prompt()

        return self._forward_grounding(state)

    def forward(self, batch):
        raise NotImplementedError   
