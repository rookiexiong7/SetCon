from typing import Optional

import torch
from torch import nn
from transformers import AutoTokenizer

from .modeling_setcon_chat import SetConChatModel
from .video_detector_adapter import SetConInternVLVideoDetectorAdapter
from third_parts.sam3.model.model_misc import DotProductScoring, MLP
from third_parts.sam3.model.sam3_image import Sam3ImageOnVideoMultiGPU
from third_parts.sam3.model.sam3_video_inference import (
    Sam3VideoInferenceWithInstanceInteractivity,
)
from third_parts.sam3.model_builder import (
    _create_geometry_encoder,
    _create_sam3_transformer,
    _create_segmentation_head,
    _create_text_encoder,
    _create_vision_backbone,
    build_tracker,
)


def _load_sam3_ckpt_for_tracker_and_features(
    model: Sam3VideoInferenceWithInstanceInteractivity,
    sam3_checkpoint_path: str,
):
    ckpt = torch.load(sam3_checkpoint_path, map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]

    detector_sd = {
        k.replace("detector.", ""): v for k, v in ckpt.items() if k.startswith("detector.")
    }
    tracker_sd = {
        k.replace("tracker.", ""): v for k, v in ckpt.items() if k.startswith("tracker.")
    }

    if len(detector_sd) > 0:
        model.detector.feature_detector.load_state_dict(detector_sd, strict=False)
    if len(tracker_sd) > 0:
        model.tracker.load_state_dict(tracker_sd, strict=False)


def _build_sam3_feature_detector(
    bpe_path: str,
    has_presence_token: bool,
) -> Sam3ImageOnVideoMultiGPU:
    visual_neck = _create_vision_backbone()
    text_encoder = _create_text_encoder(bpe_path)
    from third_parts.sam3.model.vl_combiner import SAM3VLBackbone

    backbone = SAM3VLBackbone(scalp=1, visual=visual_neck, text=text_encoder)
    transformer = _create_sam3_transformer(has_presence_token=has_presence_token)
    segmentation_head = _create_segmentation_head()
    input_geometry_encoder = _create_geometry_encoder()

    main_dot_prod_mlp = MLP(
        input_dim=256,
        hidden_dim=2048,
        output_dim=256,
        num_layers=2,
        dropout=0.1,
        residual=True,
        out_norm=nn.LayerNorm(256),
    )
    main_dot_prod_scoring = DotProductScoring(
        d_model=256,
        d_proj=256,
        prompt_mlp=main_dot_prod_mlp,
    )
    return Sam3ImageOnVideoMultiGPU(
        num_feature_levels=1,
        backbone=backbone,
        transformer=transformer,
        segmentation_head=segmentation_head,
        semantic_segmentation_head=None,
        input_geometry_encoder=input_geometry_encoder,
        use_early_fusion=True,
        use_dot_prod_scoring=True,
        dot_prod_scoring=main_dot_prod_scoring,
        supervise_joint_box_scores=has_presence_token,
    )


def _build_internvl_model_bundle(
    model_name_or_path: str,
    device,
    tokenizer_name_or_path: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path or model_name_or_path,
        trust_remote_code=True,
    )
    model = SetConChatModel.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )
    model.grounding_encoder.sam3_model.to(device)
    model.grounding_encoder.device = device
    model.grounding_encoder.set_confidence_threshold(0.5)
    model.to(device)
    model.eval()
    return model, tokenizer


def build_setcon_video_model(
    model_name_or_path: str,
    sam3_checkpoint_path: Optional[str] = None,
    processor_name_or_path: Optional[str] = None,
    tokenizer_name_or_path: Optional[str] = None,
    model: Optional[nn.Module] = None,
    processor=None,
    tokenizer=None,
    bpe_path: Optional[str] = None,
    has_presence_token: bool = True,
    apply_temporal_disambiguation: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype: Optional[torch.dtype] = None,
    compile: bool = False,
) -> Sam3VideoInferenceWithInstanceInteractivity:
    del processor_name_or_path

    tracker = build_tracker(apply_temporal_disambiguation=apply_temporal_disambiguation)
    feature_detector = _build_sam3_feature_detector(
        bpe_path=bpe_path,
        has_presence_token=has_presence_token,
    )

    if model is None or tokenizer is None:
        model, tokenizer = _build_internvl_model_bundle(
            model_name_or_path=model_name_or_path,
            device=device,
            tokenizer_name_or_path=tokenizer_name_or_path,
            torch_dtype=torch_dtype,
        )

    detector = SetConInternVLVideoDetectorAdapter(
        feature_detector=feature_detector,
        setcon_model=model,
        processor=processor,
        tokenizer=tokenizer,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
    )

    if apply_temporal_disambiguation:
        video_model = Sam3VideoInferenceWithInstanceInteractivity(
            detector=detector,
            tracker=tracker,
            score_threshold_detection=0.5,
            assoc_iou_thresh=0.1,
            det_nms_thresh=0.1,
            new_det_thresh=0.7,
            hotstart_delay=15,
            hotstart_unmatch_thresh=8,
            hotstart_dup_thresh=8,
            suppress_unmatched_only_within_hotstart=True,
            min_trk_keep_alive=-1,
            max_trk_keep_alive=30,
            init_trk_keep_alive=30,
            suppress_overlapping_based_on_recent_occlusion_threshold=0.7,
            suppress_det_close_to_boundary=False,
            fill_hole_area=16,
            recondition_every_nth_frame=16,
            masklet_confirmation_enable=False,
            decrease_trk_keep_alive_for_empty_masklets=False,
            image_size=1008,
            image_mean=(0.5, 0.5, 0.5),
            image_std=(0.5, 0.5, 0.5),
            compile_model=compile,
        )
    else:
        video_model = Sam3VideoInferenceWithInstanceInteractivity(
            detector=detector,
            tracker=tracker,
            score_threshold_detection=0.5,
            assoc_iou_thresh=0.1,
            det_nms_thresh=0.1,
            new_det_thresh=0.7,
            hotstart_delay=0,
            hotstart_unmatch_thresh=0,
            hotstart_dup_thresh=0,
            suppress_unmatched_only_within_hotstart=True,
            min_trk_keep_alive=-1,
            max_trk_keep_alive=30,
            init_trk_keep_alive=30,
            suppress_overlapping_based_on_recent_occlusion_threshold=0.7,
            suppress_det_close_to_boundary=False,
            fill_hole_area=16,
            recondition_every_nth_frame=0,
            masklet_confirmation_enable=False,
            decrease_trk_keep_alive_for_empty_masklets=False,
            image_size=1008,
            image_mean=(0.5, 0.5, 0.5),
            image_std=(0.5, 0.5, 0.5),
            compile_model=compile,
        )

    video_model.to(device=device)
    if sam3_checkpoint_path:
        _load_sam3_ckpt_for_tracker_and_features(
            model=video_model,
            sam3_checkpoint_path=sam3_checkpoint_path,
        )
    return video_model
