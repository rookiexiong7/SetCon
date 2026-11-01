import os

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from mmengine.model import BaseModule
from torchvision.transforms import v2
from xtuner.registry import BUILDER

from transformers import GroundingDinoForObjectDetection

from third_parts.sam3.model.model_misc import SAM3Output
from third_parts.sam3.model.box_ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy, masks_to_boxes
from third_parts.sam3.model.data_misc import FindStage, BatchedFindTarget, convert_my_tensors
from third_parts.sam3.model.utils.misc import copy_data_to_device
from third_parts.sam3.train.data.collator import pad_tensor_list_to_longest, packed_to_padded_naive

GD_DEFAULT = os.environ.get('SETCON_GD_PRETRAINED_PATH', 'IDEA-Research/grounding-dino-base')


def build_text_tensors(language_embd, device, dtype, hidden_dim):
    """List[Tensor[len_i, D]] (already projected to d_model) -> the text-side
    tensors Grounding DINO's encoder/decoder expect, but sourced from the LVLM
    embeddings instead of the BERT text backbone.

    Returns dict with:
      text_features            (B, L, D)   padded language embeddings
      text_token_mask          (B, L)      True = valid token
      text_self_attention_masks(B, L, L)   True = attend (full within concept)
      position_ids             (B, L)      arange over valid, 0 on pad
    """
    B = len(language_embd)
    lengths = [t.shape[0] for t in language_embd]
    max_len = max(lengths) if B > 0 else 1

    text_features = torch.zeros(B, max_len, hidden_dim, device=device, dtype=dtype)
    text_token_mask = torch.zeros(B, max_len, device=device, dtype=torch.bool)
    position_ids = torch.zeros(B, max_len, device=device, dtype=torch.long)
    for i, (t, n) in enumerate(zip(language_embd, lengths)):
        text_features[i, :n] = t.to(dtype)
        text_token_mask[i, :n] = True
        position_ids[i, :n] = torch.arange(n, device=device)

    # Full self-attention among valid tokens of the same concept; diagonal always on.
    pair = text_token_mask[:, None, :] & text_token_mask[:, :, None]
    eye = torch.eye(max_len, device=device, dtype=torch.bool)[None]
    text_self_attention_masks = pair | eye
    return {
        "text_features": text_features,
        "text_token_mask": text_token_mask,
        "text_self_attention_masks": text_self_attention_masks,
        "position_ids": position_ids,
    }


def gd_aspect_ratio_size(h, w, shortest_edge=800, longest_edge=1333):
    """GD-native resize target: scale the shortest edge to `shortest_edge`
    keeping aspect ratio, but if that pushes the longest edge past
    `longest_edge`, clamp the longest edge instead. Mirrors
    `get_size_with_aspect_ratio` in GroundingDinoImageProcessor. Returns (oh, ow)."""
    min_orig, max_orig = float(min(h, w)), float(max(h, w))
    size = shortest_edge
    if max_orig / min_orig * size > longest_edge:
        size = int(round(longest_edge * min_orig / max_orig))
    if (h <= w and h == size) or (w <= h and w == size):
        return h, w
    if w < h:
        return int(size * h / w), size
    return size, int(size * w / h)


def resize_and_pad_batch(images, normalize, device, shortest_edge=800,
                         longest_edge=1333, size_divisor=32):
    """List of HWC/CHW image tensors -> (pixel_values (B,3,Hmax,Wmax),
    pixel_mask (B,Hmax,Wmax) with 1=real 0=pad). Each image is resized keeping
    aspect ratio (GD-native), then all are padded (bottom/right) to the batch
    max, rounded up to a multiple of size_divisor."""
    resized = []
    sizes = []
    for image in images:
        img = v2.functional.to_image(image)
        _, h, w = img.shape
        oh, ow = gd_aspect_ratio_size(h, w, shortest_edge, longest_edge)
        img = v2.functional.resize(img, size=[oh, ow])
        img = normalize(img)
        resized.append(img)
        sizes.append((oh, ow))

    max_h = max(s[0] for s in sizes)
    max_w = max(s[1] for s in sizes)

    def _round(x):
        return int((x + size_divisor - 1) // size_divisor * size_divisor)

    max_h, max_w = _round(max_h), _round(max_w)

    B = len(resized)
    pixel_values = torch.zeros(B, 3, max_h, max_w, dtype=torch.float32, device=device)
    pixel_mask = torch.zeros(B, max_h, max_w, dtype=torch.long, device=device)
    for i, (img, (oh, ow)) in enumerate(zip(resized, sizes)):
        pixel_values[i, :, :oh, :ow] = img.to(device)
        pixel_mask[i, :oh, :ow] = 1
    return pixel_values, pixel_mask


def encode_vision(gd, pixel_values, pixel_mask=None):
    """Run only Grounding DINO's vision path (backbone + input projections +
    flatten + level embeddings), returning the tensors the encoder needs. No
    text is required here — early fusion happens inside the encoder.

    pixel_mask: (B, H, W) with 1 = real pixel, 0 = padding (DETR/GD convention).
    Defaults to all-ones (no padding) for backward compatibility."""
    model = gd.model
    cfg = gd.config
    batch_size, _, height, width = pixel_values.shape
    device = pixel_values.device
    if pixel_mask is None:
        pixel_mask = torch.ones((batch_size, height, width), dtype=torch.long, device=device)

    vision_features, position_embeddings_list = model.backbone(pixel_values, pixel_mask)

    feature_maps = []
    masks = []
    for level, (source, mask) in enumerate(vision_features):
        feature_maps.append(model.input_proj_vision[level](source))
        masks.append(mask)

    if cfg.num_feature_levels > len(feature_maps):
        _len = len(feature_maps)
        for level in range(_len, cfg.num_feature_levels):
            if level == _len:
                source = model.input_proj_vision[level](vision_features[-1][0])
            else:
                source = model.input_proj_vision[level](feature_maps[-1])
            mask = nn.functional.interpolate(
                pixel_mask[None].float(), size=source.shape[-2:]
            ).to(torch.bool)[0]
            pos_l = model.backbone.position_embedding(source, mask).to(source.dtype)
            feature_maps.append(source)
            masks.append(mask)
            position_embeddings_list.append(pos_l)

    source_flatten = []
    mask_flatten = []
    lvl_pos_embed_flatten = []
    spatial_shapes_list = []
    for level, (source, mask, pos_embed) in enumerate(
        zip(feature_maps, masks, position_embeddings_list)
    ):
        bs, c, h, w = source.shape
        spatial_shapes_list.append((h, w))
        source = source.flatten(2).transpose(1, 2)
        mask = mask.flatten(1)
        pos_embed = pos_embed.flatten(2).transpose(1, 2)
        lvl_pos_embed = pos_embed + model.level_embed[level].view(1, 1, -1)
        lvl_pos_embed_flatten.append(lvl_pos_embed)
        source_flatten.append(source)
        mask_flatten.append(mask)
    source_flatten = torch.cat(source_flatten, 1)
    mask_flatten = torch.cat(mask_flatten, 1)
    lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
    spatial_shapes = torch.as_tensor(
        spatial_shapes_list, dtype=torch.long, device=source_flatten.device
    )
    level_start_index = torch.cat(
        (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
    )
    valid_ratios = torch.stack([model.get_valid_ratio(m) for m in masks], 1).float()

    return {
        "source_flatten": source_flatten,
        "mask_flatten": mask_flatten,
        "lvl_pos_embed_flatten": lvl_pos_embed_flatten,
        "spatial_shapes": spatial_shapes,
        "spatial_shapes_list": spatial_shapes_list,
        "level_start_index": level_start_index,
        "valid_ratios": valid_ratios,
    }


def run_grounding(gd, state, text):
    """Replicate GroundingDinoModel.forward from the encoder onwards, using the
    injected `text` (from build_text_tensors) in place of the BERT output, then
    apply the detection heads. Returns (pred_logits (B,Q,1), pred_boxes (B,Q,4)
    cxcywh in [0,1])."""
    model = gd.model
    cfg = gd.config

    source_flatten = state["source_flatten"]
    mask_flatten = state["mask_flatten"]
    lvl_pos_embed_flatten = state["lvl_pos_embed_flatten"]
    spatial_shapes = state["spatial_shapes"]
    spatial_shapes_list = state["spatial_shapes_list"]
    level_start_index = state["level_start_index"]
    valid_ratios = state["valid_ratios"]

    text_features = text["text_features"]
    text_token_mask = text["text_token_mask"]
    text_self_attention_masks = text["text_self_attention_masks"]
    position_ids = text["position_ids"]
    batch_size = source_flatten.shape[0]

    encoder_outputs = model.encoder(
        vision_features=source_flatten,
        vision_attention_mask=~mask_flatten,
        vision_position_embedding=lvl_pos_embed_flatten,
        spatial_shapes=spatial_shapes,
        spatial_shapes_list=spatial_shapes_list,
        level_start_index=level_start_index,
        valid_ratios=valid_ratios,
        text_features=text_features,
        text_attention_mask=~text_token_mask,
        text_position_embedding=None,
        text_self_attention_masks=~text_self_attention_masks,
        text_position_ids=position_ids,
        return_dict=True,
    )
    enc_vision = encoder_outputs.last_hidden_state_vision
    enc_text = encoder_outputs.last_hidden_state_text

    # Two-stage query init (config.two_stage=True, embedding_init_target=True).
    query_embeds = model.query_position_embeddings.weight
    object_query_embedding, output_proposals = model.generate_encoder_output_proposals(
        enc_vision, ~mask_flatten, spatial_shapes
    )
    enc_outputs_class = model.encoder_output_class_embed(
        object_query_embedding, enc_text, text_token_mask
    )
    delta_bbox = model.encoder_output_bbox_embed(object_query_embedding)
    enc_outputs_coord_logits = delta_bbox + output_proposals

    topk = cfg.num_queries
    topk_logits = enc_outputs_class.max(-1)[0]
    topk_proposals = torch.topk(topk_logits, topk, dim=1)[1]
    topk_coords_logits = torch.gather(
        enc_outputs_coord_logits, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
    ).detach()
    reference_points = topk_coords_logits.sigmoid()
    init_reference_points = reference_points
    target = query_embeds.unsqueeze(0).repeat(batch_size, 1, 1)

    decoder_outputs = model.decoder(
        inputs_embeds=target,
        vision_encoder_hidden_states=enc_vision,
        vision_encoder_attention_mask=mask_flatten,
        text_encoder_hidden_states=enc_text,
        text_encoder_attention_mask=~text_token_mask,
        reference_points=reference_points,
        spatial_shapes=spatial_shapes,
        spatial_shapes_list=spatial_shapes_list,
        level_start_index=level_start_index,
        valid_ratios=valid_ratios,
        self_attn_mask=None,
        return_dict=True,
    )

    hidden_states = decoder_outputs.intermediate_hidden_states
    inter_references = decoder_outputs.intermediate_reference_points
    num_levels = hidden_states.shape[1]

    outputs_class = None
    outputs_coord = None
    for level in range(num_levels):
        if level == 0:
            reference = init_reference_points
        else:
            reference = inter_references[:, level - 1]
        reference = torch.special.logit(reference, eps=1e-5)
        logits = gd.class_embed[level](
            vision_hidden_state=hidden_states[:, level],
            text_hidden_state=enc_text,
            text_token_mask=text_token_mask,
        )
        delta = gd.bbox_embed[level](hidden_states[:, level])
        coord = (delta + reference).sigmoid()
        outputs_class, outputs_coord = logits, coord  # keep last layer

    # Per-token contrastive logits (B,Q,max_text_len), -inf on padding
    # -> max-pool over valid tokens -> (B,Q,1).
    pred_logits = outputs_class.max(-1, keepdim=True).values
    pred_boxes = outputs_coord  # (B,Q,4) cxcywh in [0,1]
    return pred_logits, pred_boxes


class GroundingDinoTrainRunner(BaseModule):
    """Language-conditioned Grounding DINO as a SetCon grounding encoder.

    Mirrors SAM3TrainRunner: preprocess_image_batch /
    inject_language_embd / configure_trainable / state_dict_save_prefixes /
    hidden_dim. The LVLM `language_embd` replaces GD's BERT text-encoder output
    (spliced in place of `text_features`), so language still flows through GD's
    encoder early-fusion, decoder text cross-attention, and contrastive scoring
    head. Boxes are GD's native differentiable outputs (cxcywh, sigmoid); the
    per-token contrastive logits are max-pooled over valid tokens to (B, Q, 1)
    to feed the SAM3 matcher/loss. Masks are NOT produced here (box-only
    supervision); the mask is generated at inference time by SAM3.
    """

    def __init__(
        self,
        loss_fn,
        pretrained_path: str = GD_DEFAULT,
        image_mean=(0.485, 0.456, 0.406),
        image_std=(0.229, 0.224, 0.225),
        shortest_edge: int = 800,
        longest_edge: int = 1333,
        size_divisor: int = 32,
        grad_checkpointing: bool = True,
    ):
        super().__init__(init_cfg=None)

        self.gd = GroundingDinoForObjectDetection.from_pretrained(pretrained_path)
        cfg = self.gd.config
        self.hidden_dim = cfg.d_model
        self.max_text_len = cfg.max_text_len
        self.grad_checkpointing = grad_checkpointing

        # GD-native preprocessing: keep aspect ratio (shortest edge -> 800,
        # longest edge capped at 1333), then pad a batch to a common size that is
        # a multiple of size_divisor. Resize is per-image (variable size), so the
        # transform here is normalize-only; resize/pad happen in
        # preprocess_image_batch.
        self.shortest_edge = shortest_edge
        self.longest_edge = longest_edge
        self.size_divisor = size_divisor
        self.normalize = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=list(image_mean), std=list(image_std)),
        ])

        self.build_loss_fn(loss_fn)

    # ---- SetCon grounding-encoder interface ----

    def configure_trainable(self):
        """Freeze the heavy vision conv backbone and the (now unused) BERT text
        backbone + text projection; train the vision input projections, encoder
        (early fusion), decoder, two-stage proposal heads, and scoring/box heads.
        """
        self.requires_grad_(True)
        self.gd.model.backbone.requires_grad_(False)
        self.gd.model.text_backbone.requires_grad_(False)
        self.gd.model.text_projection.requires_grad_(False)
        if self.grad_checkpointing:
            self._enable_encoder_checkpointing()

    def _enable_encoder_checkpointing(self):
        """Recompute each encoder layer's activations during backward instead of
        storing them. GD's encoder does NOT support HF's native
        gradient_checkpointing, and it is the encoder (6 layers over ~21760
        vision tokens per expanded image) that dominates training memory
        (~5.6 GB/image without this, ~1.7 GB/image with it). We patch each
        layer's `forward` in place — no module nesting — so state_dict key
        paths stay `gd.model.encoder.layers.<i>.*` and checkpoints remain
        loadable by the (non-checkpointed) inference runner."""
        for layer in self.gd.model.encoder.layers:
            if getattr(layer, "_ckpt_patched", False):
                continue
            orig_forward = layer.forward

            def make(fwd, current_layer):
                def wrapped(*args, **kwargs):
                    if current_layer.training and torch.is_grad_enabled():
                        return checkpoint.checkpoint(
                            lambda *a: fwd(*a, **kwargs), *args, use_reentrant=False
                        )
                    return fwd(*args, **kwargs)
                return wrapped

            layer.forward = make(orig_forward, layer)
            layer._ckpt_patched = True

    def state_dict_save_prefixes(self):
        return (
            'gd.model.input_proj_vision',
            'gd.model.query_position_embeddings',
            'gd.model.encoder',
            'gd.model.decoder',
            'gd.model.level_embed',
            'gd.model.enc_output',
            'gd.model.enc_output_norm',
            'gd.model.encoder_output_bbox_embed',
            'gd.class_embed',
            'gd.bbox_embed',
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def build_loss_fn(self, loss_fn):
        self.loss_fn = BUILDER.build(loss_fn)
        self.loss_fn.matcher = BUILDER.build(self.loss_fn.matcher)
        self.loss_fn.o2m_matcher = BUILDER.build(self.loss_fn.o2m_matcher)
        modules = []
        for loss in self.loss_fn.loss_fns_find:
            modules.append(BUILDER.build(loss))
        self.loss_fn.loss_fns_find = modules

    # ---- vision-only encoding (no text needed until early fusion) ----

    def preprocess_image_batch(self, images, state=None):
        if state is None:
            state = {}
        if not isinstance(images, list):
            raise ValueError("Images must be a list of tensors")
        assert len(images) > 0, "At least one image must be provided"

        pixel_values, pixel_mask = resize_and_pad_batch(
            images, self.normalize, self.device,
            self.shortest_edge, self.longest_edge, self.size_divisor,
        )

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            state.update(encode_vision(self.gd, pixel_values, pixel_mask))
        return state

    # ---- text injection + encoder/decoder + heads ----

    def inject_language_embd(self, gt_masks, sam_states, language_embd):
        if not isinstance(language_embd, list):
            raise ValueError("language_embd must be a list")

        device = sam_states["source_flatten"].device
        dtype = sam_states["source_flatten"].dtype
        text = build_text_tensors(language_embd, device, dtype, self.hidden_dim)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred_logits, pred_boxes = run_grounding(self.gd, sam_states, text)

        pred_boxes = pred_boxes.float()
        out = {
            "pred_logits": pred_logits.float(),
            "pred_boxes": pred_boxes,
            "pred_boxes_xyxy": box_cxcywh_to_xyxy(pred_boxes),
        }

        find_target_back = self._build_targets(gt_masks)
        out["indices"] = self.loss_fn.matcher(out, find_target_back)

        previous = SAM3Output(iter_mode=SAM3Output.IterMode.LAST_STEP_PER_STAGE)
        previous.append([out])
        loss = self.loss_fn(previous, [find_target_back])["core_loss"]
        return loss

    def _build_targets(self, gt_masks):
        find_stage, find_target = self._construct_dataitem(gt_masks)
        find_target = copy_data_to_device(find_target, self.device, non_blocking=True)
        packed_boxes = find_target.boxes.view(-1, 4)
        find_target_back = {
            "boxes": packed_boxes,
            "boxes_xyxy": box_cxcywh_to_xyxy(packed_boxes),
            "boxes_padded": find_target.boxes_padded,
            "positive_map": find_target.boxes.new_ones(len(find_target.boxes), 1),
            "num_boxes": find_target.num_boxes,
            "masks": find_target.segments,
            "semantic_masks": [],
            "is_valid_mask": find_target.is_valid_segment,
            "is_exhaustive": find_target.is_exhaustive,
            "object_ids_packed": find_target.object_ids,
            "object_ids_padded": find_target.object_ids_padded,
        }
        return find_target_back

    def _construct_dataitem(self, gt_masks):
        B = len(gt_masks)
        stage_object_ids = [list(range(m.shape[0])) for m in gt_masks]
        find_stage = FindStage(
            img_ids=list(range(B)),
            text_ids=list(range(B)),
            input_boxes=pad_tensor_list_to_longest(
                [torch.zeros(0, 4, device=self.device) for _ in range(B)], dim=0, pad_val=0
            ),
            input_boxes_label=pad_tensor_list_to_longest(
                [torch.zeros(0, dtype=torch.long, device=self.device) for _ in range(B)], dim=0, pad_val=0
            ),
            input_boxes_mask=pad_tensor_list_to_longest(
                [torch.ones(0, dtype=torch.bool, device=self.device) for _ in range(B)], dim=0, pad_val=1
            ),
            input_points=pad_tensor_list_to_longest(
                [torch.empty(0, 257, device=self.device) for _ in range(B)], dim=0, pad_val=0
            ),
            input_points_mask=pad_tensor_list_to_longest(
                [torch.empty(0, device=self.device) for _ in range(B)], dim=0, pad_val=1
            ),
            object_ids=stage_object_ids,
        )

        num_boxes = [m.shape[0] for m in gt_masks]
        gt_boxes_list = [masks_to_boxes(m) for m in gt_masks]
        boxes, segments, tgt_object_ids = [], [], []
        for b in range(B):
            n = gt_masks[b].shape[0]
            for i in range(n):
                box = box_xyxy_to_cxcywh(gt_boxes_list[b][i])
                cur_h, cur_w = gt_masks[b][i].shape
                box = box / torch.tensor([cur_w, cur_h, cur_w, cur_h]).to(box.device)
                boxes.append(box)
                segments.append(gt_masks[b][i])
                tgt_object_ids.append(i)

        find_target = BatchedFindTarget(
            num_boxes=num_boxes,
            boxes=boxes,
            boxes_padded=[],
            is_exhaustive=[True] * B,
            segments=segments,
            semantic_segments=[],
            is_valid_segment=[1] * len(boxes),
            repeated_boxes=[],
            object_ids=tgt_object_ids,
            object_ids_padded=[],
        )
        find_stage = convert_my_tensors(find_stage)
        find_target = convert_my_tensors(find_target)
        find_target.boxes_padded = packed_to_padded_naive(
            find_target.boxes.view(-1, 4), find_target.num_boxes
        )
        find_target.object_ids_padded = packed_to_padded_naive(
            find_target.object_ids, find_target.num_boxes, fill_value=-1
        )
        return find_stage, find_target

    def forward(self, batch):
        raise NotImplementedError
