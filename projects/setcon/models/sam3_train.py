import os.path

import torch
from mmengine.model import BaseModule
from torchvision.transforms import v2
from xtuner.registry import BUILDER

from third_parts.sam3.model.geometry_encoders import Prompt
from third_parts.sam3.model.data_misc import FindStage, BatchedFindTarget, convert_my_tensors
from third_parts.sam3.model.box_ops import box_xyxy_to_cxcywh, masks_to_boxes
from third_parts.sam3.model.utils.misc import copy_data_to_device
from third_parts.sam3.train.data.collator import pad_tensor_list_to_longest, packed_to_padded_naive

BASE_DIR = 'pretrained/sam3'

class SAM3TrainRunner(BaseModule):
    """ SAM3 Training Runner """
    
    def __init__(
        self, loss_fn,
        bpe_path: str = "third_parts/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        checkpoint_path: str = os.path.join(BASE_DIR, "sam3.pt"),
        eval_mode: bool = False,
    ):
        super().__init__(init_cfg=None)

        import third_parts.sam3  # noqa: F401
        from third_parts.sam3.model_builder import build_sam3_image_model
        
        self.sam3_model  = build_sam3_image_model(
            bpe_path=bpe_path, checkpoint_path=checkpoint_path, eval_mode=eval_mode
        )
        
        self.resolution = 1008
        self.transform = v2.Compose([
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(size=(self.resolution, self.resolution)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        self.device = self.sam3_model.device
        self.hidden_dim = self.sam3_model.backbone.language_backbone.d_model

        self.build_loss_fn(loss_fn)
    
    def build_loss_fn(self, loss_fn):
        self.loss_fn = BUILDER.build(loss_fn)
        self.loss_fn.matcher = BUILDER.build(self.loss_fn.matcher)
        self.loss_fn.o2m_matcher = BUILDER.build(self.loss_fn.o2m_matcher)
        
        modules = []
        for loss in self.loss_fn.loss_fns_find:
            loss_module = BUILDER.build(loss)
            modules.append(loss_module)
        self.loss_fn.loss_fns_find = modules

    def construct_sam3_dataitem(self, gt_masks, repeats: int = 1, input_points_embedding_dim: int = 257):
        """ 
        Prepare find_stage and find_target inputs for SAM3 model.
        Args:
            gt_masks: List[Tensor(num_obj, H, W)] 
        Returns:
            find_stage: FindStage
            find_target: BatchedFindTarget
        """
        B = len(gt_masks)
        
        # ---- FindStage (inputs) ----
        img_ids  = [i for i in range(B)]
        text_ids = [i for i in range(B)]
        
        input_boxes       = [torch.zeros(0, 4, device=self.device) for _ in range(B)]
        input_boxes_label = [torch.zeros(0, dtype=torch.long, device=self.device) for _ in range(B)]
        input_boxes_mask  = [torch.ones(0, dtype=torch.bool, device=self.device) for _ in range(B)]  # pad_val=1

        input_points      = [torch.empty(0, input_points_embedding_dim, device=self.device) for _ in range(B)]
        input_points_mask = [torch.empty(0, device=self.device) for _ in range(B)]  # pad_val=1
        
        # local object index per item in the batch
        stage_object_ids = [list(range(m.shape[0])) for m in gt_masks]

        # pad to longest
        input_points      = pad_tensor_list_to_longest(input_points, dim=0, pad_val=0)
        input_points_mask = pad_tensor_list_to_longest(input_points_mask, dim=0, pad_val=1)

        input_boxes       = pad_tensor_list_to_longest(input_boxes, dim=0, pad_val=0)
        input_boxes_label = pad_tensor_list_to_longest(input_boxes_label, dim=0, pad_val=0)
        input_boxes_mask  = pad_tensor_list_to_longest(input_boxes_mask, dim=0, pad_val=1)

        find_stage = FindStage(
            img_ids=img_ids,
            text_ids=text_ids,
            input_boxes=input_boxes,
            input_boxes_label=input_boxes_label,
            input_boxes_mask=input_boxes_mask,
            input_points=input_points,
            input_points_mask=input_points_mask,
            object_ids=stage_object_ids,
        )

        # ---- BatchedFindTarget (targets) ----
        num_boxes = [m.shape[0] for m in gt_masks]  # per query
        
        # packed boxes / packed masks
        gt_boxes_list = [masks_to_boxes(m) for m in gt_masks]          # List[(n_i,4)]
        boxes, segments, tgt_object_ids, repeated_boxes = [], [], [], []
        for b in range(B):
            n = gt_masks[b].shape[0]
            current_boxes, current_segments = [], []
            for i in range(n):
                # normalize
                box = box_xyxy_to_cxcywh(gt_boxes_list[b][i])
                cur_h, cur_w = gt_masks[b][i].shape
                box = box / torch.tensor([cur_w, cur_h, cur_w, cur_h]).to(box.device)
                current_boxes.append(box)
                current_segments.append(gt_masks[b][i])
                tgt_object_ids.append(i)
            # append to the global list
            boxes.extend(current_boxes)                           # List[(4)] * sum(n_i)
            segments.extend(current_segments)                     # List[(H, W)] * sum(n_i)
            if repeats > 0:
                for _ in range(repeats):
                    repeated_boxes.extend(current_boxes)

        is_valid_segment = [1] * len(boxes)                            # per object (packed)
        is_exhaustive = [True] * B                                     # per query

        find_target = BatchedFindTarget(
            num_boxes=num_boxes,
            boxes=boxes,
            boxes_padded=[],
            is_exhaustive=is_exhaustive,
            segments=segments,
            semantic_segments=[],
            is_valid_segment=is_valid_segment,
            repeated_boxes=repeated_boxes if repeats > 0 else [],
            object_ids=tgt_object_ids,
            object_ids_padded=[],
        )
        # to tensors + padded views
        find_stage  = convert_my_tensors(find_stage)
        find_target = convert_my_tensors(find_target)

        find_target.boxes_padded = packed_to_padded_naive(
            find_target.boxes.view(-1, 4), find_target.num_boxes
        )
        find_target.object_ids_padded = packed_to_padded_naive(
            find_target.object_ids, find_target.num_boxes, fill_value=-1
        )

        return find_stage, find_target

    def preprocess_image_batch(self, images, state=None) -> torch.Tensor:
        if state is None:
            state = {}

        if not isinstance(images, list):
            raise ValueError("Images must be a list of tensors")
        assert len(images) > 0, "At least one image must be provided"

        images = [
            self.transform(v2.functional.to_image(image))
            for image in images
        ]
        images = torch.stack(images, dim=0)
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            state["backbone_out"] = self.sam3_model.backbone.forward_image(images)
        
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

    def inject_language_embd(self, gt_masks, sam_states, language_embd):
        """
        Inject language embeddings into SAM3 for grounding.
        
        Args:
            sam_states: SAM3 backbone states
            language_embd: Language embeddings, List[Tensor[label_len, hidden_dim]]: multi-token per segment
        """
        from third_parts.sam3.model.model_misc import SAM3Output
        
        if isinstance(language_embd, list):
            # New format: List[Tensor[label_len, hidden_dim]]
            B = len(language_embd)
            if B == 0:
                # Handle empty case
                device = sam_states["backbone_out"]["vision_features"].device
                dtype = sam_states["backbone_out"]["vision_features"].dtype
                hidden_dim = sam_states["backbone_out"]["vision_features"].shape[-1]
                language_features = torch.zeros(1, 0, hidden_dim, device=device, dtype=dtype)
                language_mask = torch.zeros(0, 1, device=device, dtype=torch.bool)
            else:
                label_lengths = [tokens.shape[0] for tokens in language_embd]
                max_label_len = max(label_lengths)
                hidden_dim = language_embd[0].shape[-1]
                device = language_embd[0].device
                dtype = language_embd[0].dtype
                
                # Pad to [max_label_len, B, hidden_dim]
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

        sam_states["backbone_out"].update(text_output)

        find_stage, find_target = self.construct_sam3_dataitem(gt_masks)
        find_stage = copy_data_to_device(find_stage, self.device, non_blocking=True)
        find_target = copy_data_to_device(find_target, self.device, non_blocking=True)
        find_targets = [
            copy_data_to_device(
                self.sam3_model.back_convert(find_target),
                self.device,
                non_blocking=True,
            )
        ]

        geometric_prompt = Prompt(
            box_embeddings=find_stage.input_boxes,
            box_mask=find_stage.input_boxes_mask,
            box_labels=find_stage.input_boxes_label,
        )
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = self.sam3_model.forward_grounding(
                backbone_out=sam_states["backbone_out"],
                find_input=find_stage,
                geometric_prompt=geometric_prompt,
                find_target=find_target,
            )
        
        previous_stages_out = SAM3Output(iter_mode=SAM3Output.IterMode.LAST_STEP_PER_STAGE)
        previous_stages_out.append([outputs])
        
        if len(self.loss_fn.loss_fns_find) > 0:
            iter_loss = self.loss_fn(previous_stages_out, find_targets)['core_loss']
        else:
            iter_loss = 0.0
        
        return iter_loss

    def forward(self, batch):
        raise NotImplementedError
