from typing import Literal
from collections import OrderedDict
from pycocotools import mask as _mask
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from mmengine.model import BaseModel
from xtuner.registry import BUILDER

from peft import PeftModelForCausalLM

from transformers import AutoImageProcessor, AutoVideoProcessor


class SetConModel(BaseModel):
    def __init__(self,
                 mllm,
                 tokenizer,
                 grounding_encoder,
                 torch_dtype=torch.bfloat16,
                 pretrained_pth=None,
                 frozen_sam2_decoder=True,
                 special_tokens=None,
                 loss_sample_points=False,
                 num_points=12544,
                 template=None,
                 # for arch selection
                 arch_type:Literal['intern_vl', 'qwen', 'llava']='intern_vl',
                 # ext
                 # preprocessor=None,
                 # bs
                 training_bs:int=0,
                 # decoder warmup: freeze decoder for the first N steps
                 decoder_warmup_steps:int=0,
                 ):
        super().__init__()
        if special_tokens is None:
            special_tokens = ['<ref>', '</ref>']

        self.mllm = BUILDER.build(mllm)
        self.arch_type = arch_type

        tokenizer = BUILDER.build(tokenizer)
        self.tokenizer = tokenizer
        self._add_special_tokens(tokenizer, special_tokens)

        if arch_type == 'qwen':
            image_processor = AutoImageProcessor.from_pretrained(mllm['model_path'], trust_remote_code=True)
            video_processor = AutoVideoProcessor.from_pretrained(mllm['model_path'], trust_remote_code=True)
            self.mllm._init_processor(image_processor, video_processor)

        self.grounding_encoder = BUILDER.build(grounding_encoder)
        self.grounding_encoder.requires_grad_(False)
        self.grounding_encoder.sam3_model.transformer.requires_grad_(True)
        self.grounding_encoder.sam3_model.transformer.decoder.requires_grad_(False)

        # FIX: Untie weights for Qwen model
        if self.arch_type == 'qwen' and self.mllm.model.config.tie_word_embeddings:
            print("Untying embed_tokens and lm_head weights for Qwen model.")
            self.mllm.model.config.tie_word_embeddings = False
            lm_head = self.mllm.model.get_output_embeddings()
            if lm_head is not None:
                input_embeddings = self.mllm.model.get_input_embeddings()
                lm_head.weight = nn.Parameter(input_embeddings.weight.clone())

        in_dim = self.mllm.get_embedding_size()
        out_dim = self.grounding_encoder.hidden_dim
        self.text_hidden_fcs = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim), nn.Dropout(0.0)
        )

        self.torch_dtype = torch_dtype
        
        if self.arch_type == 'qwen' and self.mllm.use_llm_lora:
            self.mllm.manual_prepare_llm_for_lora()
        if self.arch_type == 'intern_vl' and self.mllm.llm_lora_config:
            self.mllm.manual_prepare_llm_for_lora()
        
        if pretrained_pth is not None:
            pretrained_state_dict = torch.load(pretrained_pth, map_location='cpu', weights_only=False)['state_dict']
            self.load_state_dict(pretrained_state_dict, strict=False)
            # self.mllm.model.modules_to_save = None
            # if hasattr(self.mllm.model, 'language_model'):
            #     # for internvl only; qwen has been fixed in mllm folder
            #     self.mllm.model.language_model.modules_to_save = None
            # print(f'Load pretrained weight from {pretrained_pth}')

            if self.arch_type == 'qwen':
                print("Force updating lm_head weight from pretrained state_dict.")
                # lm_head_key = 'mllm.model.lm_head.weight'
                lm_head_key = 'mllm.model.base_model.model.lm_head.modules_to_save.default.weight'
                if lm_head_key in pretrained_state_dict:
                    lm_head_weight = pretrained_state_dict[lm_head_key]
                    self.mllm.model.get_output_embeddings().weight.data.copy_(lm_head_weight)
                    print(f"Successfully updated lm_head weight from key: {lm_head_key}")
                else:
                    print(f"Warning: lm_head weight key '{lm_head_key}' not found in pretrained_state_dict.")
                
                emd_key = 'mllm.model.base_model.model.model.language_model.embed_tokens.modules_to_save.default.weight'
                if emd_key in pretrained_state_dict:
                    emd_weight = pretrained_state_dict[emd_key]
                    self.mllm.model.get_input_embeddings().weight.data.copy_(emd_weight)
                    print(f"Successfully updated emd weight from key: {emd_key}")
                else:
                    print(f"Warning: emd weight key '{emd_key}' not found in pretrained_state_dict.")

        self.loss_sample_points = loss_sample_points
        self.num_points = num_points
        self.oversample_ratio = 3.0
        self.importance_sample_ratio = 0.75

        self.template = template
        self.bs = training_bs

        # self.mllm.model.requires_grad_(False)
        # self.text_hidden_fcs.requires_grad_(False)
        # Print gradient status of all weights in self.mllm.model.base_model.model
        print("\n" + "="*80)
        print("GRADIENT STATUS OF MLLM.MODEL WEIGHTS")
        print("="*80)
        
        try:
            base_model = self.mllm.model
            total_params = 0
            trainable_params = 0
            
            for name, param in base_model.named_parameters():
                total_params += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
                    grad_status = "✓ TRAINABLE"
                else:
                    grad_status = "✗ FROZEN"
                
                print(f"{name:<60} | {grad_status} | Shape: {tuple(param.shape)} | Params: {param.numel():,}")
            
            print("-" * 80)
            print(f"SUMMARY:")
            print(f"  Total parameters: {total_params:,}")
            print(f"  Trainable parameters: {trainable_params:,}")
            print(f"  Frozen parameters: {total_params - trainable_params:,}")
            print(f"  Trainable ratio: {trainable_params/total_params*100:.2f}%")
            print("=" * 80)
            
        except Exception as e:
            print(f"Failed to access self.mllm.model: {e}")
            print("Available attributes in self.mllm.model:")
            print([attr for attr in dir(self.mllm.model) if not attr.startswith('_')])


    def _add_special_tokens(self, tokenizer, special_tokens):
        self.mllm.add_special_tokens(tokenizer, special_tokens)
        self.ref_start_token_idx = tokenizer("<ref>", add_special_tokens=False).input_ids[0]
        self.ref_end_token_idx = tokenizer("</ref>", add_special_tokens=False).input_ids[0]

    def _extract_ref_embeddings(self, input_ids, hidden_states, _zero):
        """
        Extract embeddings for <ref>label</ref> format.
        
        Returns:
            all_label_tokens: List[Tensor[*, hidden_dim]]
                If use_mean_pooling=True: List[Tensor[1, hidden_dim]] - mean pooled
                If use_mean_pooling=False: List[Tensor[label_len, hidden_dim]]
            seg_token_counts: [batch_size]
        """
        batch_size, _, _ = hidden_states.shape
        
        all_label_tokens = []  # List of [label_len, hidden_dim] tensors
        seg_counts = []
        
        for b in range(batch_size):
            sample_ids = input_ids[b]
            sample_hidden = hidden_states[b]
            
            # Find all <ref> and </ref> positions
            ref_start_positions = (sample_ids == self.ref_start_token_idx).nonzero(as_tuple=True)[0]
            ref_end_positions = (sample_ids == self.ref_end_token_idx).nonzero(as_tuple=True)[0]
            
            # Number of segments
            num_refs = min(len(ref_start_positions), len(ref_end_positions))
            seg_counts.append(num_refs)

            # Extract embeddings for each <ref>...</ref> pair
            for i in range(num_refs):
                start_pos = ref_start_positions[i].item()
                end_pos = ref_end_positions[i].item()
                
                # Get hidden states of tokens between <ref> and </ref> (exclusive)
                if end_pos > start_pos + 1:
                    label_hidden = sample_hidden[start_pos + 1:end_pos] + _zero  # [label_len, hidden_dim]
                else:
                    # Empty label, use <ref> token's hidden state as fallback
                    label_hidden = (sample_hidden[start_pos] + _zero).unsqueeze(0)  # [1, hidden_dim]
                
                all_label_tokens.append(label_hidden)
        
        seg_token_counts = torch.tensor(seg_counts, dtype=torch.int, device=input_ids.device)
        

        return all_label_tokens, seg_token_counts

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        return super().load_state_dict(state_dict, strict, assign)

    def _merge_lora(self):
        if isinstance(self.mllm.model, PeftModelForCausalLM):
            self.mllm.model = self.mllm.model.merge_and_unload()
            return
        
        try:
            self.mllm.model.language_model = self.mllm.model.language_model.merge_and_unload()
        except:
            print("Skip language model, no LoRA in it !!!")
        try:
            self.mllm.model.vision_model = self.mllm.model.vision_model.merge_and_unload()
        except:
            print("Skip vision encoder, no LoRA in it !!!")
        return

    def all_state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        return state_dict

    def state_dict(self, *args, **kwargs):
        prefix = kwargs.pop('prefix', '')
        state_dict_mllm = self.mllm.state_dict(*args, prefix=prefix + 'mllm.', **kwargs)
        state_dict_sam2 = self.grounding_encoder.state_dict(*args, prefix=prefix + 'grounding_encoder.', **kwargs)
        state_dict_text = self.text_hidden_fcs.state_dict(*args, prefix=prefix + 'text_hidden_fcs.', **kwargs)
        to_return = OrderedDict()
        to_return.update(state_dict_mllm)
        to_return.update(
            {k: v
             for k, v in state_dict_sam2.items() if k.startswith('grounding_encoder.sam3_model')})
        to_return.update(state_dict_text)
        return to_return

    def check_obj_number(self, pred_embeddings_list_video, gt_masks_video, fix_number=5):
        assert len(pred_embeddings_list_video) == len(gt_masks_video)
        ret_pred_embeddings_list_video = []
        ret_gt_masks_video = []
        for pred_mebeds, gt_masks in zip(pred_embeddings_list_video, gt_masks_video):
            # assert len(pred_mebeds) == len(gt_masks)
            if len(pred_mebeds) != len(gt_masks):
                min_num = min(len(pred_mebeds), len(gt_masks))
                pred_mebeds = pred_mebeds[:min_num]
                gt_masks = gt_masks[:min_num]
            if len(pred_mebeds) != fix_number:
                if len(pred_mebeds) > fix_number:
                    _idxs = torch.randperm(pred_mebeds.shape[0])
                    _idxs = _idxs[:fix_number]
                    pred_mebeds = pred_mebeds[_idxs]
                    gt_masks = gt_masks[_idxs]
                else:
                    n_repeat = fix_number // len(pred_mebeds) + 1
                    pred_mebeds = torch.cat([pred_mebeds] * n_repeat, dim=0)[:fix_number]
                    gt_masks = torch.cat([gt_masks] * n_repeat, dim=0)[:fix_number]
            ret_pred_embeddings_list_video.append(pred_mebeds)
            ret_gt_masks_video.append(gt_masks)
        return ret_pred_embeddings_list_video, ret_gt_masks_video

    def _get_pesudo_data(self, dtype, device):
        g_pixel_values = torch.zeros((3, 1024, 1024), dtype=dtype, device=device)
        g_pixel_values = [g_pixel_values] * self.bs
        frames_per_batch = [1] * self.bs
        gt_masks = torch.zeros((5, 256, 256), dtype=torch.uint8, device=device)
        gt_masks = [gt_masks] * self.bs
        return g_pixel_values, frames_per_batch, gt_masks

    def _expand_supervision_and_build_aux(
        self,
        g_pixel_values,
        gt_masks,
        frames_per_batch,
        tokens_per_sample,
    ):
        """Expand per-sample supervision to per-frame/per-token."""
        gt_masks_expand = []           # List[expanded_tokens, Tensor[num_obj, 288, 288]]
        g_pixel_values_expand = []     # List[expanded_tokens, Tensor[3, H, W]]
        label_tokens_expand = []       # List[expanded_tokens, Tensor[num_query, D_sam]]
        token_groups = []              # List[num_groups, List[token_indices]]

        current_idx = 0
        current_frame = 0
        for i_bs, n_frames in enumerate(frames_per_batch):
            sample_frames = g_pixel_values[current_frame: current_frame + n_frames]  # List[n_frames, Tensor[3, H, W]]
            current_frame += n_frames

            gt_masks_per_token = gt_masks[i_bs]      # List[num_tokens_gt, Tensor[F_sel,N_obj,H,W]]
            sample_tokens = tokens_per_sample[i_bs]  # List[num_tokens_q, Tensor[num_query, D_sam]]

            num_tokens_gt = len(gt_masks_per_token)
            num_tokens_q = len(sample_tokens)
            use_num_tokens = min(num_tokens_gt, num_tokens_q)
            if num_tokens_gt != num_tokens_q:
                print(f"[Warning] Batch {i_bs}: GT tokens={num_tokens_gt} vs Query tokens={num_tokens_q}, truncate to {use_num_tokens}.")
            if use_num_tokens == 0:
                continue

            resized_masks_per_token = []  # List[num_tokens, List[n_frames, Tensor[num_obj,288,288]]]
            for tok_i in range(use_num_tokens):
                gt_mask = gt_masks_per_token[tok_i] # Tensor[F_sel,N_obj,H,W]
                tok_masks_by_frame = []

                for f_idx in range(n_frames):
                    if f_idx < gt_mask.shape[0]:
                        cur_mask = gt_mask[f_idx]   # Tensor[num_obj,288,288]
                    else:
                        cur_mask = torch.zeros((0, gt_mask.shape[-2], gt_mask.shape[-1]), device=gt_mask.device)
                    if cur_mask.shape[0] == 0:
                        kept_resized = torch.zeros((0, 288, 288), dtype=gt_mask.dtype, device=gt_mask.device)
                    else:
                        resized = F.interpolate(cur_mask.unsqueeze(0).to(torch.float32), size=(288, 288), mode='nearest').squeeze(0)
                        resized = (resized > 0.5).to(gt_mask.dtype)
                        valid_obj = resized.reshape(resized.shape[0], -1).sum(dim=1) > 0
                        kept_resized = resized[valid_obj] if valid_obj.any() else resized[:0]
                    tok_masks_by_frame.append(kept_resized)

                resized_masks_per_token.append(tok_masks_by_frame)

            for f_idx, frame_img in enumerate(sample_frames):
                group_indices = list(range(current_idx, current_idx + use_num_tokens))
                token_groups.append(group_indices)
                # For tokens in the same group, mean-pool the first token-group
                # to a single token only when concatenating with following token-groups.
                for tok_i in range(use_num_tokens):
                    g_pixel_values_expand.append(frame_img)
                    frame_token_gt_288 = resized_masks_per_token[tok_i][f_idx]  # Tensor[num_obj, 288, 288]
                    gt_masks_expand.append(frame_token_gt_288)
                    if tok_i == 0:
                        cur_label_tokens = sample_tokens[0]
                    else:
                        cur_label_tokens = torch.cat([sample_tokens[0], sample_tokens[tok_i]], dim=0)

                    label_tokens_expand.append(cur_label_tokens)
                current_idx += use_num_tokens

        assert current_frame == len(g_pixel_values), \
            f"frames_per_batch mismatch: consumed {current_frame}, got {len(g_pixel_values)} frames"

        return g_pixel_values_expand, gt_masks_expand, label_tokens_expand, token_groups

    def forward(self, data, data_samples=None, mode='loss'):

        # 1) Run MLLM and prepare segmentation supervision inputs
        g_pixel_values = data.pop('g_pixel_values', None)
        gt_masks = data.pop('masks', None)
        frames_per_batch = data.pop('frames_per_batch', None)
        input_ids = data['input_ids']
        output = self.mllm(data, data_samples, mode)
        if gt_masks is None:
            # require zero seg datas
            seg_valid = False
            g_pixel_values, frames_per_batch, gt_masks = self._get_pesudo_data(
                dtype=self.torch_dtype,
                device=input_ids.device,
            )
        else:
            seg_valid = True

        hidden_states = output.hidden_states[-1]
        _zero = hidden_states.mean() * 0.0

        # 2) Parse language prompts: <ref>...</ref>
        has_ref_format = (input_ids == self.ref_start_token_idx).any() and (input_ids == self.ref_end_token_idx).any()

        all_label_tokens = None
        seg_token_counts = None

        if seg_valid:
            if has_ref_format:
                ref_tokens, ref_counts = self._extract_ref_embeddings(input_ids, hidden_states, _zero)
                all_label_tokens = [self.text_hidden_fcs(tokens) for tokens in ref_tokens]
                seg_token_counts = ref_counts
            else:
                seg_token_counts = torch.zeros(input_ids.shape[0], dtype=torch.int, device=input_ids.device)
                seg_token_counts += 5
                fallback_embeddings = self.text_hidden_fcs(hidden_states[:, :5].flatten(0, 1) + _zero)
                all_label_tokens = [fallback_embeddings[i:i + 1] for i in range(fallback_embeddings.shape[0])]
        else:
            seg_token_counts = torch.zeros(input_ids.shape[0], dtype=torch.int, device=input_ids.device)
            seg_token_counts += 5
            fallback_embeddings = self.text_hidden_fcs(hidden_states[:, :5].flatten(0, 1) + _zero) # Tensor[num_seg_tokens, hidden_dim]
            all_label_tokens = [fallback_embeddings[i:i + 1] for i in range(fallback_embeddings.shape[0])]

        # 3) Expand query/mask supervision to EVERY frame for video/multi-image inputs.
        # gt_masks: List[B][num_tokens, Tensor[num_obj, H, W] or Tensor[H, W]]
        # all_label_tokens: List[sum(num_tokens), Tensor[num_query, D_sam]]
        tokens_per_sample = []
        token_offset = 0
        for i_bs in range(input_ids.shape[0]):
            n_tok = int(seg_token_counts[i_bs].item())
            tokens_per_sample.append(all_label_tokens[token_offset: token_offset + n_tok])
            token_offset += n_tok

        expanded_data = self._expand_supervision_and_build_aux(
            g_pixel_values=g_pixel_values,
            gt_masks=gt_masks,
            frames_per_batch=frames_per_batch,
            tokens_per_sample=tokens_per_sample,
        )
        g_pixel_values_expand, gt_masks_expand, label_tokens_expand, token_groups = expanded_data
        # import pdb; pdb.set_trace()
        if len(g_pixel_values_expand) == 0:
            # No valid token-frame pair in this batch.
            loss_sam3 = hidden_states.mean() * 0.0
        else:
            sam_states = self.grounding_encoder.preprocess_image_batch(g_pixel_values_expand)

            # 4) Inject language embeddings into grounding encoder (expanded list format)
            if len(label_tokens_expand) != len(gt_masks_expand):
                raise RuntimeError(
                    f"Token/GT mismatch after frame expansion: tokens={len(label_tokens_expand)} vs gt={len(gt_masks_expand)}."
                )
            loss_sam3 = self.grounding_encoder.inject_language_embd(gt_masks_expand, sam_states, label_tokens_expand)

        # 5) Scale segmentation loss and return
        loss_sam3 = loss_sam3 * (1.0 if seg_valid else 0.0)

        loss_dict = {
            'loss_sam3': loss_sam3,
            'llm_loss': output.loss,
        }
        return loss_dict


    def generate_video_pred_embeddings(self, pred_embeddings_list, frames_per_batch):
        assert len(pred_embeddings_list) == len(frames_per_batch)
        pred_embeddings_list_video = []
        for pred_embedding_batch, frame_nums in zip(pred_embeddings_list, frames_per_batch):
            pred_embeddings_list_video += [pred_embedding_batch] * frame_nums
        return pred_embeddings_list_video

    def process_video_gt_masks(self, gt_masks, frames_per_batch):
        gt_masks_video = []

        assert len(gt_masks) == len(frames_per_batch)
        for gt_masks_batch, frames_num in zip(gt_masks, frames_per_batch):
            N, H, W = gt_masks_batch.shape
            assert N % frames_num == 0
            gt_masks_batch = gt_masks_batch.reshape(
                N // frames_num, frames_num, H, W)
            for i in range(frames_num):
                gt_masks_video.append(gt_masks_batch[:, i])
        return gt_masks_video

    def preparing_for_generation(self, metainfo, **kwargs):
        raise NotImplementedError("SetCon does not support preparing for generation, please use predict_video instead.")

def mask_to_rle(mask):
    rle = []
    for m in mask:
        rle.append(_mask.encode(np.asfortranarray(m.astype(np.uint8))))
        rle[-1]['counts'] = rle[-1]['counts'].decode()
    return rle
