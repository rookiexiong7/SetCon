import copy
import torch
from torch import nn
from transformers import Qwen3VLForConditionalGeneration
from transformers.modeling_utils import PreTrainedModel

from .configuration_setcon_chat import SetConChatConfigQwen

from .sam3runner import SAM3Runner

import numpy as np

from qwen_vl_utils import process_vision_info

class SetConChatModelQwen(PreTrainedModel):
    config_class = SetConChatConfigQwen
    main_input_name = 'pixel_values'
    base_model_prefix = 'language_model'
    _no_split_modules = ['Qwen3VisionTransformerPretrainedModel', 'Qwen3VLDecoderLayer', 'SAM2']
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True



    def __init__(self, config: SetConChatConfigQwen, model=None, use_flash_attn=True):
        super().__init__(config)
        # self.extra_image_processor = DirectResize(target_length=1024, )

        self.min_pixels = 512 * 28 * 28
        self.max_pixels = 2048 * 28 * 28

        self.torch_dtype = torch.bfloat16

        if model is not None:
            self.model=model
        else:
            self.model = Qwen3VLForConditionalGeneration(config)

        llm_hidden_size = config.text_config.hidden_size

        self.grounding_encoder = SAM3Runner()
        
        out_dim = self.grounding_encoder.hidden_dim
        in_dim = llm_hidden_size
        self.text_hidden_fcs = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim), nn.Dropout(0.0)
        )

    @property
    def lm_head(self):
        return self.model.lm_head

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.model.get_output_embeddings()

    def _generate(self, messages, processor_kwargs):
        processed_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        mm_inputs = self.processor(
            text=[processed_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **processor_kwargs,
        ).to(self.device)

        generate_output = self.model.generate(
            **mm_inputs,
            max_new_tokens=2048,
            do_sample=False,
            output_hidden_states=True,
            return_dict_in_generate=True
        )

        generate_output_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(mm_inputs.input_ids, generate_output.sequences)
        ]
        predict = self.processor.batch_decode(generate_output_trimmed, skip_special_tokens=False)[0].strip()
        return generate_output, predict

    def _to_chw(self, image):
        if torch.is_tensor(image):
            frame_tensor = image.detach().cpu()
            if frame_tensor.dim() == 3 and frame_tensor.shape[0] == 3:
                return frame_tensor.contiguous()
            if frame_tensor.dim() == 3 and frame_tensor.shape[-1] == 3:
                return frame_tensor.permute(2, 0, 1).contiguous()
            raise ValueError(f"Unsupported tensor image shape: {tuple(frame_tensor.shape)}")
        arr = np.array(image)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(f"Unsupported image shape: {arr.shape}")
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    def _extract_seg_tokens(self, generate_output):
        # Return List[num_labels, Tensor(num_tokens, hidden_dim)]
        hidden_states = generate_output.hidden_states
        last_hidden_states = [item[-1][0] for item in hidden_states]
        last_hidden_states = torch.cat(last_hidden_states, dim=0)

        output_ids = generate_output.sequences[0][:-1]
        has_ref_format = (output_ids == self.ref_start_token_idx).any() and (output_ids == self.ref_end_token_idx).any()

        if not has_ref_format:
            return []

        all_label_tokens = get_seg_hidden_states(
            last_hidden_states,
            output_ids,
            ref_start_id=self.ref_start_token_idx,
            ref_end_id=self.ref_end_token_idx,
            return_all_tokens=True,
        )
        all_label_tokens = [self.text_hidden_fcs(tokens) for tokens in all_label_tokens]
        all_label_tokens = self._concat_first_with_others(all_label_tokens)
        return all_label_tokens

    def _concat_first_with_others(self, all_label_tokens):
        """Mean-pool first token-group to 1 token; concat it with each following group."""
        if len(all_label_tokens) <= 1:
            return all_label_tokens

        first_tokens = all_label_tokens[0]
        first_tokens_mean = first_tokens.mean(dim=0, keepdim=True)
        merged_tokens = [first_tokens]
        for idx in range(1, len(all_label_tokens)):
            merged_tokens.append(torch.cat([first_tokens, all_label_tokens[idx]], dim=0))
        return merged_tokens

    @torch.inference_mode()
    def decode_hidden_states(self, image=None, sam_states=None, seg_hidden_states=None):
        ret_masks = []
        if image is None:
            return ret_masks
        if seg_hidden_states is None or len(seg_hidden_states) == 0:
            return ret_masks

        frame_tensor = self._to_chw(image)
        # import pdb; pdb.set_trace()
        try:
            if sam_states is None:
                sam_states = self.grounding_encoder.set_image(frame_tensor)
            # outputs = self.grounding_encoder.set_text_prompt(seg_hidden_states, sam_states)
            
        except Exception as e:
            print("SAM3 Exception:", e)
            outputs = {'scores': [], 'boxes': [], 'masks': []}

        # ret_masks.append(outputs)
        for i in range(len(seg_hidden_states)):
            seg_tokens = [seg_hidden_states[i]]
            try:
                if sam_states is None:
                    raise RuntimeError("SAM3 image state is unavailable.")
                outputs = self.grounding_encoder.set_text_prompt(seg_tokens, sam_states)
            except Exception as e:
                print("SAM3 Exception:", e)
                outputs = {'scores': [], 'boxes': [], 'masks': []}

            ret_masks.append(copy.deepcopy({
                'scores': outputs.get('scores', []),
                'boxes': outputs.get('boxes', []),
                'masks': outputs.get('masks', []),
            }))

        return ret_masks

    @torch.inference_mode()
    def infer_hidden_states(
            self,
            image=None,
            video=None,
            text=None,
            past_text='',
            tokenizer=None,
            processor=None,
    ):
        # Set processors and tokens
        self.processor = processor
        self.ref_start_token_idx = self.processor.tokenizer.convert_tokens_to_ids('<ref>')
        self.ref_end_token_idx = self.processor.tokenizer.convert_tokens_to_ids('</ref>')

        # Prepare messages and processor kwargs
        text = (text or "").replace('<image>', "")
        frames_to_decode = []

        if image is None and video is None and '<image>' not in past_text: # Text Mode
            messages = [{"role": "user", "content": [{"type": "text", "text": past_text + text}]}]
            processor_kwargs = {}
        elif video is not None: # Video Mode
            frames_to_decode = list(video)
            content = [{"type": "image", "image": frame} for frame in video]
            content.append({"type": "text", "text": text})
            messages = [{"role": "user", "content": content}]
            processor_kwargs = {"min_pixels": self.min_pixels, "max_pixels": self.max_pixels}
        else: # Image Mode
            frames_to_decode = [image]
            messages = [{
                "role": "user",
                "content": [{"type": "image", "image": image}, {"type": "text", "text": text}]
            }]
            processor_kwargs = {"min_pixels": self.min_pixels, "max_pixels": self.max_pixels}

        # Generate and decode
        generate_output, predict = self._generate(messages, processor_kwargs)
        return_dict = {
            'prediction': predict,
            'frames_to_decode': frames_to_decode,
        }
        if image is None and video is None and '<image>' not in past_text: # Text Mode
            return_dict['text_only_mode'] = True
            return_dict['seg_hidden_states'] = []
        else:
            all_label_tokens = self._extract_seg_tokens(generate_output)
            return_dict['text_only_mode'] = False
            return_dict['seg_hidden_states'] = all_label_tokens
        return return_dict
    
    @torch.inference_mode()
    def predict_forward(
            self,
            image=None,
            video=None,
            text=None,
            past_text='',
            mask_prompts=None,
            tokenizer=None,
            processor=None,
    ):
        infer_result = self.infer_hidden_states(
            image=image,
            video=video,
            text=text,
            past_text=past_text,
            tokenizer=tokenizer,
            processor=processor,
        )
        predict = infer_result['prediction']
        all_label_tokens = infer_result['seg_hidden_states']
        frames_to_process = infer_result['frames_to_decode']
        text_only_mode = infer_result['text_only_mode']
        ret_masks = []

        if text_only_mode:
            return {'prediction': predict, 'prediction_dict': ret_masks}

        num_segs = len(all_label_tokens)
        
        if num_segs == 0:
            return {'prediction': predict, 'prediction_dict': ret_masks}
        
        for frame_idx, frame_data in enumerate(frames_to_process):
            frame_outputs = self.decode_hidden_states(
                image=frame_data,
                seg_hidden_states=all_label_tokens,
            )

            # For single image, keep backward-compatible flat output: List[seg_outputs]
            if len(frames_to_process) == 1:
                ret_masks = frame_outputs
            else:
                ret_masks.append({
                    'frame_idx': frame_idx,
                    'segments': frame_outputs,
                })
        
        return {'prediction': predict, 'prediction_dict': ret_masks}
        

def get_seg_hidden_states(hidden_states, output_ids, ref_start_id=None, ref_end_id=None, return_all_tokens=False):
    """Extract hidden states from <ref>label</ref> spans."""
    if ref_start_id is None or ref_end_id is None:
        raise ValueError('ref_start_id and ref_end_id are required')

    n_out = len(output_ids)
    hidden_states = hidden_states[-n_out:]

    start_positions = (output_ids == ref_start_id).nonzero(as_tuple=True)[0]
    end_positions = (output_ids == ref_end_id).nonzero(as_tuple=True)[0]
    num_segs = min(len(start_positions), len(end_positions))

    if num_segs == 0:
        return [] if return_all_tokens else hidden_states[0:0]

    all_label_tokens = []
    for i in range(num_segs):
        start_pos = start_positions[i].item()
        end_pos = end_positions[i].item()
        if end_pos > start_pos + 1:
            label_hidden = hidden_states[start_pos + 1:end_pos]
        else:
            label_hidden = hidden_states[start_pos:start_pos + 1]
        all_label_tokens.append(label_hidden)

    if return_all_tokens:
        return all_label_tokens

    seg_embeddings = [tokens.mean(dim=0) for tokens in all_label_tokens]
    return torch.stack(seg_embeddings, dim=0)
