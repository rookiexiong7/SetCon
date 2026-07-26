import copy
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms as T
import transformers
from PIL import Image
from torch import nn
from torch.nn import CrossEntropyLoss
from torchvision.transforms.functional import InterpolationMode
from transformers import (
    GenerationConfig,
    LlamaForCausalLM,
    Qwen2ForCausalLM,
    Qwen3ForCausalLM,
    Qwen3MoeForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from .configuration_setcon_chat import SetConChatConfig
from .conversation import get_conv_template
from .modeling_intern_vit import InternVisionModel, has_flash_attn
from .sam3runner import SAM3Runner

logger = logging.get_logger(__name__)


def version_cmp(v1, v2, op='eq'):
    import operator
    from packaging import version

    return getattr(operator, op)(version.parse(v1), version.parse(v2))


class StopWordStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, stop_word):
        self.tokenizer = tokenizer
        self.stop_word = stop_word
        self.length = len(stop_word)

    def __call__(self, input_ids, *args, **kwargs) -> bool:
        cur_text = self.tokenizer.decode(input_ids[0])
        cur_text = cur_text.replace('\r', '').replace('\n', '')
        return cur_text[-self.length:] == self.stop_word


def get_stop_criteria(tokenizer, stop_words=None):
    stop_criteria = StoppingCriteriaList()
    for word in stop_words or []:
        stop_criteria.append(StopWordStoppingCriteria(tokenizer, word))
    return stop_criteria


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    }
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))

    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


class DirectResize:
    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        img = Image.fromarray(image.astype(np.uint8), mode='RGB')
        return np.array(img.resize((self.target_length, self.target_length)))


class SetConChatModel(PreTrainedModel):
    config_class = SetConChatConfig
    main_input_name = 'pixel_values'
    base_model_prefix = 'language_model'
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True
    _no_split_modules = ['InternVisionModel', 'Qwen3DecoderLayer', 'SAM3Runner']
    _tp_plan = ''

    def __init__(self, config: SetConChatConfig, vision_model=None, language_model=None, use_flash_attn=True):
        super().__init__(config)

        assert version_cmp(transformers.__version__, '4.37.0', 'ge')
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio ** 2))
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version

        use_flash_attn = use_flash_attn if has_flash_attn else False
        config.vision_config.use_flash_attn = True if use_flash_attn else False
        config.llm_config._attn_implementation = 'flash_attention_2' if use_flash_attn else 'eager'

        self.vision_model = vision_model if vision_model is not None else InternVisionModel(config.vision_config)
        if language_model is not None:
            self.language_model = language_model
        else:
            architecture = config.llm_config.architectures[0]
            if architecture == 'LlamaForCausalLM':
                self.language_model = LlamaForCausalLM(config.llm_config)
            elif architecture == 'Qwen2ForCausalLM':
                self.language_model = Qwen2ForCausalLM(config.llm_config)
            elif architecture == 'Qwen3MoeForCausalLM':
                self.language_model = Qwen3MoeForCausalLM(config.llm_config)
            elif architecture == 'Qwen3ForCausalLM':
                self.language_model = Qwen3ForCausalLM(config.llm_config)
            else:
                raise NotImplementedError(f'{architecture} is not implemented.')

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size
        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )

        self.img_context_token_id = None
        self.conv_template = get_conv_template(self.template)
        self.system_message = self.conv_template.system_message

        self.grounding_encoder = SAM3Runner()
        self.text_hidden_fcs = nn.Sequential(
            nn.Linear(llm_hidden_size, llm_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(llm_hidden_size, self.grounding_encoder.hidden_dim),
            nn.Dropout(0.0),
        )

        self.init_prediction_config = False

    @property
    def lm_head(self):
        return self.language_model.get_output_embeddings()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_input_embeddings(self, value):
        return self.language_model.set_input_embeddings(value)

    def set_output_embeddings(self, value):
        return self.language_model.set_output_embeddings(value)

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(
            n,
            int(h * scale_factor),
            int(w * scale_factor),
            int(c / (scale_factor * scale_factor)),
        )
        if self.ps_version == 'v1':
            warnings.warn(
                "In ps_version 'v1', the height and width have not been swapped back, "
                'which results in a transposed image.'
            )
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True,
            ).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True,
            ).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]
        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        return self.mlp1(vit_embeds)

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_flags: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        image_flags = image_flags.squeeze(-1)
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()
        vit_embeds = self.extract_feature(pixel_values)
        vit_embeds = vit_embeds[image_flags == 1]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)
        flat_input_ids = input_ids.reshape(B * N)
        selected = flat_input_ids == self.img_context_token_id
        input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)
        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        loss = None
        if labels is not None:
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (outputs.logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def generate(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        visual_features: Optional[torch.FloatTensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:
        assert self.img_context_token_id is not None
        device = self.device
        if pixel_values is not None:
            vit_embeds = visual_features if visual_features is not None else self.extract_feature(pixel_values.to(device))
            input_embeds = self.language_model.get_input_embeddings()(input_ids.to(device))
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)
            flat_input_ids = input_ids.reshape(B * N).to(device)
            selected = flat_input_ids == self.img_context_token_id
            if int(selected.sum()) != vit_embeds.reshape(-1, C).shape[0]:
                raise ValueError(
                    f'Image token mismatch: text has {int(selected.sum())} IMG_CONTEXT tokens, '
                    f'vision has {vit_embeds.reshape(-1, C).shape[0]} embeddings.'
                )
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)
            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids.to(device))

        return self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask.to(device),
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            return_dict_in_generate=return_dict,
            use_cache=True,
            **generate_kwargs,
        )

    def preparing_for_generation(self, tokenizer, max_new_tokens=2048, torch_dtype=torch.bfloat16):
        self.tokenizer = tokenizer
        stop_words = []
        if hasattr(self.conv_template, 'sep') and self.conv_template.sep:
            stop_words.append(self.conv_template.sep.strip())
        self.stop_criteria = get_stop_criteria(tokenizer, stop_words)

        self.gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        )
        self.gen_config.transformers_version = transformers.__version__
        self.torch_dtype = torch_dtype

        image_size = self.config.force_image_size or self.config.vision_config.image_size
        self.image_size = image_size
        self.min_dynamic_patch = getattr(self.config, 'min_dynamic_patch', 1)
        self.max_dynamic_patch = getattr(self.config, 'max_dynamic_patch', 12)
        self.use_thumbnail = getattr(self.config, 'use_thumbnail', True)
        self.patch_token = self.num_image_token
        self.IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
        self.IMG_START_TOKEN = '<img>'
        self.IMG_END_TOKEN = '</img>'
        self.img_context_token_id = tokenizer.convert_tokens_to_ids(self.IMG_CONTEXT_TOKEN)
        self.ref_start_token_idx = tokenizer.convert_tokens_to_ids('<ref>')
        self.ref_end_token_idx = tokenizer.convert_tokens_to_ids('</ref>')
        self.transformer = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        self.extra_image_processor = DirectResize(target_length=1024)
        self.init_prediction_config = True

    def _to_chw(self, image):
        if torch.is_tensor(image):
            frame_tensor = image.detach().cpu()
            if frame_tensor.dim() == 3 and frame_tensor.shape[0] == 3:
                return frame_tensor.contiguous()
            if frame_tensor.dim() == 3 and frame_tensor.shape[-1] == 3:
                return frame_tensor.permute(2, 0, 1).contiguous()
            raise ValueError(f'Unsupported tensor image shape: {tuple(frame_tensor.shape)}')
        arr = np.array(image)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(f'Unsupported image shape: {arr.shape}')
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    def _build_query(self, text, image_token_str=''):
        conv = get_conv_template(self.template)
        conv.system_message = self.system_message
        conv.append_message(conv.roles[0], text.replace('<image>', image_token_str, 1))
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt(), conv

    def _extract_seg_tokens(self, generate_output):
        hidden_states = generate_output.hidden_states
        last_hidden_states = [item[-1][0] for item in hidden_states]
        last_hidden_states = torch.cat(last_hidden_states, dim=0)
        output_ids = generate_output.sequences[0][:-1]
        all_label_tokens = get_seg_hidden_states(
            last_hidden_states,
            output_ids,
            ref_start_id=self.ref_start_token_idx,
            ref_end_id=self.ref_end_token_idx,
            return_all_tokens=True,
        )
        all_label_tokens = [self.text_hidden_fcs(tokens) for tokens in all_label_tokens]
        return self._concat_first_with_others(all_label_tokens)

    def _concat_first_with_others(self, all_label_tokens):
        if len(all_label_tokens) <= 1:
            return all_label_tokens
        first_tokens = all_label_tokens[0]
        return [first_tokens] + [torch.cat([first_tokens, tokens], dim=0) for tokens in all_label_tokens[1:]]

    @torch.inference_mode()
    def decode_hidden_states(self, image=None, sam_states=None, seg_hidden_states=None,
                             preprocessed_image=None, original_size=None):
        ret_masks = []
        if (image is None and preprocessed_image is None) or seg_hidden_states is None or len(seg_hidden_states) == 0:
            return ret_masks
        try:
            if sam_states is None:
                if preprocessed_image is not None:
                    # Video path: image is already the tracker's resolution-sized
                    oh, ow = original_size
                    sam_states = self.grounding_encoder.set_image(
                        preprocessed_image, preprocessed=True,
                        original_height=oh, original_width=ow,
                    )
                else:
                    frame_tensor = self._to_chw(image)
                    sam_states = self.grounding_encoder.set_image(frame_tensor)
        except Exception as e:
            print('SAM3 Exception:', e)
            return [
                {'scores': [], 'boxes': [], 'masks': []}
                for _ in range(len(seg_hidden_states))
            ]

        for seg_tokens in seg_hidden_states:
            try:
                outputs = self.grounding_encoder.set_text_prompt([seg_tokens], sam_states)
            except Exception as e:
                print('SAM3 Exception:', e)
                outputs = {'scores': [], 'boxes': [], 'masks': []}
            ret_masks.append(copy.deepcopy({
                'scores': outputs.get('scores', []),
                'boxes': outputs.get('boxes', []),
                'masks': outputs.get('masks', []),
            }))
        return ret_masks

    @torch.inference_mode()
    def infer_hidden_states(self, image=None, video=None, text=None, past_text='', tokenizer=None, processor=None):
        if not self.init_prediction_config:
            if tokenizer is None:
                raise ValueError('tokenizer is required before generation is prepared')
            self.preparing_for_generation(tokenizer=tokenizer)

        text = text or ''
        frames_to_decode = []
        pixel_values = None

        if image is None and video is None and '<image>' not in past_text:
            query_text = (past_text or '') + text.replace('<image>', '')
            query, conv = self._build_query(query_text, '')
        else:
            if video is not None:
                frames = list(video)
                frames_to_decode = frames
                vision_frames = frames[:5]
                pixel_values = torch.stack([self.transformer(frame) for frame in vision_frames], dim=0)
                num_image_tokens = pixel_values.shape[0] * self.patch_token
            else:
                frames_to_decode = [image]
                images = dynamic_preprocess(
                    image,
                    self.min_dynamic_patch,
                    self.max_dynamic_patch,
                    self.image_size,
                    self.use_thumbnail,
                )
                pixel_values = torch.stack([self.transformer(img) for img in images], dim=0)
                num_image_tokens = pixel_values.shape[0] * self.patch_token

            image_token_str = (
                f'{self.IMG_START_TOKEN}'
                f'{self.IMG_CONTEXT_TOKEN * num_image_tokens}'
                f'{self.IMG_END_TOKEN}'
            )
            if '<image>' not in text:
                text = '<image>\n' + text
            query, conv = self._build_query((past_text or '') + text, image_token_str)

        model_inputs = self.tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].to(self.device)
        attention_mask = model_inputs['attention_mask'].to(self.device)
        if pixel_values is not None:
            pixel_values = pixel_values.to(device=self.device, dtype=self.vision_model.dtype)

        sep = getattr(conv, 'sep', None)
        if sep:
            eos_token_id = self.tokenizer.convert_tokens_to_ids(sep.strip())
            if eos_token_id is not None:
                self.gen_config.eos_token_id = eos_token_id

        generate_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=self.gen_config,
            stopping_criteria=self.stop_criteria,
            output_hidden_states=True,
            return_dict=True,
        )
        predict = self.tokenizer.decode(generate_output.sequences[0], skip_special_tokens=False).strip()

        return_dict = {'prediction': predict, 'frames_to_decode': frames_to_decode}
        if image is None and video is None and '<image>' not in past_text:
            return_dict['text_only_mode'] = True
            return_dict['seg_hidden_states'] = []
        else:
            return_dict['text_only_mode'] = False
            return_dict['seg_hidden_states'] = self._extract_seg_tokens(generate_output)
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

        if infer_result['text_only_mode']:
            return {'prediction': predict, 'prediction_dict': []}
        if len(all_label_tokens) == 0:
            return {'prediction': predict, 'prediction_dict': []}

        ret_masks = []
        for frame_idx, frame_data in enumerate(frames_to_process):
            frame_outputs = self.decode_hidden_states(
                image=frame_data,
                seg_hidden_states=all_label_tokens,
            )
            if len(frames_to_process) == 1:
                ret_masks = frame_outputs
            else:
                ret_masks.append({'frame_idx': frame_idx, 'segments': frame_outputs})

        return {'prediction': predict, 'prediction_dict': ret_masks}


def get_seg_hidden_states(hidden_states, output_ids, ref_start_id=None, ref_end_id=None, return_all_tokens=False):
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
    return torch.stack([tokens.mean(dim=0) for tokens in all_label_tokens], dim=0)
