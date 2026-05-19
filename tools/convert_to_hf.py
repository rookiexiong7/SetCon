import argparse
import copy
import os.path as osp
import torch
from mmengine.dist import master_only
from xtuner.registry import BUILDER
from xtuner.configs import cfgs_name_path
from mmengine.config import Config
from mmengine.fileio import PetrelBackend, get_file_backend
from mmengine.config import ConfigDict
import os
import re

def convert_dict2config_dict(input):
    input = ConfigDict(**input)
    for key in input.keys():
        if isinstance(input[key], dict):
            input[key] = convert_dict2config_dict(input[key])
    return input

TORCH_DTYPE_MAP = dict(
    fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32, auto='auto')


def set_config_dtype(config_dict, dtype_name, force_top=True):
    if force_top:
        config_dict["dtype"] = dtype_name
    for key, value in config_dict.items():
        if key in {"dtype", "torch_dtype"}:
            config_dict[key] = dtype_name
        elif isinstance(value, dict):
            set_config_dtype(value, dtype_name, force_top=False)


def cast_floating_module_tensors(module, dtype):
    for param in module.parameters():
        if param.is_floating_point() and param.dtype != dtype:
            param.data = param.data.to(dtype=dtype)
            if param.grad is not None:
                param.grad.data = param.grad.data.to(dtype=dtype)

    for submodule in module.modules():
        for name, buffer in submodule._buffers.items():
            if buffer is not None and buffer.is_floating_point() and buffer.dtype != dtype:
                submodule._buffers[name] = buffer.to(dtype=dtype)


def parse_args():
    parser = argparse.ArgumentParser(description='toHF script')
    parser.add_argument('config', help='config file name or path.')
    parser.add_argument('--pth-model', help='pth model file')
    parser.add_argument(
        '--save-path', type=str, default=None, help='save folder name')
    args = parser.parse_args()
    return args

@master_only
def master_print(msg):
    print(msg)


def copy_weight_from_state_dict(state_dict, keys, target_weight, weight_name):
    for key in keys:
        if key in state_dict:
            target_weight.data.copy_(state_dict[key])
            print(f"Successfully updated {weight_name} weight from key: {key}")
            return True
    print(
        f"Warning: {weight_name} weight key not found in state_dict. "
        f"Tried: {keys}. Keeping the base model weight."
    )
    return False


def main():
    args = parse_args()

    # build model
    if not osp.isfile(args.config):
        try:
            args.config = cfgs_name_path[args.config]
        except KeyError:
            raise FileNotFoundError(f'Cannot find {args.config}')

    # load config
    cfg = Config.fromfile(args.config)
    arch_type = cfg.model.get('arch_type', 'internvl')
    model = BUILDER.build(cfg.model)
    backend = get_file_backend(args.pth_model)

    if isinstance(backend, PetrelBackend):
        state_dict = torch.load(args.pth_model, map_location='cpu', weights_only=False)
    else:
        state_dict = torch.load(args.pth_model, map_location='cpu', weights_only=False)

    state_dict = state_dict['state_dict']

    model.load_state_dict(state_dict, strict=False)
    print(f'Load PTH model from {args.pth_model}')

    if 'qwen' in arch_type:
        lm_head_keys = [
            'mllm.model.base_model.model.lm_head.modules_to_save.default.weight',
            'mllm.model.lm_head.weight',
        ]
        embed_tokens_keys = [
            'mllm.model.base_model.model.model.language_model.embed_tokens.modules_to_save.default.weight',
            'mllm.model.model.language_model.embed_tokens.weight',
        ]
    else:
        lm_head_keys = [
            'mllm.model.language_model.base_model.model.lm_head.modules_to_save.default.weight',
            'mllm.model.language_model.lm_head.weight',
        ]
        embed_tokens_keys = [
            'mllm.model.language_model.base_model.model.model.embed_tokens.modules_to_save.default.weight',
            'mllm.model.language_model.model.embed_tokens.weight',
        ]

    print("Force updating lm_head and embed_tokens weights from state_dict when available.")
    copy_weight_from_state_dict(
        state_dict, lm_head_keys, model.mllm.model.get_output_embeddings().weight, 'lm_head'
    )
    copy_weight_from_state_dict(
        state_dict, embed_tokens_keys, model.mllm.model.get_input_embeddings().weight, 'embed_tokens'
    )

    iter_str = os.path.basename(args.pth_model).split('.')[0]

    model._merge_lora()

    model.mllm.model.modules_to_save = None
    if hasattr(model.mllm.model, 'language_model'):
        # for internvl only; qwen has been fixed in mllm folder
        model.mllm.model.language_model.modules_to_save = None
    model.mllm.model.transfer_to_hf = True

    all_state_dict = model.all_state_dict()

    all_state_dict_new = {}

    # build the hf format model
    from projects.setcon.hf.models.configuration_setcon_chat import SetConChatConfig
    from projects.setcon.hf.models.modeling_setcon_chat import SetConChatModel

    if 'qwen3' in cfg.path.lower():
        from projects.setcon.hf.models_qwen3vl.configuration_setcon_chat import SetConChatConfigQwen
        from projects.setcon.hf.models_qwen3vl.modeling_setcon_qwen import SetConChatModelQwen

    print("arch_type:", arch_type)
    print(cfg.model)

    if 'qwen' not in arch_type:
        config = SetConChatConfig.from_pretrained(cfg.path)
    else:
        config = SetConChatConfigQwen.from_pretrained(cfg.path)
    
    config_dict = config.to_dict()
    set_config_dtype(config_dict, "bfloat16")
    
    if 'qwen' in arch_type:
        config_dict["text_config"]["vocab_size"] = len(model.mllm.tokenizer)
        config_dict["tie_word_embeddings"] = False
    else:
        config_dict["llm_config"]["vocab_size"] = len(model.mllm.tokenizer)

    # Handle Jinja template modification for Qwen models
    template_str = cfg.template
    if 'qwen' in arch_type:
        print("Qwen model detected. Removing system prompt from Jinja template.")
        system_prompt_pattern = re.compile(
            r"{% if loop\.first and message\['role'] != 'system' %}.*?<\|im_end\|>\s*{% endif %}",
            re.DOTALL
        )
        template_str = system_prompt_pattern.sub('', template_str)

    config_dict["template"] = template_str


    if 'qwen' in arch_type:
        # for qwen
        name_map = {'mllm.': '', }
            #'.gamma': '.g_weight'}
        for key in all_state_dict.keys():
            new_key = copy.deepcopy(key)
            for _text in name_map.keys():
                new_key = new_key.replace(_text, name_map[_text])
            all_state_dict_new[new_key] = all_state_dict[key]

        config_dict['auto_map'] = \
        {'AutoConfig': 'configuration_setcon_chat.SetConChatConfigQwen',
         'AutoModel': 'modeling_setcon_qwen.SetConChatModelQwen',
         'AutoModelForCausalLM': 'modeling_setcon_qwen.SetConChatModelQwen'}

        setcon_hf_config = SetConChatConfigQwen(**config_dict)
        setcon_hf_config.text_config.tie_word_embeddings = False

        setcon_hf_config.save_pretrained("./tmp/setcon_config_test_qwen")

    else:
        name_map = {'mllm.model.': '', }
        # '.gamma': '.g_weight'}

        for key in all_state_dict.keys():
            new_key = copy.deepcopy(key)
            for _text in name_map.keys():
                new_key = new_key.replace(_text, name_map[_text])
            all_state_dict_new[new_key] = all_state_dict[key]
        
        config_dict['auto_map'] = \
        {'AutoConfig': 'configuration_setcon_chat.SetConChatConfig',
         'AutoModel': 'modeling_setcon_chat.SetConChatModel',
         'AutoModelForCausalLM': 'modeling_setcon_chat.SetConChatModel'}
        
        setcon_hf_config = SetConChatConfig(**config_dict)

    if 'qwen' in arch_type:
        # for qwen
        hf_setcon_model = SetConChatModelQwen(
            setcon_hf_config, model=model.mllm.model
        )
    else:
        hf_setcon_model = SetConChatModel(
            setcon_hf_config, vision_model=model.mllm.model.vision_model,
            language_model=model.mllm.model.language_model,
        )

    if '_training_step' in all_state_dict_new:
        all_state_dict_new.pop('_training_step')
    missing_keys, unexpected_keys = hf_setcon_model.load_state_dict(all_state_dict_new)

    if args.save_path is None:
        args.save_path = f"./{os.path.dirname(args.pth_model)}_{iter_str}_hf"
    
    # SAM3 stores rotary frequency caches as complex64 tensors. Current
    # transformers/safetensors loading rejects C64, while torch .bin can load it.
    cast_floating_module_tensors(hf_setcon_model, torch.bfloat16)
    hf_setcon_model.save_pretrained(args.save_path, safe_serialization=False)

    if 'qwen' in arch_type:
        model.mllm.processor.save_pretrained(args.save_path)
    else:
        model.mllm.tokenizer.save_pretrained(args.save_path)

    master_print("\n--- Weight Loading Report ---")
    if missing_keys:
        master_print(f"Warning: Missing keys: {missing_keys}")
    if unexpected_keys:
        master_print(f"Warning: Unexpected keys: {unexpected_keys}")
    if not missing_keys and not unexpected_keys:
        master_print("All keys matched successfully!")

    print(f"Save the model into {args.save_path}")

if __name__ == '__main__':
    main()
