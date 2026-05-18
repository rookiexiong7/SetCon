from third_parts.sam3.model.sam3_video_predictor import (
    Sam3VideoPredictor as _Sam3VideoPredictor,
    Sam3VideoPredictorMultiGPU as _Sam3VideoPredictorMultiGPU,
)


class SetConVideoPredictor(_Sam3VideoPredictor):
    def __init__(
        self,
        qwen_model_name_or_path,
        sam3_checkpoint_path=None,
        processor_name_or_path=None,
        tokenizer_name_or_path=None,
        qwen_model=None,
        qwen_processor=None,
        qwen_tokenizer=None,
        bpe_path=None,
        has_presence_token=True,
        geo_encoder_use_img_cross_attn=True,
        strict_state_dict_loading=True,
        async_loading_frames=False,
        video_loader_type="cv2",
        apply_temporal_disambiguation=True,
        device="cuda",
        torch_dtype=None,
        compile=False,
    ):
        self.async_loading_frames = async_loading_frames
        self.video_loader_type = video_loader_type
        from .video_predictor_builder import build_setcon_video_model

        self.model = (
            build_setcon_video_model(
                qwen_model_name_or_path=qwen_model_name_or_path,
                sam3_checkpoint_path=sam3_checkpoint_path,
                processor_name_or_path=processor_name_or_path,
                tokenizer_name_or_path=tokenizer_name_or_path,
                qwen_model=qwen_model,
                qwen_processor=qwen_processor,
                qwen_tokenizer=qwen_tokenizer,
                bpe_path=bpe_path,
                has_presence_token=has_presence_token,
                apply_temporal_disambiguation=apply_temporal_disambiguation,
                device=device,
                torch_dtype=torch_dtype,
                compile=compile,
            )
            .cuda()
            .eval()
        )


class SetConVideoPredictorMultiGPU(_Sam3VideoPredictorMultiGPU, SetConVideoPredictor):
    pass


def build_setcon_video_predictor(*model_args, gpus_to_use=None, **model_kwargs):
    return SetConVideoPredictorMultiGPU(
        *model_args, gpus_to_use=gpus_to_use, **model_kwargs
    )
