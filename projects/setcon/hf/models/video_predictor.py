from third_parts.sam3.model.sam3_video_predictor import (
    Sam3VideoPredictor as _Sam3VideoPredictor,
    Sam3VideoPredictorMultiGPU as _Sam3VideoPredictorMultiGPU,
)


class SetConInternVLVideoPredictor(_Sam3VideoPredictor):
    def __init__(
        self,
        model_name_or_path,
        sam3_checkpoint_path=None,
        processor_name_or_path=None,
        tokenizer_name_or_path=None,
        model=None,
        processor=None,
        tokenizer=None,
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
        del geo_encoder_use_img_cross_attn, strict_state_dict_loading
        self.async_loading_frames = async_loading_frames
        self.video_loader_type = video_loader_type
        from .video_predictor_builder import build_setcon_video_model

        self.model = (
            build_setcon_video_model(
                model_name_or_path=model_name_or_path,
                sam3_checkpoint_path=sam3_checkpoint_path,
                processor_name_or_path=processor_name_or_path,
                tokenizer_name_or_path=tokenizer_name_or_path,
                model=model,
                processor=processor,
                tokenizer=tokenizer,
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


class SetConInternVLVideoPredictorMultiGPU(
    _Sam3VideoPredictorMultiGPU,
    SetConInternVLVideoPredictor,
):
    pass


def build_setcon_video_predictor(*model_args, gpus_to_use=None, **model_kwargs):
    return SetConInternVLVideoPredictorMultiGPU(
        *model_args,
        gpus_to_use=gpus_to_use,
        **model_kwargs,
    )
