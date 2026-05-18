from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from torch.optim import AdamW
from transformers import AutoTokenizer

from xtuner.dataset.samplers import LengthGroupedSampler
from xtuner.engine.runner import TrainLoop
from xtuner.utils import PROMPT_TEMPLATE

from peft import LoraConfig

from projects.setcon.models import (
    DirectResize,
    SAM3TrainRunner,
    SetConModel,
)
from projects.setcon.models.mllm.internvl import InternVLMLLM
from projects.setcon.datasets.data_utils import ConcatDatasetSetCon, setcon_collect_fn
from projects.setcon.datasets.setcon_image_seg import SetConMultiSegDataset
from projects.setcon.datasets.setcon_video_seg import SetConVideoSegDataset

from third_parts.sam3.train.loss.sam3_loss import Sam3LossWrapper
from third_parts.sam3.train.matcher import BinaryHungarianMatcherV2, BinaryOneToManyMatcher
from third_parts.sam3.train.loss.loss_fns import Boxes, Masks, IABCEMdetr

COCO_IMAGE_ROOT = 'COCO'
REASONSEG_IMAGE_ROOT = 'ReasonSeg'
MEVIS_FRAME_ROOT = 'MeViSv2/train/JPEGImages'
REF_DAVIS_FRAME_ROOT = 'Ref-DAVIS/Ref-DAVIS/train/JPEGImages'
REFER_YOUTUBE_FRAME_ROOT = 'Refer-YouTube-VOS/train/JPEGImages'
REVOS_FRAME_ROOT = 'ReVOS'

path = 'pretrained/models--OpenGVLab--InternVL3_5-8B/snapshots/9bb6a56ad9cc69db95e2d4eeb15a52bbcac4ef79'
pretrained_pth = None

template = "internlm2_chat"
prompt_template = PROMPT_TEMPLATE.internlm2_chat
max_length = 8192

batch_size = 2
accumulative_counts = 2
dataloader_num_workers = 1
max_epochs = 1
optim_type = AdamW
lr = 4e-5
betas = (0.9, 0.999)
weight_decay = 0.05
max_norm = 1
warmup_ratio = 0.05

save_steps = 3000
save_total_limit = 10

special_tokens = ['<ref>', '</ref>']

tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=path,
    trust_remote_code=True,
    padding_side='right')

extra_image_processor = dict(
    type=DirectResize,
    target_length=1024,
)

loss_sam3 = dict(
    type=Sam3LossWrapper,
    normalization="local",
    matcher=dict(
        type=BinaryHungarianMatcherV2,
        focal=True,
        cost_class=2.0,
        cost_bbox=5.0,
        cost_giou=2.0,
        alpha=0.25,
        gamma=2,
        stable=False,
    ),
    o2m_weight=2.0,
    o2m_matcher=dict(
        type=BinaryOneToManyMatcher,
        alpha=0.3,
        threshold=0.4,
        topk=4,
    ),
    use_o2m_matcher_on_o2m_aux=False,
    loss_fns_find=[
        dict(
            type=Boxes,
            weight_dict=dict(
                loss_bbox=5.0,
                loss_giou=2.0,
            ),
        ),
        dict(
            type=Masks,
            focal_alpha=0.25,
            focal_gamma=2,
            weight_dict=dict(
                loss_mask=200.0,
                loss_dice=10.0,
            ),
            compute_aux=False,
        ),
        dict(
            type=IABCEMdetr,
            weak_loss=False,
            weight_dict=dict(
                loss_ce=20.0,
                presence_loss=20.0,
            ),
            pos_weight=5.0,
            alpha=0.25,
            gamma=2,
            use_presence=True,
            pos_focal=False,
            pad_n_queries=200,
            pad_scale_pos=1.0,
        ),
    ],
)

model = dict(
    type=SetConModel,
    training_bs=batch_size,
    special_tokens=special_tokens,
    pretrained_pth=pretrained_pth,
    loss_sample_points=True,
    frozen_sam2_decoder=False,
    arch_type='intern_vl',
    mllm=dict(
        type=InternVLMLLM,
        model_path=path,
        freeze_llm=True,
        freeze_visual_encoder=True,
        llm_lora=dict(
            type=LoraConfig,
            r=128,
            lora_alpha=256,
            lora_dropout=0.05,
            bias='none',
            task_type='CAUSAL_LM',
            target_modules=None,
        ),
    ),
    tokenizer=tokenizer,
    grounding_encoder=dict(
        type=SAM3TrainRunner,
        loss_fn=loss_sam3,
    ),
    decoder_warmup_steps=1000000,
)

DATA_ROOT = 'setcon_training_datasets/'

default_dataset_configs = dict(
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    prompt_template=prompt_template,
    arch_type='intern_vl',
    min_dynamic_patch=1,
    max_dynamic_patch=12,
    image_size=448,
    use_thumbnail=True,
    downsample_ratio=0.5,
    patch_size=14,
)

VIDEO_DATASET_PATH = DATA_ROOT + 'video/'
train_video_configs = [
    dict(
        type=SetConVideoSegDataset,
        json_file=VIDEO_DATASET_PATH + 'mevis_train.jsonl',
        frame_root=MEVIS_FRAME_ROOT,
        num_sample_frames=8,
        max_length=8192,
        repeats=3.0,
        name='MeViS_VideoSeg',
        **default_dataset_configs,
    ),
    dict(
        type=SetConVideoSegDataset,
        json_file=VIDEO_DATASET_PATH + 'ref_davis_train.jsonl',
        frame_root=REF_DAVIS_FRAME_ROOT,
        num_sample_frames=8,
        max_length=8192,
        repeats=3.0,
        name='RefDAVIS_VideoSeg',
        **default_dataset_configs,
    ),
    dict(
        type=SetConVideoSegDataset,
        json_file=VIDEO_DATASET_PATH + 'refer_youtube_vos_train.jsonl',
        frame_root=REFER_YOUTUBE_FRAME_ROOT,
        num_sample_frames=8,
        max_length=8192,
        repeats=3.0,
        name='ReferYouTubeVOS_VideoSeg',
        **default_dataset_configs,
    ),
    dict(
        type=SetConVideoSegDataset,
        json_file=VIDEO_DATASET_PATH + 'revos_train.jsonl',
        frame_root=REVOS_FRAME_ROOT,
        num_sample_frames=8,
        max_length=8192,
        repeats=3.0,
        name='ReVOS_VideoSeg',
        **default_dataset_configs,
    ),
]

IMAGE_DATASET_PATH = DATA_ROOT + 'image/'
train_image_configs = [
    dict(
        type=SetConMultiSegDataset,
        name='MultiSegDataset',
        data_root=IMAGE_DATASET_PATH,
        ann_file='muse_part0_fixed_filtered.jsonl',
        image_root=COCO_IMAGE_ROOT,
        max_length=max_length,
        repeats=1,
        **default_dataset_configs,
    ),
    dict(
        type=SetConMultiSegDataset,
        name='MultiSegDataset',
        data_root=IMAGE_DATASET_PATH,
        ann_file='muse_part1_fixed_filtered.jsonl',
        image_root=COCO_IMAGE_ROOT,
        max_length=max_length,
        repeats=1,
        **default_dataset_configs,
    ),
    dict(
        type=SetConMultiSegDataset,
        name='MultiSegDataset',
        data_root=IMAGE_DATASET_PATH,
        ann_file='refcoco+.jsonl',
        image_root=COCO_IMAGE_ROOT,
        max_length=max_length,
        repeats=1,
        **default_dataset_configs,
    ),
    dict(
        type=SetConMultiSegDataset,
        name='MultiSegDataset',
        data_root=IMAGE_DATASET_PATH,
        ann_file='refcocog.jsonl',
        image_root=COCO_IMAGE_ROOT,
        max_length=max_length,
        repeats=1,
        **default_dataset_configs,
    ),
    dict(
        type=SetConMultiSegDataset,
        name='MultiSegDataset',
        data_root=IMAGE_DATASET_PATH,
        ann_file='refcoco.jsonl',
        image_root=COCO_IMAGE_ROOT,
        max_length=max_length,
        repeats=1,
        **default_dataset_configs,
    ),
    dict(
        type=SetConMultiSegDataset,
        name='MultiSegDataset',
        data_root=IMAGE_DATASET_PATH,
        ann_file='reasonseg_annotated.jsonl',
        image_root=REASONSEG_IMAGE_ROOT,
        max_length=max_length,
        repeats=10,
        **default_dataset_configs,
    ),
    dict(
        type=SetConMultiSegDataset,
        name='MultiSegDataset',
        data_root=IMAGE_DATASET_PATH,
        ann_file='grefcoco_part0.jsonl',
        image_root=COCO_IMAGE_ROOT,
        max_length=max_length,
        repeats=1,
        **default_dataset_configs,
    ),
    dict(
        type=SetConMultiSegDataset,
        name='MultiSegDataset',
        data_root=IMAGE_DATASET_PATH,
        ann_file='grefcoco_part1.jsonl',
        image_root=COCO_IMAGE_ROOT,
        max_length=max_length,
        repeats=1,
        **default_dataset_configs,
    ),
]

train_dataset = dict(
    type=ConcatDatasetSetCon,
    datasets=[
        *train_video_configs,
        *train_image_configs,
    ],
)
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=train_dataset,
    sampler=dict(
        type=LengthGroupedSampler,
        length_property='modality_length',
        per_device_batch_size=batch_size * accumulative_counts),
    collate_fn=dict(type=setcon_collect_fn),
)

optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(
        type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale='dynamic',
    dtype='bfloat16'
)

param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,
        convert_to_iter_based=True),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        end=max_epochs,
        convert_to_iter_based=True)
]

train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)
custom_hooks = []
default_hooks = dict(
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=dict(
        type=CheckpointHook,
        save_optimizer=False,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit),
    sampler_seed=dict(type=DistSamplerSeedHook),
)
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
visualizer = None
log_level = 'INFO'
load_from = None
resume = False
randomness = dict(seed=None, deterministic=False)
log_processor = dict(by_epoch=False)
