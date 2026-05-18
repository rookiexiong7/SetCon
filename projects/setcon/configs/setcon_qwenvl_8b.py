from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from torch.optim import AdamW
from transformers import AutoTokenizer, Qwen3VLProcessor

from xtuner.dataset.samplers import LengthGroupedSampler
from xtuner.engine.runner import TrainLoop
from xtuner.utils import PROMPT_TEMPLATE

from peft import LoraConfig

from projects.setcon.models import (
    DirectResize,
    SAM3TrainRunner,
    SetConModel,
)
from projects.setcon.models.mllm.qwen3vl import Qwen3VL
from projects.setcon.datasets.data_utils import ConcatDatasetSetCon, setcon_collect_fn
from projects.setcon.datasets.setcon_image_seg import SetConMultiSegDataset
from projects.setcon.datasets.setcon_video_seg import SetConVideoSegDataset

from third_parts.sam3.train.loss.sam3_loss import Sam3LossWrapper
from third_parts.sam3.train.matcher import BinaryHungarianMatcherV2, BinaryOneToManyMatcher
from third_parts.sam3.train.loss.loss_fns import Boxes, Masks, IABCEMdetr

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
COCO_IMAGE_ROOT = 'COCO'
REASONSEG_IMAGE_ROOT = 'ReasonSeg'
MEVIS_FRAME_ROOT = 'MeViSv2/train/JPEGImages'
REF_DAVIS_FRAME_ROOT = 'Ref-DAVIS/Ref-DAVIS/train/JPEGImages'
REFER_YOUTUBE_FRAME_ROOT = 'Refer-YouTube-VOS/train/JPEGImages'
REVOS_FRAME_ROOT = 'ReVOS'

# Model
path = 'pretrained/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'
pretrained_pth = None

# Data
template = "qwen_chat"
prompt_template = PROMPT_TEMPLATE.qwen_chat
max_length = 8192

# Scheduler & Optimizer
batch_size = 2          # per_device debug now
accumulative_counts = 4 # on 16 gpus
dataloader_num_workers = 1
max_epochs = 1
optim_type = AdamW
# official 1024 -> 4e-5
lr = 4e-5
betas = (0.9, 0.999)
weight_decay = 0.05
max_norm = 1  # grad clip
warmup_ratio = 0.05

# Save
save_steps = 3000
save_total_limit = 10  # Maximum checkpoints to keep (-1 means unlimited)

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
#######################################################################
#            PART 2  Model & Tokenizer & Image Processor              #
#######################################################################

loss_sam3=dict(
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
    arch_type='qwen',
    mllm=dict(
        type=Qwen3VL,
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
            modules_to_save=['lm_head', 'embed_tokens'],
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

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################

# JSONL annotations live under this path relative to the launch directory.
DATA_ROOT = 'setcon_training_datasets/'

default_dataset_configs = dict(
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    prompt_template=prompt_template,
    # max_length=max_length,
    arch_type='qwen',
    preprocessor=dict(
        type=Qwen3VLProcessor.from_pretrained,
        pretrained_model_name_or_path=path,
        trust_remote_code=True,
    )
)

###################### MultiImageRefSeg ################################

VIDEO_DATASET_PATH = DATA_ROOT + 'video/'
# Frame paths are resolved as frame_root / video_id / frame_name.
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
        name='MeViS_VideoSeg',
        **default_dataset_configs,
    ),
    dict(
        type=SetConVideoSegDataset,
        json_file=VIDEO_DATASET_PATH + 'refer_youtube_vos_train.jsonl',
        frame_root=REFER_YOUTUBE_FRAME_ROOT,
        num_sample_frames=8,
        max_length=8192,
        repeats=3.0,
        name='MeViS_VideoSeg',
        **default_dataset_configs,
    ),
    dict(
        type=SetConVideoSegDataset,
        json_file=VIDEO_DATASET_PATH + 'revos_train.jsonl',
        frame_root=REVOS_FRAME_ROOT,
        num_sample_frames=8,
        max_length=8192,
        repeats=3.0,
        name='MeViS_VideoSeg',
        **default_dataset_configs,
    ),
]

IMAGE_DATASET_PATH = DATA_ROOT + 'image/'
# Image paths in JSONL are resolved relative to the dataset-specific image_root.
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

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(
        type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale='dynamic',
    dtype='bfloat16'
)

# learning policy
# More information: https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md  # noqa: E501
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

# train, val, test setting
train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
# Log the dialogue periodically during the training process, optional
custom_hooks = [
    # dict(type=DatasetInfoHook, tokenizer=tokenizer),
]

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),
    # print log every 10 iterations.
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    # enable the parameter scheduler.
    param_scheduler=dict(type=ParamSchedulerHook),
    # save checkpoint per `save_steps`.
    checkpoint=dict(
        type=CheckpointHook,
        save_optimizer=False,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit),
    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type=DistSamplerSeedHook),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

# set visualizer
visualizer = None

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)

# set log processor
log_processor = dict(by_epoch=False)
