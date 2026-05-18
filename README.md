# SetCon: Towards Open-Ended Referring Segmentation via Set-Level Concept Prediction

Official implementation of **SetCon: Towards Open-Ended Referring Segmentation via Set-Level Concept Prediction**.

**Zhixiong Zhang, Yizhuo Li, Shuangrui Ding, Yuhang Zang, Shengyuan Ding, Long Xing, Yibin Wang, Qiaosheng Zhang, Jiaqi Wang**

<p align="center" style="font-size: 5 em; margin-top: 0.5em">
    <a href="https://arxiv.org/abs/2605.xxxxx" style="text-decoration: none; margin: 0 10px;">
    <img src="https://img.shields.io/badge/Paper-arXiv-b31b1b?logo=arxiv&style=for-the-badge">
    </a>
    <a href="https://rookiexiong7/SetCon" style="text-decoration: none; margin: 0 10px;">
    <img src="https://img.shields.io/badge/Demo-GitHub%20Pages-2f2f2f?logo=github&style=for-the-badge">
    </a>
    <a href="https://huggingface.co/OpenIXCLab/SeC-4B" style="text-decoration: none; margin: 0 10px;">
    <img src="https://img.shields.io/badge/Model-HuggingFace-orange?logo=huggingface&style=for-the-badge">
    </a>
    <a href="https://huggingface.co/datasets/OpenIXCLab/SeCVOS" style="text-decoration: none; margin: 0 10px;">
    <img src="https://img.shields.io/badge/Dataset-HuggingFace-orange?logo=huggingface&style=for-the-badge">
    </a>

</p>

## 📜 News
🚀 [2026/5/19] The [Paper](https://arxiv.org/abs/2605.xxxxx) and [Code](https://rookiexiong7/SetCon) are released!

## 🔥 Highlights

- 🔥We introduce **Set-Concept Segmentation (SetCon)**, which reformulates open-ended referring segmentation as explicit set-level concept prediction instead of treating multiple targets as independent special-token outputs.
- 🔥SetCon couples LVLM-side reasoning with set-level mask decoding through a language-grounded concept interface, organized hierarchically into a global set-level concept that defines the target scope and fine-grained sub-concepts that align with target subsets.
- 🔥SetCon supports both image and video referring segmentation with the same concept-driven interface, producing complete mask sets for images and temporally propagated masks for videos, with strong results on gRefCOCO, MUSE, MeViS, and Ref-SeCVOS.


## 🛠️ Usage

### 1. Install environment and dependencies

SetCon is configured for Python `>=3.11,<3.12`. The project dependencies are declared in `pyproject.toml`.

```bash
cd Setcon
uv sync --extra latest
source .venv/bin/activate
```


### 2. Download the Pretrained Checkpoints

Download the SetCon checkpoint from [🤗HuggingFace](https://huggingface.co/OpenIXCLab/SeC-4B) and place it in the following directory : 
```
saved_models
  ├── SetCon-8B
  │   └── config.json
  │   └── generation_config.json
  ...
```

### 3. Inference and Evaluation

If you want to test **SetCon** inference on a single image, please refer to `demo.py`.

The evaluation code can be found in `projects/setcon/evaluation`.


### 4. Training
Download the pretrained [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) and [SAM 3](https://huggingface.co/facebook/sam3), and place them in the ./pretrained directory.

Then, download the original frames and SetCon annotations from [🤗HuggingFace](https://huggingface.co/OpenIXCLab/SeC-4B) and place the SetCon training annotations in the following directory:

```
setcon_training_datasets/
  ├── image/
  │   ├── grefcoco_part0.jsonl
  │   ├── grefcoco_part1.jsonl
  │   ├── muse_part0_fixed_filtered.jsonl
  │   ├── muse_part1_fixed_filtered.jsonl
  │   ├── reasonseg_annotated.jsonl
  │   ├── refcoco.jsonl
  │   ├── refcoco+.jsonl
  │   └── refcocog.jsonl
  └── video/
      ├── mevis_train.jsonl
      ├── ref_davis_train.jsonl
      ├── refer_youtube_vos_train.jsonl
      └── revos_train.jsonl
```

The `image_eval/` directory is used only for evaluation and is not required for training.

Before training, update the image and video frame root variables in `projects/setcon/configs/setcon_qwenvl_8b.py` to match your local dataset paths, including `COCO_IMAGE_ROOT`, `REASONSEG_IMAGE_ROOT`, `MEVIS_FRAME_ROOT`, `REF_DAVIS_FRAME_ROOT`, `REFER_YOUTUBE_FRAME_ROOT`, and `REVOS_FRAME_ROOT`.

Please run the following script to train using 8 GPUS, we suggest using at least 8 A100 GPUs:
```bash
bash tools/dist.sh train projects/setcon/configs/setcon_qwenvl_8b.py 8
```


## ❤️ Acknowledgments and License
This repository are licensed under a [Apache License 2.0](LICENSE). 

This repo benefits from [SAM 3](https://github.com/facebookresearch/sam3)and [Sa2VA](https://github.com/magic-research/Sa2VA). Thanks for their wonderful works.

## ✒️ Citation
If you find our work helpful for your research, please consider giving a star ⭐ and citation 📝
```bibtex

```
