<br>
<p align="center">
<h1 align="center"><strong>CoVoMix: Advancing Zero-Shot Speech Generation for Human-like Multi-talker Conversations
</strong></h1>
  </p>

<p align="center">
  <a href="https://arxiv.org/abs/2404.06690" target='_**blank**'>
    <img src="https://img.shields.io/badge/arxiv-2404.06690-blue?">
  </a> 
  <a href="https://arxiv.org/pdf/2404.06690" target='_blank'>
    <img src="https://img.shields.io/badge/Paper-📖-blue?">
  </a> 
  <a href="https://www.microsoft.com/en-us/research/project/covomix/" target='_blank'>
    <img src="https://img.shields.io/badge/Project-&#x1F680-blue">
  </a>
  <a href="https://youtu.be/OZPkBXhWT78" target='_blank'>
    <img src="https://img.shields.io/badge/Demo-&#x1f917-blue">
  </a>
  <a href="" target='_blank'>
    <img src="https://visitor-badge.laobi.icu/badge?page_id=OpenRobotLab.pointllm&left_color=gray&right_color=blue">
  </a>
</p>


## 🏠 Introduction

We introduce <b>CoVoMix</b>: Conversational Voice Mixture Generation, a novel model for zero-shot, human-like, multi-speaker, multi-round dialogue speech generation. In addition, we devise a comprehensive set of metrics for measuring the effectiveness of dialogue modeling and generation. Our experimental results show that CoVoMix can generate dialogues that are not only human-like in their naturalness and coherence but also involve multiple speakers engaging in multiple rounds of conversation. These dialogues, generated within a single channel, are characterized by seamless speech transitions, including overlapping speech, and appropriate paralinguistic behaviors such as laughter and coughing.

To avoid abuse, well-trained models and services will not be provided.

Please do not hesitate to tell us if you have any feedback!

## 📋 Contents
- [💬 Specification of dependencies](#specification-of-dependencies)
- [🔍 Data preparation](#data-preparation) 
- [📦 Training](#training)
- [🤖 Inference](#inference)
- [🔗 Citation](#citation)

## 💬 Specification of dependencies

```
conda create -n covomix python=3.8
source activate covomix
pip install voicebox_pytorch==0.0.34 jiwer speechbrain textgrid matplotlib soundfile librosa
cd fairseq 
pip install --editable ./
cd ../
pip install -r requirements.txt
```


## 🔍 Data preparation

Make sure you have already downloaded the Fisher English Dataset. If not, you need to first download it from https://catalog.ldc.upenn.edu/LDC2004T19. 

#### 1. Monologue Dataset Preparation
```
python data_preparation/process_fisher_data.py \
   --audio_root=<audio (.wav) directory>
   --transcript_root=<LDC Fisher dataset directory> \
   --dest_root=<destination directory> \
   --data_sets=LDC2004S13-Part1,LDC2005S13-Part2 \
   --remove_noises \
   --min_slice_duration 10 \
```

#### 2. Dialogue Dataset Preparation

**Dialogue for training Text2semantic model:**   
```
python data_preparation/process_fisher_data_conversation_overlap_text2semantic.py \
   --audio_root=<audio (.wav) directory>
   --transcript_root=<LDC Fisher dataset directory> \
   --dest_root=<destination directory> \
   --data_sets=LDC2004S13-Part1,LDC2005S13-Part2 \
   --remove_noises \
```

**Dialogue for training Acoustic Model:**
```
python data_preparation/process_fisher_data_conversation.py \
   --audio_root=<audio (.wav) directory>
   --transcript_root=<LDC Fisher dataset directory> \
   --dest_root=<destination directory> \
   --data_sets=LDC2004S13-Part1,LDC2005S13-Part2 \
   --remove_noises \
```


#### 3. Feature Extraction

**Extract Mel-spectrogram**
```
python data_preparation/prepare_8k_mel_20ms.py \
    --processed_path <audio (.wav) directory>
    --target_path <destination directory> 
```

**Saving text from json file**
```
bash data_preparation/save_txt.sh $json_file $target_dir
```

**Extract Hubert Semantic Token Sequence**
```
cd fairseq-hubert \
python get_fisher_semantic_tokens.py \
    --process_dir <audio (.wav) directory>
    --target_dir <destination directory>
```


#### 4. Inference data preparation

Before doing inference, you need to 

1. Prepare text directory. The text directory contains .txt files with text that you want to synthesis. If you want to generate dialogue, please use '[spkchange]' to separate each speaker's utterance. 

2. Prepare prompt directory. The prompt directory for monologue generation contains .wav that has the same name as in text directory. The acoustic prompt for dialogue generation needs two  wavefiles, and the prompt directory for dialogue contains .wav files that named "textfilename_1.wav" and "textfilename_2.wav". 

3. Extract hubert semantic tokens for prompts. 
```
python get_fisher_semantic_tokens.py \
    --process_dir CoVoMix/exp/monologue/prompt_dir  
    --target_dir CoVoMix/exp/monologue/prompt_dir 
```
## 📦 Training 


#### 1. Train Text2semantic Model
**Training CoSingle Model:** bash running_command/T2S_CoSingle.sh

**Training CoMix Model:** bash running_command/T2S_CoMix.sh


#### 2. Train Acoustic Model

**Training VoSingle Model:** bash running_command/Acous_VoSingle.sh

**Training VoMix Model:** bash running_command/Acous_VoMix.sh

#### 3. Train HiFiGAN Model

```
cd hifi-gan
python train.py \
    --input_wavs_dir= \
    --input_val_wavs_dir <audio (.wav) directory> \
    --config config_covomix.json \
    --checkpoint_path <destination directory> \
    --stdout_interval 10
```

#### 4. Train speaker verification model


## 🤖 Inference 
Please make sure that the paired text and prompts have identical filename and ended with .txt and .wav respectively. 


#### 1. Monologue Generation

There are 3 modes that you can choose: covosingle, covosinx and covomix. Run this code for inference with CoVoSingle model. If you want to use other mode, please change --mode and the corresponding checkpoints. 

```
python monologue_generation.py \
    --t2s_ckpt CoSingle.ckpt \
    --acous_ckpt VoSingle.ckpt \
    --hifigan_ckpt vocoder.ckpt \
    --text_dir path_to_text_dir \
    --prompt_dir path_to_prompt_dir \
    --saved_dir path_to_covosingle_saved_dir \
    --mode covosingle \
```


#### 2. Dialogue Generation 

There are 3 modes that you can choose: covosingle, covosinx and covomix. Run this example code for inference with CoVoSingle model. If you want to use other mode, please change --mode and the corresponding checkpoints. 

```
python dialogue_generation.py \
    --t2s_ckpt CoMix.ckpt \
    --acous_ckpt VoMix.ckpt \
    --hifigan_ckpt vocoder.ckpt \
    --text_dir path_to_text_dir \
    --prompt_dir path_to_prompt_dir \
    --saved_dir path_to_covomix \
    --mode covomix
```

## Acknowledgement

A significant portion of the training code in this repository is based on voicebox-pytorch by lucidrains (https://github.com/lucidrains/voicebox-pytorch). I would like to express my gratitude to the author for providing this excellent resource, which has been instrumental in the development of this project.


The HuBERT semantic token extraction in this repository is based on the Fairseq repository by Facebook AI Research (FAIR) (https://github.com/facebookresearch/fairseq.git). I would like to thank the FAIR team for their open-source contributions, which have greatly supported the development of this project.



## 🔗 Citation

To cite this repository

```bibtex
@article{zhang2024covomix,
  title={CoVoMix: Advancing Zero-Shot Speech Generation for Human-like Multi-talker Conversations},
  author={Leying Zhang, Yao Qian, Long Zhou, Shujie Liu, Dongmei Wang, Xiaofei Wang, Midia Yousefi, Yanmin Qian, Jinyu Li, Lei He, Sheng Zhao, Michael Zeng},
  journal={Proceedings of the 38th International Conference on Neural Information Processing Systems (NeurIPS 2024)},
  year={2024}
}
```