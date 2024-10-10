import os

# Get the current PATH
original_path = os.environ.get('PATH')

# Add CUDA path to the PATH
cuda_path = '/usr/local/cuda/bin'
new_path = cuda_path + ':' + original_path

# Set the new PATH
os.environ['PATH'] = new_path

import glob
from argparse import ArgumentParser
from os.path import join
import torch
from soundfile import write
from torchaudio import load
import torchaudio
from tqdm import tqdm
import torch.nn.functional as F
import time
import json
from transformers import BertTokenizer, BertModel
import librosa

from scipy.io.wavfile import write
from covomix.util.other import ensure_dir, pad_spec
import random
import wespeakerruntime as wespeaker
import numpy as np
from covomix.util.other import energy_ratios, mean_std
from covomix.conditional_model import CoVoMixModel
from covomix.vocoder.models import Generator
from covomix.vocoder.env import AttrDict
from covomix.covomix_model.text2semantic import TextToSemantic
from data_preparation.generate_mel import mel_spectrogram
from torch.nn.utils.rnn import pad_sequence

MAX_WAV_VALUE = 32768.0

# Initialization of Hifigan 8k
## Define basic functions

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

def mel_decode_to_wav(generator, mel): 
    with torch.no_grad():
        y_g_hat = generator(mel)
        audio = y_g_hat.squeeze()
        audio = audio * MAX_WAV_VALUE
        audio = audio.cpu().numpy().astype('int16')
        
    return audio


# Extract Mel-spectrogram

sample_rate = 8000
hop_size = 160
win_size =  480 
fmin= 0
fmax = 4000
n_fft= 480

def extract_mel(x_path,sample_rate=8000, hop_size=160, win_size=480, fmin=0, fmax=4000, n_fft=480, num_mels=80, channel_idx = None):
    x_path = x_path.replace("_hubert_code.npy",".wav").replace(".hubert_code.npy",".wav")
    if os.path.exists(x_path.replace(".wav",".mel.npy")):
        mel = np.load(x_path.replace(".wav",".mel.npy"))
        return mel
    
    if channel_idx == None:
        wav, sr = librosa.load(x_path, sr=sample_rate)
    else: 
        wav, sr = librosa.load(x_path, sr=sample_rate, mono=False)
        wav = wav[channel_idx]
    wav = np.clip(wav, -1, 1)
    x = torch.FloatTensor(wav)
    mel = mel_spectrogram(x.unsqueeze(0), n_fft=n_fft, num_mels=num_mels, sampling_rate=sample_rate,
                            hop_size=hop_size, win_size=win_size, fmin=fmin, fmax=fmax)
    mel = mel[0].cpu()
    return mel


# Process hubert code and mel to make it equal len
def equal_len(hubert_code, mel):
    equal_len = min(hubert_code.shape[0], mel.shape[1])
    hubert_code = hubert_code[:equal_len]
    mel = mel[:,:equal_len]
    return hubert_code, mel


def create_fix_mask(seq_len, mask_ratio):
    
    # Calculate the number of elements to mask
    num_elements_to_mask = int(mask_ratio * seq_len)
    
    # Generate a random start index for the mask
    start_index = np.random.randint(0, seq_len - num_elements_to_mask + 1)
    
    # Create a mask where the selected continuous elements are True
    mask = torch.zeros(seq_len)
    mask[-num_elements_to_mask:] = 1
    
    return mask

def create_random_mask(seq_len, mask_ratio):
    
    # Calculate the number of elements to mask
    num_elements_to_mask = int(mask_ratio * seq_len)
    
    # Generate a random start index for the mask
    start_index = np.random.randint(0, seq_len - num_elements_to_mask + 1)
    
    # Create a mask where the selected continuous elements are True
    mask = torch.zeros(seq_len)
    mask[start_index:start_index + num_elements_to_mask] = 1
    
    return mask

def prepare_oracle_data_for_training(mel_files, i, shuffle_spec=False, random_mask = False):
    try:
        mel = np.load(mel_files[i])
        phoneme = np.load(mel_files[i].replace(".mel.npy",".hubert_code.npy"))
        phoneme = phoneme.astype(int)
    except:
        print("Bad files",mel_files[i] )
        mel = np.load(mel_files[0])
        phoneme = np.load(mel_files[0].replace(".mel.npy",".hubert_code.npy"))
        phoneme = phoneme.astype(int)
    
    equal_len = min(phoneme.shape[0], mel.shape[1])
    mel = mel[:,:equal_len]
    phoneme = phoneme[:equal_len]
    
    mel = torch.tensor(mel).permute(1,0)
    phoneme = torch.LongTensor(phoneme)
    
    # # formula applies for center=True
    max_len = torch.randint(low=300, high=500, size=(1,)).item()
    current_len = mel.shape[0]
    if current_len > max_len:
        if shuffle_spec:
            start = int(np.random.uniform(0, current_len-max_len))
        else:
            start = int((current_len-max_len)/2)
        mel = mel[start:start+max_len,:]
        phoneme = phoneme[start:start+max_len]
    
    
    frac_lengths = np.random.uniform(0.7,1.0)
    if random_mask:
        mask = create_random_mask(seq_len = phoneme.shape[0], mask_ratio=frac_lengths)
    else: 
        mask = create_fix_mask(seq_len = phoneme.shape[0], mask_ratio=frac_lengths)
    
    return mel, phoneme, mask




def prepare_oracle_data_for_training_from_specific_file(mel_file, shuffle_spec=False, fix_start_point = None, frac_lengths = 0.8,mix_1channel_mel=False, random_mask=False):
    if mix_1channel_mel:
        mel = np.load(mel_file)
        equal_len = mel.shape[1]
        phoneme = np.zeros(equal_len)
    else: 
        try:
            mel = np.load(mel_file)
            phoneme = np.load(mel_file.replace(".mel.npy","-16k.hubert_code.npy"))
            phoneme = phoneme.astype(int)
            equal_len = min(phoneme.shape[0], mel.shape[1])
        except:
            # The mixed channel wav does not have phonem
            
            print("file not exist",mel_file, mel_file.replace(".mel.npy","-16k.hubert_code.npy") )
    
    
    mel = mel[:,:equal_len]
    phoneme = phoneme[:equal_len]
    
    mel = torch.tensor(mel).permute(1,0)
    phoneme = torch.LongTensor(phoneme)
    
    # # formula applies for center=True
    max_len = 1000
    current_len = mel.shape[0]
    if current_len > max_len:
        if shuffle_spec:
            if fix_start_point == None:
                start = int(np.random.uniform(0, current_len-max_len))
                fix_start_point = start
            else: 
                start = fix_start_point
        else:
            start = int((current_len-max_len)/2)
        mel = mel[start:start+max_len,:]
        phoneme = phoneme[start:start+max_len]
    
    if frac_lengths == None:
        frac_lengths = np.random.uniform(0.7,1.0)
    if random_mask:
        mask = create_random_mask(seq_len = phoneme.shape[0], mask_ratio=frac_lengths)
    else: 
        mask = create_fix_mask(seq_len = phoneme.shape[0], mask_ratio=frac_lengths)
    
    return mel, phoneme, mask, fix_start_point




def prepare_oracle_data_for_training_with_prompt(mel_files, i, shuffle_spec=False,random_mask=False):
    try:
        mel = np.load(mel_files[i])
        phoneme = np.load(mel_files[i].replace(".mel.npy",".hubert_code.npy"))
        phoneme = phoneme.astype(int)
    except:
        print("Bad files",mel_files[i] )
        mel = np.load(mel_files[0])
        phoneme = np.load(mel_files[0].replace(".mel.npy",".hubert_code.npy"))
        phoneme = phoneme.astype(int)
    
    equal_len = min(phoneme.shape[0], mel.shape[1])
    mel = mel[:,:equal_len]
    phoneme = phoneme[:equal_len]
    
    mel = torch.tensor(mel).permute(1,0)
    phoneme = torch.LongTensor(phoneme)
    
    # # formula applies for center=True
    max_len = torch.randint(low=300, high=700, size=(1,)).item()
    current_len = mel.shape[0]
    if current_len > max_len:
        if shuffle_spec:
            start = int(np.random.uniform(0, current_len-max_len))
        else:
            start = int((current_len-max_len)/2)
        mel = mel[start:start+max_len,:]
        phoneme = phoneme[start:start+max_len]
    
    
    # frac_lengths = np.random.uniform(0.7,1.0)
    # mask = create_fix_mask(seq_len = phoneme.shape[0], mask_ratio=frac_lengths)
    j = choose_prompt(mel_files, i)
    prompt_mel = np.load(mel_files[j])
    prompt_phoneme = np.load(mel_files[j].replace(".mel.npy",".hubert_code.npy"))
    prompt_phoneme = prompt_phoneme.astype(int)
    prompt_equal_len = min(prompt_phoneme.shape[0], prompt_mel.shape[1])
    prompt_mel = prompt_mel[:,:prompt_equal_len]
    prompt_phoneme = prompt_phoneme[:prompt_equal_len]
    
    prompt_mel = torch.tensor(prompt_mel).permute(1,0)
    prompt_phoneme = torch.LongTensor(prompt_phoneme)
    
    # # formula applies for center=True
    max_len = torch.randint(low=100, high=200, size=(1,)).item()
    current_len = prompt_mel.shape[0]
    if current_len > max_len:
        if shuffle_spec:
            start = int(np.random.uniform(0, current_len-max_len))
        else:
            start = int((current_len-max_len)/2)
        prompt_mel = prompt_mel[start:start+max_len,:]
        prompt_phoneme = prompt_phoneme[start:start+max_len]
    
    mel = torch.cat((prompt_mel, mel), dim=0)
    phoneme = torch.cat((prompt_phoneme, phoneme), dim=0)
    mask = torch.ones(phoneme.shape[0])
    mask[:prompt_phoneme.shape[0]] = 0
    mask = mask.bool()
    
    return mel, phoneme, mask

def choose_prompt(mel_files, i, long_hubert_code_list=None):
    # return the index of the prompt    

    filename = mel_files[i]
    j = random.randint(max(i-30,0), min(i+30,len(mel_files)-1))
    index = 0
    while mel_files[i].split("-")[0] != mel_files[j].split("-")[0] and index < 10:
        j = random.randint(max(i-30,0), min(i+30,len(mel_files)-1)) 
    
    if index >= 10 and mel_files[i].split("-")[0] != mel_files[j].split("-")[0] and long_hubert_code_list != None:
        return choose_prompt(long_hubert_code_list, i)
    else: 
        return j


def choose_different_spk(mel_files,  i, long_hubert_code_list=None):
    # return the index of the utterance with different speaker

    filename = mel_files[i]
    j = random.randint(max(i-150,0), min(i+150,len(mel_files)-1))
    index = 0
    while mel_files[i].split("-")[0] == mel_files[j].split("-")[0] and index < 10:
        j = random.randint(max(i-500,0), min(i+500,len(mel_files)-1)) 
        
    return j


def choose_prompt_backchannel_or_long(mel_files, i, long_hubert_code_list=None):
    # return the index of the prompt    
    if long_hubert_code_list == None:
        filename = mel_files[i]
    else:
        filename = long_hubert_code_list[i]
    j = random.randint(max(i-30,0), min(i+30,len(mel_files)-1))
    index = 0
    while mel_files[i].split("-")[0] != mel_files[j].split("-")[0] and index < 10:
        j = random.randint(max(i-30,0), min(i+30,len(mel_files)-1)) 
    
    if index >= 10 and mel_files[i].split("-")[0] != mel_files[j].split("-")[0] and long_hubert_code_list != None:
        return choose_prompt(long_hubert_code_list, i)
    else: 
        return j, mel_files




def prepare_text2semantic_multispk_data(long_hubert_code_list, backchannel_list, i, num_spk = 2, num_round = 3):
    # This code only works for 2 spks (num_spk == 2)
    text_list = []
    hubert_code_list1 = []
    hubert_code_list2 = []  
    
    for index in range(num_spk):
        if index == 0:
            j = i # first spk
            first_spk_utt = i
        else: 
            j = choose_different_spk(long_hubert_code_list, i) #second spk
            second_spk_utt = j

        try:
            hubert_code = np.load(long_hubert_code_list[j]) #mel here is the hubert_code
            hubert_code = torch.tensor(hubert_code.astype(int))
            with open(long_hubert_code_list[j].replace("-16k.hubert_code.npy",".txt").replace(".hubert_code.npy",".txt"), 'r') as file: 
                text = file.read()
        except:
            print("Bad files",long_hubert_code_list[j])
            hubert_code = np.load(long_hubert_code_list[0])
            hubert_code = torch.tensor(hubert_code.astype(int))
            with open(long_hubert_code_list[0].replace("-16k.hubert_code.npy",".txt").replace(".hubert_code.npy",".txt"), 'r') as file:
                text = file.read()
        silence_code_for_another_spk = (torch.ones_like(hubert_code) * 157).long()
    
        text_list.append(text)
        if index % num_spk == 0:
            hubert_code_list1.append(hubert_code)
            hubert_code_list2.append(silence_code_for_another_spk)
        else:
            hubert_code_list1.append(silence_code_for_another_spk)
            hubert_code_list2.append(hubert_code)
            
    phoneme = ["[spkchange]".join(text_list)]    
    hubert_code_output_1 = torch.cat(hubert_code, dim = -1)
    hubert_code_output = torch.cat((hubert_code,hubert_code),dim=1) 
    
    return hubert_code_output, phoneme, mask