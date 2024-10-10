import os

os.environ["OMP_NUM_THREADS"] = "1"

import glob
import json
import logging
import sys
import traceback
from multiprocessing.pool import Pool
import numpy as np
import pandas as pd
from tqdm import tqdm
import textgrid
import json
import warnings
import copy
from tqdm import tqdm

import torch
import torch
import struct
import pyworld as pw
from scipy.io.wavfile import read, write
import parser
import argparse

import librosa
import numpy as np

from data_preparation.generate_mel import mel_spectrogram
import re
import subprocess
from multiprocessing.pool import Pool



gamma = 0
mcepInput = 3  # 0 for dB, 3 for magnitude
alpha = 0.45
en_floor = 10 ** (-80 / 20)
FFT_SIZE = 2048

f0_bin = 256
f0_max = 1100.0
f0_min = 50.0


def f0_to_coarse(f0):
    f0_mel = 1127 * np.log(1 + f0 / 700)
    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)
    # f0_mel[f0_mel == 0] = 0
    # 大于0的分为255个箱
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * (f0_bin - 2) / (f0_mel_max - f0_mel_min) + 1

    f0_mel[f0_mel < 0] = 1
    f0_mel[f0_mel > f0_bin - 1] = f0_bin - 1
    f0_coarse = np.rint(f0_mel)
    f0_coarse = f0_coarse.astype(int)
    # print('Max f0', np.max(f0_coarse), ' ||Min f0', np.min(f0_coarse))
    assert (np.max(f0_coarse) <= 256 and np.min(f0_coarse) >= 0)
    return f0_coarse




def get_pitch(wav_data, mel, sample_rate,hop_size):
    """

    :param wav_data: [T]
    :param mel: [T, 80]
    :param hparams:
    :return:
    """
    _f0, t = pw.dio(wav_data.astype(np.double), sample_rate,
                    frame_period=hop_size / sample_rate * 1000)
    f0 = pw.stonemask(wav_data.astype(np.double), _f0, t, sample_rate)  # pitch refinement
    #print(len(mel),mel.shape, len(f0), f0.shape)
    delta_l = len(mel) - len(f0)
    assert np.abs(delta_l) <= 2
    if delta_l > 0:
        f0 = np.concatenate([f0] + [f0[-1]] * delta_l)
    f0 = f0[:len(mel)]
    pitch_coarse = f0_to_coarse(f0) + 1
    return f0, pitch_coarse




def process_utterance(wav_path,
                      fft_size=1024,
                      hop_size=256,
                      win_length=1024,
                      num_mels=80,
                      fmin=0,
                      fmax=8000,
                      sample_rate=22050):
    
    MAX_WAV_VALUE = 32768.0

    try:
        wav, sr = librosa.load(wav_path, sr=sample_rate)
    except:
        print("error file",wav_path)
        
    wav = np.clip(wav, -1, 1)
    x = torch.FloatTensor(wav)
    
    mel = mel_spectrogram(x.unsqueeze(0), n_fft=fft_size, num_mels=num_mels, sampling_rate=sample_rate,
                        hop_size=hop_size, win_size=win_length, fmin=fmin, fmax=fmax)
    mel = mel.cpu().numpy()[0]
    wav = wav * MAX_WAV_VALUE
    wav = wav.astype('int16')
        
    return wav, mel


def data_preprocessing_item(tg_fn,n_fft,hop_size,win_size,fmin,fmax,sample_rate):
    #print("begin executing data_preprocessing_item")
    wav_fn = tg_fn
    wav, mel = process_utterance(wav_fn,
                    fft_size=n_fft,
                    hop_size=hop_size,
                    win_length=win_size,
                    num_mels=80,
                    fmin=fmin,
                    fmax=fmax,
                    sample_rate = sample_rate)
      
    return mel

# def call_back(res):
#     print(f'Hello,World! {res}')

def err_call_back(err):
    print(f'error: {str(err)}')

if __name__ == "__main__":
    # input parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed_path', type=str, required = True, help='path to the processed data')
    parser.add_argument('--target_path', type=str, required = True, help='path to save the mel-spectrograms')
    args = parser.parse_args()
    
    p = Pool(processes=35)
    sample_rate = 8000
    hop_size = 160
    win_size =  480 
    fmin= 0
    fmax = 4000
    n_fft= 480

    
    processed_path = args.processed_path
    all_textgrid_fns = sorted(glob.glob(f'{processed_path}/*.wav'))
    for tg_fn in tqdm(all_textgrid_fns):
        save_dir = args.target_path
        #print("save_dir",save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)   
        file_name = os.path.basename(tg_fn).split(".")[0]
        
        future = p.apply_async(data_preprocessing_item, args=(tg_fn,n_fft,hop_size,win_size,fmin,fmax,sample_rate,),error_callback=err_call_back)
        mel = future.get()

        np.save(os.path.join(save_dir,  file_name+'.mel'), mel)
        
    p.close()