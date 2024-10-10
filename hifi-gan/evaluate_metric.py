from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import argparse
import json
import torch
from scipy.io.wavfile import write
from env import AttrDict
from meldataset import mel_spectrogram, MAX_WAV_VALUE, load_wav
from models import Generator
from pesq import pesq
from pystoi import stoi
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio
from pypesq import pesq as py_pesq
import pandas as pd
from tqdm import tqdm
from metric_utils import mean_std

h = None
device = None


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def get_mel(x):
    return mel_spectrogram(x, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def evaluate(a):
    
    filelist = sorted(glob.glob(os.path.join(a.input_wavs_dir, "*.wav")))[:1000]

    data = {"filename": [], "pesq_nb":[], "estoi": [], "stoi":[]}
    sr = 8000

    for i, wavfile in enumerate(tqdm(filelist)):
        filename = os.path.basename(wavfile).replace("_generated","")
        spk = filename.split("_")[2]
        gt_wav, sr = load_wav(wavfile)
        synthesis_wav, sr = load_wav(os.path.join(a.gt_wavs_dir, spk, filename))
        #gt_wav = gt_wav.squeeze().numpy()
        #synthesis_wav = synthesis_wav.squeeze().numpy()

        min_len = min(len(gt_wav),len(synthesis_wav))
        gt_wav = gt_wav[:min_len]
        synthesis_wav = synthesis_wav[:min_len]
        data["pesq_nb"].append(pesq(sr, gt_wav, synthesis_wav, 'nb'))
        estoi_score = stoi(gt_wav, synthesis_wav, sr,  extended=True)
        stoi_score = stoi(gt_wav, synthesis_wav, sr,  extended=False)
        data["filename"].append(filename)
        data["estoi"].append(estoi_score)
        data["stoi"].append(stoi_score)
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(a.output_dir, "_results.csv"), index=False)
    print("PESQ_nb: {:.2f} ± {:.2f}".format(*mean_std(df["pesq_nb"].to_numpy())))
    print("ESTOI: {:.2f} ± {:.2f}".format(*mean_std(df["estoi"].to_numpy())))
    print("STOI: {:.2f} ± {:.2f}".format(*mean_std(df["stoi"].to_numpy())))
    

def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_wavs_dir', default='generated_files')
    parser.add_argument('--gt_wavs_dir', default='gt files')
    parser.add_argument('--output_dir', default='gt files')

    a = parser.parse_args()

    evaluate(a)


if __name__ == '__main__':
    main()

