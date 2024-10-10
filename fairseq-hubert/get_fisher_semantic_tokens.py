import torch
from torch import nn
import torch.nn.functional as F
from torchaudio import load
import torchaudio
from scipy.io.wavfile import read, write
import torchaudio
import numpy as np
import argparse
from tqdm import tqdm
import glob
import os
import soundfile as sf
import matplotlib.pyplot as plt
import parser
from examples.textless_nlp.dgslm.dgslm_utils import HubertTokenizer
from tqdm import tqdm



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--process_dir", type=str, default="/data/v-leyzhang/dGSLM/train_only_laughter", help='Directory containing the wav data ')
    parser.add_argument("--target_dir", type=str, default="/data/v-leyzhang/dGSLM/train_only_laughter_semantic_tokens", help='Directory containing the generated semantic tokens')
    parser.add_argument("--hubert_path", type=str, default="/data/v-leyzhang/dGSLM/hubert_fisher.pt", help='Directory containing the hubert ckpt')
    parser.add_argument("--km_path", type=str, default="/data/v-leyzhang/dGSLM/hubert_fisher_km_500.bin", help='Directory containing the kmeans ckpt')
    
    args = parser.parse_args()
    
    encoder = HubertTokenizer(hubert_path = args.hubert_path,
                              hubert_layer = 12,
                              km_path = args.km_path)
    
    process_files = glob.glob(os.path.join(args.process_dir, "*.wav"))
    for process_file in tqdm(process_files):
        codes = encoder.wav2code(process_file,1)
        codes = codes.split(" ")
        hubert_codes = np.array(codes)
        file_name = process_file.split("/")[-1].split(".")[0]
        np.save(os.path.join(args.target_dir,  file_name+'.hubert_code'), hubert_codes)
        
    
