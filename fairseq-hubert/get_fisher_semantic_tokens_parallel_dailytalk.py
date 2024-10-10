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
import parser
from examples.textless_nlp.dgslm.dgslm_utils import HubertTokenizer
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

def process_one_file(process_file):
    val = ['258', '1325', '1104', '1983', '403', '634', '359', '2033', '289', '468', '2171', '1046', '23', '590', '633', '1172', '1902', '2170', '2169', '1418', '877', '1023', '1987', '1417', '684', '71', '760', '1887', '2436', '726', '155', '478', '643', '2081', '819', '1455', '1545', '111', '2031', '2384', '2109', '454', '1238', '162', '1786', '2371', '2387', '1618', '221', '608', '1126', '1936', '1250', '2406', '1917', '898', '1133', '373', '1857', '364', '1020', '696', '749', '881', '280', '1112', '143', '1898', '312', '2279', '1431', '671', '1399', '1908', '1993', '1103', '1414', '59', '2216', '126', '168', '2306', '1091', '578', '1366', '1452', '1829', '2377', '1704', '1800', '1805', '1928', '2282', '30', '79', '1581', '1970', '1949', '371', '1298', '308', '969', '1251', '343', '64', '588', '164', '2522', '1229', '784', '1860', '489', '264', '185', '2075', '443', '1543', '1705', '1211', '188', '2060', '422', '2463', '195', '321', '2126', '1793', '507']
    dialogue = process_file.split("/")[-2]
    file_name = process_file.split("/")[-1].split(".")[0]
    if dialogue not in val:
        return
    else: 
        try:
            codes = encoder.wav2code(process_file, 1)
            codes = codes.split(" ")
            hubert_codes = np.array(codes)
            np.save(os.path.join(args.target_dir, dialogue,file_name + '.hubert_code'), hubert_codes)
        except:
            print("error in process_file", process_file)
            return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--process_dir", type=str, default="/data/v-leyzhang/dGSLM/train_only_laughter", help='Directory containing the wav data ')
    parser.add_argument("--target_dir", type=str, default="/data/v-leyzhang/dGSLM/train_only_laughter_semantic_tokens", help='Directory containing the generated semantic tokens')
    parser.add_argument("--hubert_path", type=str, default="/data/v-leyzhang/dGSLM/hubert_fisher.pt", help='Directory containing the hubert ckpt')
    parser.add_argument("--km_path", type=str, default="/data/v-leyzhang/dGSLM/hubert_fisher_km_500.bin", help='Directory containing the kmeans ckpt')
    global args
    args = parser.parse_args()
    
   
    global encoder
    encoder = HubertTokenizer(hubert_path = args.hubert_path,
                              hubert_layer = 12,
                              km_path = args.km_path)
    encoder
    global process_files
    process_files = glob.glob(os.path.join(args.process_dir, "*/*.wav"))
    
    print("len(process_files)", len(process_files))
    
    # Use a ThreadPoolExecutor to process files in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(process_one_file, process_files), total=len(process_files)))
    
    
    # for process_file in tqdm(process_files):
    #     codes = encoder.wav2code(process_file,1)
    #     codes = codes.split(" ")
    #     hubert_codes = np.array(codes)
    #     file_name = process_file.split("/")[-1].split(".")[0]
    #     np.save(os.path.join(args.target_dir,  file_name+'.hubert_code'), hubert_codes)
        
    
