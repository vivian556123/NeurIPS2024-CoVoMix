import os

# Get the current PATH
original_path = os.environ.get('PATH')

# Add CUDA path to the PATH
cuda_path = '/usr/local/cuda/bin'
new_path = cuda_path + ':' + original_path

# Set the new PATH
os.environ['PATH'] = new_path

import matplotlib.pyplot as plt
import math
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
from transformers import BertTokenizer
import librosa
from tqdm import tqdm

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


# Utils 

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


def extract_mel(x_path, channel_idx = None):
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

def equal_len(hubert_code, mel):
    equal_len = min(hubert_code.shape[0], mel.shape[1])
    hubert_code = hubert_code[:equal_len]
    mel = mel[:,:equal_len]
    return hubert_code, mel

def prepare_oracle_hubert(prompt):
    phoneme_context = np.load(prompt).astype(int)
    phoneme_context = torch.LongTensor(phoneme_context)
    mel_context = extract_mel(prompt.replace(".hubert_code.npy",".wav")) # [80,T]
    phoneme_context, mel_context = equal_len(phoneme_context, mel_context)
    if len(phoneme_context) > 400: # The maximum prompt length is 8s
        phoneme_context = phoneme_context[:400]
        mel_context = mel_context[:,:400]
    return phoneme_context, mel_context.permute(1,0)

def load_text2semantic_model(ckpt):
    text2semantic = CoVoMixModel.load_from_checkpoint(ckpt, base_dir='', batch_size=16, num_workers=0)
    
    text2semantic.eval()
    text2semantic = text2semantic.to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_tokens(['[laughter]'])
    tokenizer.add_tokens(['[spkchange]'])
    tokenizer.add_tokens(['[spka]'])
    tokenizer.add_tokens(['[spkb]'])
    tokenizer.add_tokens(['[partialoverlap]'])
    tokenizer.add_tokens(['[backchannel]'])
    return text2semantic, tokenizer



def remove_punctuation(text):
    punctuation = '''!()-{};:'"\,<>./?@#$%^&*_~'''
    text = text.lower()
    for x in text:
        if x in punctuation:
            text = text.replace(x, "")
    return text


        
def repeat_and_trim_tensor(tensor, T2):
    T1, D = tensor.shape
    # Calculate the total repeat factor (how many times to repeat the entire T1 dimension)
    repeat_factor = -(-T2 // T1)  # Ceiling division
    # Repeat the tensor
    extended_tensor = tensor.repeat(repeat_factor, 1)
    # Trim the tensor to the desired T2 length
    trimmed_tensor = extended_tensor[:T2, :]

    return trimmed_tensor



def remove_unwanted_backchannels(sequence):
    parts = sequence.split()  # Splits the sequence into parts. Adjust the split method if your delimiter is not a space.
    result = []

    for i in range(len(parts)):
        if parts[i] == '[backchannel]' or parts[i] == '[partialoverlap]':
            # Check if the previous part is not '[spkchange]'
            if i == 0 or parts[i-1] != '[spkchange]':
                continue  # Skip this backchannel

        result.append(parts[i])

    return ' '.join(result)  # Joins the parts back into a single string. Adjust if your delimiter is not a space.
    

def covosingle(model, text2semantic, tokenizer, saved_dir, text_dir, prompt_dir):
    # Read evaluation pairs
    text_list = glob.glob(os.path.join(text_dir, "*.txt"))
    
    for text_file in tqdm(text_list):
        prompt = os.path.join(prompt_dir,os.path.basename(text_file.replace(".txt",".hubert_code.npy")))
        semantic_prompt, mel_prompt = prepare_oracle_hubert(prompt) 
                       
        with open (text_file, "r", encoding='utf-8') as f:
            phone_txt = f.read()
        
        if phone_txt != "" or phone_txt != "\n" or phone_txt != "\t" or phone_txt != " ":
            phone_txt = remove_punctuation(phone_txt)
            phone_txt = phone_txt.lower()
            phone_input = cosingle_pred(phone_txt, tokenizer, text2semantic)
            phone_input = torch.cat((semantic_prompt, phone_input))
            phone_input = torch.clamp(phone_input, max=501)
            mel_input = torch.zeros((phone_input.shape[0], 80))
            mel_input[:len(mel_prompt),:] = mel_prompt
            mask = torch.zeros(phone_input.shape[0]).bool()
            mask[len(mel_prompt):] = True
            print("phone_input",phone_input.shape, "mel_input", mel_input.shape, "mask", mask.shape)
            sampled_mel_total = model.synthesis_sample(phoneme_ids = phone_input.unsqueeze(dim=0).to(device), 
                                               cond = mel_input.unsqueeze(dim=0).to(device), 
                                               mask = mask.unsqueeze(dim=0).to(device), 
                                               cond_scale = 0.7,)
            sampled_mel = sampled_mel_total[:, mask,:]
            print("sampled_mel",sampled_mel.shape)
            generate_speech = mel_decode_to_wav(generator.to(device), sampled_mel.permute(0,2,1).squeeze(0).to(device))
            filename = os.path.basename(text_file.replace(".txt",".wav"))
            write(join(saved_dir, filename), 8000, generate_speech)
            print("Saved wavfile",join(saved_dir, filename))

def cosingle_pred(phone_txt, tokenizer, text2semantic):    
    # Tokenize text
    txt_after_tokenizer = tokenizer([phone_txt], padding=True, truncation=True, return_tensors="pt")
    phoneme_input = txt_after_tokenizer.input_ids.to(device)
    
    semantic_token = text2semantic.synthesis_sample_text2semantic(phoneme_input)
    semantic_token = semantic_token.squeeze().cpu()
    return semantic_token


    
def covosinx(model, text2semantic, tokenizer, saved_dir, text_dir, prompt_dir): # 1spk means test on 1spk
    with open(os.path.join(saved_dir,"config.txt"),"w") as f:
        f.write("Vocoder: "+str(h)+'\n')
        f.write("t2s_ckpt: "+str(t2s_ckpt)+'\n')
        f.write("acoustic model: "+acous_ckpt+'\n')
    
    # Read evaluation pairs
    text_list = glob.glob(os.path.join(text_dir, "*.txt"))
    
    for text_file in tqdm(text_list):
        prompt = os.path.join(prompt_dir,os.path.basename(text_file).replace(".txt",".hubert_code.npy"))
        
        semantic_prompt, mel_prompt = prepare_oracle_hubert(prompt) 
        mel_input_A = mel_prompt 
        semantic_A = semantic_prompt
        mel_input_B = mel_prompt 
        semantic_B = semantic_prompt
        min_prompt_len = min(mel_input_A.shape[0], mel_input_B.shape[0])
        mel_input_A = mel_input_A[:min_prompt_len,:]
        mel_input_B = mel_input_B[:min_prompt_len,:]
        semantic_A = semantic_A[:min_prompt_len]
        semantic_B = semantic_B[:min_prompt_len]
        mel_input_prompt = torch.cat((mel_input_A, mel_input_B),dim=-1)
                
        
        with open (text_file, "r", encoding='utf-8') as f:
            phone_txt = f.read()
        if phone_txt != "" or phone_txt != "\n" or phone_txt != "\t" or phone_txt != " ":
            phone_txt = remove_punctuation(phone_txt)
            phone_txt = phone_txt.lower()
            phone_input1 = cosingle_pred(phone_txt, tokenizer, text2semantic)
            phone_input2 = torch.ones_like(phone_input1)*157
            semantic_A = torch.cat((semantic_A, phone_input1))
            semantic_B = torch.cat((semantic_B, phone_input2))
                

            max_phone_len = max(semantic_A.shape[0], semantic_B.shape[0])
            semantic_A = torch.nn.functional.pad(semantic_A, (0, max_phone_len-semantic_A.shape[0]), 'constant', 157)
            semantic_B = torch.nn.functional.pad(semantic_B, (0, max_phone_len-semantic_B.shape[0]), 'constant', 157)
            phone_input = torch.cat((semantic_A.unsqueeze(-1),semantic_B.unsqueeze(-1)),dim=-1)
            phone_input = torch.clamp(phone_input, max=501)

            mask = torch.zeros(phone_input.shape[0]).bool()
            mask[min_prompt_len:] = True
            mel_input = torch.zeros((phone_input.shape[0], 160))
            mel_input[:min_prompt_len,:] = mel_input_prompt

            # Synthesis and save 1 output models
            sampled_mel_total = model.synthesis_sample(phoneme_ids = phone_input.unsqueeze(dim=0).to(device), cond = mel_input.unsqueeze(dim=0).to(device), mask = mask.unsqueeze(dim=0).to(device), cond_scale = 0.7)
            valid_mel = sampled_mel_total[:, mask,:]
            generate_speech = mel_decode_to_wav(generator.to(device), valid_mel.permute(0,2,1).squeeze(0).to(device))
            #print("generate_speech1",generate_speech1.shape)
            filename = os.path.basename(text_file.replace(".txt",".wav"))
            write(join(saved_dir, filename), 8000, generate_speech)
            print("Saved wavfile",join(saved_dir, filename))

 
    
def covomix(model, text2semantic, tokenizer, saved_dir, text_dir, prompt_dir): # 1spk means test on 1spk
    # Model Initialization
    
    with open(os.path.join(saved_dir,"config.txt"),"w") as f:
        f.write("Vocoder: "+str(h)+'\n')
        f.write("t2s_ckpt: "+str(t2s_ckpt)+'\n')
        f.write("acoustic model: "+acous_ckpt+'\n')
    
    # Generation
    text_list = glob.glob(os.path.join(text_dir, "*.txt"))
    
    for text_file in tqdm(text_list):
        prompt = os.path.join(prompt_dir,os.path.basename(text_file).replace(".txt",".hubert_code.npy"))
        semantic_prompt, mel_prompt = prepare_oracle_hubert(prompt) 
        mel_input_A = mel_prompt 
        semantic_A = semantic_prompt
        mel_input_B = mel_prompt 
        semantic_B = semantic_prompt
        min_prompt_len = min(mel_input_A.shape[0], mel_input_B.shape[0])
        mel_input_A = mel_input_A[:min_prompt_len,:]
        mel_input_B = mel_input_B[:min_prompt_len,:]
        semantic_A = semantic_A[:min_prompt_len]
        semantic_B = semantic_B[:min_prompt_len]
        mel_input_prompt = torch.cat((mel_input_A, mel_input_B),dim=-1)
                
        
        with open (text_file, "r", encoding='utf-8') as f:
            phone_txt = f.read()
        
        if phone_txt != "" or phone_txt != "\n" or phone_txt != "\t" or phone_txt != " ":
            phone_txt = remove_punctuation(phone_txt)
            phone_txt = phone_txt.lower()
            
            phone_input1, phone_input2, mel_to_synthesis = comix_pred(phone_txt,tokenizer,text2semantic)  
            semantic_A = torch.cat((semantic_A, phone_input1))
            semantic_B = torch.cat((semantic_B, phone_input2))
                

            max_phone_len = max(semantic_A.shape[0], semantic_B.shape[0])
            semantic_A = torch.nn.functional.pad(semantic_A, (0, max_phone_len-semantic_A.shape[0]), 'constant', 157)
            semantic_B = torch.nn.functional.pad(semantic_B, (0, max_phone_len-semantic_B.shape[0]), 'constant', 157)
            phone_input = torch.cat((semantic_A.unsqueeze(-1),semantic_B.unsqueeze(-1)),dim=-1)
            phone_input = torch.clamp(phone_input, max=501)

            mask = torch.zeros(phone_input.shape[0]).bool()
            mask[min_prompt_len:] = True
            mel_input = torch.zeros((phone_input.shape[0], 160))
            mel_input[:min_prompt_len,:] = mel_input_prompt

            # Synthesis and save 1 output models
            sampled_mel_total = model.synthesis_sample(phoneme_ids = phone_input.unsqueeze(dim=0).to(device), cond = mel_input.unsqueeze(dim=0).to(device), mask = mask.unsqueeze(dim=0).to(device), cond_scale = 0.7)
            valid_mel = sampled_mel_total[:, mask,:]
            generate_speech = mel_decode_to_wav(generator.to(device), valid_mel.permute(0,2,1).squeeze(0).to(device))
            #print("generate_speech1",generate_speech1.shape)
            filename = os.path.basename(text_file.replace(".txt",".wav"))
            write(join(saved_dir, filename), 8000, generate_speech)
            print("Saved wavfile",join(saved_dir, filename))


def comix_pred(txt,tokenizer,text2semantic):
    
    # Tokenize text
    txt_after_tokenizer = tokenizer([txt], padding=True, truncation=True, return_tensors="pt").input_ids.to(device)
    # Predicted semantic token
    semantic_token = text2semantic.synthesis_sample_text2semantic(txt_after_tokenizer)
    semantic_token = semantic_token.squeeze().cpu()
    half = semantic_token.shape[0]//2
    semantic_token_1 = semantic_token[:half]
    semantic_token_2 = semantic_token[half:]
    
    mel_to_synthesis = torch.zeros((80,len(semantic_token_1)))
    return semantic_token_1, semantic_token_2, mel_to_synthesis
            


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--t2s_ckpt", type=str, default = "/pretrained_models/cosingle.ckpt", help='text2semantic model checkpint')
    parser.add_argument("--acous_ckpt", type=str, default="/pretrained_models/comix.ckpt", help='acoustic model checkpoint')
    parser.add_argument("--hifigan_ckpt", type=str, default="/pretrained_models/vocoder.ckpt", help="evaluation mode")
    parser.add_argument("--text_dir", type=str, default = "test/test_dir", help='directory containing text to synthesize')
    parser.add_argument("--prompt_dir", type=str, default = "test/monologue_prompt_dir", help='directory containing acoustic prompt (each monologue has 1 prompt)')
    parser.add_argument("--saved_dir", type=str, default = ".saved_dir", help='target directory')
    parser.add_argument("--seed", type=int, default = 30, help='random seed')
    parser.add_argument("--mode", type=str, choices=["covosingle", "covosinx", "covomix"], default = "covosingle", help='inference mode')

    args = parser.parse_args()
    print(args)
    
    hifigan_ckpt = args.hifigan_ckpt
    t2s_ckpt = args.t2s_ckpt
    acous_ckpt = args.acous_ckpt
    text_dir = args.text_dir
    prompt_dir = args.prompt_dir
    saved_dir = args.saved_dir
    seed = args.seed
    mode = args.mode
    ensure_dir(file_path=saved_dir)
    
    
    # Parameters for Extracting Mel-spectrogram
    global MAX_WAV_VALUE, sample_rate, hop_size, win_size, fmin, fmax, n_fft, num_mels
    MAX_WAV_VALUE = 32768.0
    sample_rate = 8000
    hop_size = 160
    win_size =  480 
    fmin= 0
    fmax = 4000
    n_fft= 480
    num_mels = 80
      
    # set seed
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
  
    # Model Initialization    
    ## Initialize vocoder
    config_file = os.path.join(os.path.split(hifigan_ckpt)[0], 'vocoder_config.json')
    with open(config_file) as f:
        data = f.read() 
    json_config = json.loads(data)
    h = AttrDict(json_config)
    
    global device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    ## Generator loading
    global generator
    generator = Generator(h).to(device)
    state_dict_g = load_checkpoint(hifigan_ckpt, device)
    generator.load_state_dict(state_dict_g['generator'])
    generator.eval()
    generator.remove_weight_norm()
    print("Successfully loaded vocoder: Hifigan-8k 20ms-hop-length traind on Fisher for 4k max frequency and 8k sampling rate, trained for 400k steps")
    
    text2semantic, tokenizer = load_text2semantic_model(t2s_ckpt)
    
    model = CoVoMixModel.load_from_checkpoint(acous_ckpt, base_dir='', batch_size=16, num_workers=0) 

    model.eval()
    model = model.to(device)

    with open(os.path.join(saved_dir,"config.txt"),"w") as f:
        f.write("Vocoder: "+str(h)+'\n')
        f.write("t2s_ckpt: "+str(t2s_ckpt)+'\n')
        f.write("acoustic model: "+acous_ckpt+'\n')
    print("Successfully loaded models, start inference...")
    
   
    
    if mode == "covosingle":
        covosingle(model, text2semantic, tokenizer, saved_dir, text_dir, prompt_dir)
    elif mode == "covosinx":
        covosinx(model, text2semantic, tokenizer, saved_dir, text_dir, prompt_dir)    
    elif mode == "covomix":
        covomix(model, text2semantic, tokenizer, saved_dir, text_dir, prompt_dir)
    else:
        print("mode ",mode," is not supported, we only support covosingle, covosinx, covomix")
    
    