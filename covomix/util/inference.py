import torch
import numpy as np
from torchaudio import load

from pesq import pesq
from pystoi import stoi
import torch.nn.functional as F
import random
import os
from .other import si_sdr, pad_spec
import wespeakerruntime as wespeaker
import torchaudio.compliance.kaldi as kaldi
import random
from scipy import signal
from scipy.io import wavfile
import torchaudio.sox_effects as sox_effects
#from speechbrain.pretrained import HIFIGAN
import torchaudio 
import jiwer
from torchtext.data.metrics import bleu_score
from .generate_mel import mel_spectrogram
import librosa

# Settings
sr = 16000
snr = 0.5
N = 30
corrector_steps = 1



def evaluate_acoustic_predictor_hubert(model, num_eval_files, speech_prompt = False):
    
    device = model.device
    all_mel_files = model.data_module.valid_set.mel_files
    
    # Select test files uniformly accros validation files
    total_num_files = len(all_mel_files)
    indices = torch.linspace(0, total_num_files-1, num_eval_files, dtype=torch.int)
    mel_files = list(all_mel_files[i] for i in indices)
    phoneme_files = list(all_mel_files[i].replace(".mel.npy",".hubert_code.npy") for i in indices)


    _accuracy = 0
    _l2 = 0
    # iterate over files
    for (phoneme_file_path, mel_file_path) in zip(phoneme_files, mel_files):
        # Load wavs
        phoneme_file = np.load(phoneme_file_path).astype(int)
        mel_file = np.load(mel_file_path)
        equal_length = min(phoneme_file.shape[0], mel_file.shape[1])
        phoneme_file = torch.LongTensor(phoneme_file[:equal_length])
        mel_file = torch.Tensor(mel_file[:,:equal_length]).permute(1,0)
        
        acoustic_mask = torch.zeros_like(phoneme_file)
        acoustic_mask[:int(len(phoneme_file)*0.7)] = 1
        acoustic_mask = acoustic_mask.bool()
        
        acoustic_mask2 = torch.zeros_like(mel_file)
        acoustic_mask2[int(len(phoneme_file)*0.7):,:] = 1
        mel_input = mel_file  * acoustic_mask2
        
        predicted_mel = model.synthesis_sample(phoneme_ids = phoneme_file.unsqueeze(dim=0).to(device), 
                                                           cond = mel_input.unsqueeze(dim=0).to(device), 
                                                           mask = acoustic_mask.unsqueeze(dim=0).to(device), 
                                                           cond_scale = 0.7)
        predicted_mel = predicted_mel[:,:int(len(phoneme_file)*0.7),:].to(device)
        gt_mel = mel_file[:int(len(phoneme_file)*0.7),:].unsqueeze(dim=0).to(device)
        
        _accuracy += 0
        _l2 += F.mse_loss(predicted_mel, gt_mel)
        
        
        
    return _accuracy/num_eval_files, _l2/num_eval_files



def evaluate_acoustic_predictor_hubert_2input_2output(model, num_eval_files, speech_prompt = False):
    
    device = model.device
    all_mel_files = model.data_module.valid_set.mel_files
    
    # Select test files uniformly accros validation files
    total_num_files = len(all_mel_files)
    indices = torch.linspace(0, total_num_files-1, num_eval_files, dtype=torch.int)
    mel_files = list(all_mel_files[i].replace(".mel.npy","-A.mel.npy") for i in indices)
    phoneme_files = list(all_mel_files[i].replace(".mel.npy","-A-16k.hubert_code.npy") for i in indices)


    _accuracy = 0
    _l2 = 0
    # iterate over files
   
    for (phoneme_file_path, mel_file_path) in zip(phoneme_files, mel_files):
        # Load wavs
        phoneme_file = np.load(phoneme_file_path).astype(int)
        mel_file = np.load(mel_file_path)
        equal_length = min(phoneme_file.shape[0], mel_file.shape[1])
        
        phoneme_file2= np.load(random.choice(all_mel_files).replace(".mel.npy","-A-16k.hubert_code.npy")).astype(int)
        mel_file2 = np.load(random.choice(all_mel_files).replace(".mel.npy","-A.mel.npy"))
        equal_length2 = min(phoneme_file2.shape[0], mel_file2.shape[1])
        final_equal_length = min(equal_length, equal_length2)

        phoneme_file = torch.LongTensor(phoneme_file[:final_equal_length])
        mel_file = torch.Tensor(mel_file[:,:final_equal_length]).permute(1,0)
    
        phoneme_file2 = torch.LongTensor(phoneme_file2[:final_equal_length])
        mel_file2 = torch.Tensor(mel_file2[:,:final_equal_length]).permute(1,0)
    
        mel_file = torch.cat((mel_file, mel_file2), dim=1)
        phoneme_file = torch.cat((phoneme_file.unsqueeze(-1), phoneme_file2.unsqueeze(-1)), dim=-1)
        
        acoustic_mask = torch.zeros_like(phoneme_file)
        acoustic_mask[int(len(phoneme_file)*0.5):] = 1
        acoustic_mask = acoustic_mask.bool()
        
        acoustic_mask2 = torch.zeros_like(mel_file)
        acoustic_mask2[:int(len(phoneme_file)*0.5),:] = 1
        mel_input = mel_file  * acoustic_mask2
        
        predicted_mel = model.synthesis_sample(phoneme_ids = phoneme_file.unsqueeze(dim=0).to(device), 
                                                           cond = mel_input.unsqueeze(dim=0).to(device), 
                                                           mask = acoustic_mask.unsqueeze(dim=0).to(device), 
                                                           cond_scale = 0.7)
        predicted_mel = predicted_mel[:,int(len(phoneme_file)*0.5):,:].to(device)
        gt_mel = mel_file[int(len(phoneme_file)*0.5):,:].unsqueeze(dim=0).to(device)
        
        _accuracy += 0
        _l2 += F.mse_loss(predicted_mel, gt_mel)
        
        
        
    return _accuracy/num_eval_files, _l2/num_eval_files




def repeat_and_trim_tensor(tensor, T2):
    B,T1, D = tensor.shape
    # Calculate the total repeat factor (how many times to repeat the entire T1 dimension)
    repeat_factor = -(-T2 // T1)  # Ceiling division
    # Repeat the tensor
    extended_tensor = tensor.repeat(1,repeat_factor, 1)
    # Trim the tensor to the desired T2 length
    trimmed_tensor = extended_tensor[:,:T2, :]

    return trimmed_tensor

def evaluate_acoustic_predictor_hubert_2input_1output(model, num_eval_files, speech_prompt = False):
    
    device = model.device
    all_mel_files = model.data_module.valid_set.mel_files
    
    # Select test files uniformly accros validation files
    total_num_files = len(all_mel_files)
    indices = torch.linspace(0, total_num_files-1, num_eval_files, dtype=torch.int)
    mel_files = list(all_mel_files[i].replace(".mel.npy","-A.mel.npy") for i in indices)
    phoneme_files = list(all_mel_files[i].replace(".mel.npy","-A-16k.hubert_code.npy") for i in indices)


    _accuracy = 0
    _l2 = 0
    # iterate over files
   
    for (phoneme_file_path, mel_file_path) in zip(phoneme_files, mel_files):
        # Load wavs
        phoneme_file = np.load(phoneme_file_path).astype(int)
        mel_file = np.load(mel_file_path)
        equal_length = min(phoneme_file.shape[0], mel_file.shape[1])
        
        phoneme_file2= np.load(phoneme_file_path.replace("-A","-B")).astype(int)
        mel_file2 = np.load(mel_file_path.replace("-A","-B"))
        equal_length2 = min(phoneme_file2.shape[0], mel_file2.shape[1])
        final_equal_length = min(equal_length, equal_length2)
        
        gt_mel_file = np.load(mel_file_path.replace("-A",""))

        phoneme_file = torch.LongTensor(phoneme_file[:final_equal_length])
        mel_file = torch.Tensor(mel_file[:,:final_equal_length]).permute(1,0)
    
        phoneme_file2 = torch.LongTensor(phoneme_file2[:final_equal_length])
        mel_file2 = torch.Tensor(mel_file2[:,:final_equal_length]).permute(1,0)
        gt_mel_file = torch.Tensor(gt_mel_file[:,:final_equal_length]).permute(1,0)
    
    
        mel_file = torch.cat((mel_file, mel_file2), dim=1)
        phoneme_file = torch.cat((phoneme_file.unsqueeze(-1), phoneme_file2.unsqueeze(-1)), dim=-1)
        
        acoustic_mask = torch.zeros_like(phoneme_file)
        acoustic_mask[int(len(phoneme_file)*0.5):] = 1
        acoustic_mask = acoustic_mask.bool()
        
        acoustic_mask2 = torch.zeros_like(mel_file)
        acoustic_mask2[:int(len(phoneme_file)*0.5),:] = 1
        mel_input = mel_file  * acoustic_mask2
        
        
        if model.data_module.repeat_prompt:
            prompt_length = int(len(phoneme_file)*0.5)
            phoneme_ids = phoneme_file.unsqueeze(dim=0)[:,prompt_length:,:].to(device)
            cond = repeat_and_trim_tensor(mel_input[:prompt_length,:].unsqueeze(dim=0),phoneme_ids.shape[1]).to(device)
            print("inferemce",phoneme_ids.shape, cond.shape)
            mask = torch.ones_like(phoneme_ids).to(device)
        else: 
            phoneme_ids = phoneme_file.unsqueeze(dim=0).to(device)
            cond = mel_input.unsqueeze(dim=0).to(device)
            mask = acoustic_mask.unsqueeze(dim=0).to(device)
        print("phoneme_ids",phoneme_ids.shape,"cond",cond.shape,"mask",mask.shape)
        predicted_mel = model.synthesis_sample(phoneme_ids = phoneme_ids, 
                                                           cond = cond, 
                                                           mask = mask, 
                                                           cond_scale = 0.7)
        print("predicted_mel",predicted_mel.shape,"gt_mel_file",gt_mel_file.shape)
        if model.data_module.repeat_prompt:
            predicted_mel = predicted_mel.to(device)
        else:
            predicted_mel = predicted_mel[:,int(len(phoneme_file)*0.5):,:].to(device)
        gt_mel = gt_mel_file[int(len(phoneme_file)*0.5):,:].unsqueeze(dim=0).to(device)
        
        _accuracy += 0
        _l2 += F.mse_loss(predicted_mel, gt_mel)
        
        
        
    return _accuracy/num_eval_files, _l2/num_eval_files


def evaluate_e3tts(model, num_eval_files):
    
    device = model.device
    all_mel_files = model.data_module.valid_set.mel_files
    
    # Select test files uniformly accros validation files
    total_num_files = len(all_mel_files)
    indices = torch.linspace(0, total_num_files-1, num_eval_files, dtype=torch.int)
    mel_files = list(all_mel_files[i] for i in indices)
    phoneme_files = list(all_mel_files[i].replace(".mel.npy",".txt") for i in indices)


    _accuracy = 0
    _l2 = 0
    # iterate over files
    for (phoneme_file_path, mel_file_path) in zip(phoneme_files, mel_files):
        # Load wavs
        with open(phoneme_file_path, 'r') as file:
            texts = file.read()
        with torch.no_grad():
            pho_emb_inputs = model.data_module.tokenizer([texts], padding=True, truncation=True, return_tensors="pt")
            pho_emb_outputs = model.data_module.bert_model(**pho_emb_inputs)
            embeddings = pho_emb_outputs.last_hidden_state
            pho_input_mask_expanded = pho_emb_inputs['attention_mask'].unsqueeze(-1).expand(embeddings.size()).float()
            phoneme_input = [embeddings.to(device),pho_input_mask_expanded.to(device)]
        
        
        
        mel_file = np.load(mel_file_path)
        mel_file = torch.Tensor(mel_file).permute(1,0)
        
        length_prompt = 1024
        print("mel_file",mel_file.shape)

        if mel_file.shape[0] < length_prompt:
            mel_input = F.pad(mel_file, (0,0,0, length_prompt - mel_file.shape[0]),value=-15)
        else:
            mel_input = mel_file[:length_prompt,:]
        
        print("mel_input",mel_input.shape)
        predicted_mel = model.synthesis_sample_e3tts(phoneme_ids = phoneme_input, 
                                                           cond = mel_input.unsqueeze(dim=0).to(device), 
                                                           mask = None, 
                                                           cond_scale = 0.7)
        predicted_mel = predicted_mel.to(device)
        gt_mel = mel_input.unsqueeze(dim=0).to(device)
        
        _accuracy += 0
        _l2 += F.mse_loss(predicted_mel, gt_mel)
        
        
        
    return _accuracy/num_eval_files, _l2/num_eval_files




def evaluate_text2semantic(model, num_eval_files, speech_prompt = False):
    
    device = model.device
    all_mel_files = model.data_module.valid_set.mel_files
    
    # Select test files uniformly accros validation files
    total_num_files = len(all_mel_files)
    indices = torch.linspace(0, total_num_files-1, num_eval_files, dtype=torch.int)
    mel_files = list(all_mel_files[i]  for i in indices)
    phoneme_files = list(all_mel_files[i].replace("-16k.hubert_code.npy",".txt").replace("_1.hubert_code.npy",".txt") for i in indices)

    text2semantic_two_output = model.text2semantic_two_output
    _accuracy = 0
    _l2 = 0
    # iterate over files
    for (phoneme_file_path, mel_file_path) in zip(phoneme_files, mel_files):
        # Load wavs
        with open(phoneme_file_path, 'r') as file:
            texts = file.read()
            if model.data_module.use_spk_tag :
                texts = model.data_module.valid_set.transform_text(texts)
                print("transformed texts", texts)
        prompt_mel = None
        with torch.no_grad():
            if model.num_text_token_ids < 200: # g2p
                g2p_output = model.data_module.g2p_with_special_tokens(texts, model.data_module.tokenizer, model.data_module.phoneme_to_index)
                phoneme_input = torch.LongTensor(g2p_output).to(device).unsqueeze(0)
            else:
                pho_emb_inputs = model.data_module.tokenizer([texts], padding=True, truncation=True, return_tensors="pt")
                phoneme_input = pho_emb_inputs.input_ids.to(device)
                
        semantic_file = np.load(mel_file_path)
        semantic_file = semantic_file.astype(int)
        gt_semantic_token = torch.tensor(semantic_file).to(device)
        
        predicted_semantic_tokens = model.synthesis_sample_text2semantic(grapheme_token_ids = phoneme_input, prompt_mel = prompt_mel) 
        if text2semantic_two_output:
            half = predicted_semantic_tokens.shape[0]//2
            predicted_semantic_tokens = predicted_semantic_tokens[:half]
        
        # Pad the shorter tensor
        padding_value=501
        print("predicted_semantic_tokens",predicted_semantic_tokens.shape,"gt_semantic_token",gt_semantic_token.shape)
        max_length = max(predicted_semantic_tokens.size(0), gt_semantic_token.size(0))
        predicted_semantic_tokens = F.pad(predicted_semantic_tokens, (0, max_length - predicted_semantic_tokens.size(0)),value=padding_value).to(device)
        gt_semantic_token = F.pad(gt_semantic_token, (0, max_length - gt_semantic_token.size(0)),value=padding_value).to(device)
        
        #print("After padding predicted_semantic_tokens",predicted_semantic_tokens.shape,"gt_semantic_token",gt_semantic_token.shape)
        # matching_elements = (predicted_semantic_tokens == gt_semantic_token) & (predicted_semantic_tokens != padding_value) & (gt_semantic_token != padding_value)
        # accuracy = torch.sum(matching_elements).float() / torch.sum((predicted_semantic_tokens != padding_value) & (gt_semantic_token != padding_value)).float()
        gt_semantic_token = gt_semantic_token.squeeze()
        predicted_semantic_tokens = predicted_semantic_tokens.squeeze()
        gt_list = gt_semantic_token.tolist()
        pred_list = predicted_semantic_tokens.tolist()
        gt_str = ' '.join(map(str, gt_list))
        pred_str = ' '.join(map(str, pred_list))

        # Calculate WER for each pair
        wer = jiwer.wer(gt_str, pred_str)
        
        # Calculate BLEU score
        #candidates = [[torch.tensor([str(x)]) for x in pred_list]]
        #references = [[torch.tensor([str(x)]) for x in gt_list]]
        #print("candidates",pred_str, "references",gt_str)
        #bleu = bleu_score([pred_str], [gt_str])
        #print("blue",bleu,"wer",wer)
        _accuracy += 0.0
        _l2 += wer
        
        
        
    return _accuracy/num_eval_files, _l2/num_eval_files



def evaluate_text2semantic_bert_init(model, num_eval_files):
    
    device = model.device
    all_mel_files = model.data_module.valid_set.mel_files
    
    # Select test files uniformly accros validation files
    total_num_files = len(all_mel_files)
    indices = torch.linspace(0, total_num_files-1, num_eval_files, dtype=torch.int)
    mel_files = list(all_mel_files[i]  for i in indices)
    phoneme_files = list(all_mel_files[i].replace("-16k.hubert_code.npy",".txt") for i in indices)


    _accuracy = 0
    _l2 = 0
    # iterate over files
    for (phoneme_file_path, mel_file_path) in zip(phoneme_files, mel_files):
        # Load wavs
        with open(phoneme_file_path, 'r') as file:
            texts = file.read()
        with torch.no_grad():
            pho_emb_inputs = model.data_module.tokenizer([texts], padding=True, truncation=True, return_tensors="pt")
            #phoneme_input = pho_emb_inputs.input_ids.to(device)
            pho_emb_outputs = model.data_module.bert_model(**pho_emb_inputs)
            text_embeddings = pho_emb_outputs.last_hidden_state
            text_mask = pho_emb_inputs['attention_mask'].bool()
            model_input = [text_embeddings, text_mask]
                
        semantic_file = np.load(mel_file_path)
        semantic_file = semantic_file.astype(int)
        gt_semantic_token = torch.tensor(semantic_file).to(device)
        
        predicted_semantic_tokens = model.synthesis_sample_text2semantic(grapheme_token_ids = model_input) 
        # Pad the shorter tensor
        padding_value=501
        #print("predicted_semantic_tokens",predicted_semantic_tokens.shape,"gt_semantic_token",gt_semantic_token.shape)
        max_length = max(predicted_semantic_tokens.size(0), gt_semantic_token.size(0))
        predicted_semantic_tokens = F.pad(predicted_semantic_tokens, (0, max_length - predicted_semantic_tokens.size(0)),value=padding_value).to(device)
        gt_semantic_token = F.pad(gt_semantic_token, (0, max_length - gt_semantic_token.size(0)),value=padding_value).to(device)
        
        #print("After padding predicted_semantic_tokens",predicted_semantic_tokens.shape,"gt_semantic_token",gt_semantic_token.shape)
        # matching_elements = (predicted_semantic_tokens == gt_semantic_token) & (predicted_semantic_tokens != padding_value) & (gt_semantic_token != padding_value)
        # accuracy = torch.sum(matching_elements).float() / torch.sum((predicted_semantic_tokens != padding_value) & (gt_semantic_token != padding_value)).float()
        gt_semantic_token = gt_semantic_token.squeeze()
        predicted_semantic_tokens = predicted_semantic_tokens.squeeze()
        gt_list = gt_semantic_token.tolist()
        pred_list = predicted_semantic_tokens.tolist()
        gt_str = ' '.join(map(str, gt_list))
        pred_str = ' '.join(map(str, pred_list))

        # Calculate WER for each pair
        wer = jiwer.wer(gt_str, pred_str)
        
        # Calculate BLEU score
        #candidates = [[torch.tensor([str(x)]) for x in pred_list]]
        #references = [[torch.tensor([str(x)]) for x in gt_list]]
        #print("candidates",pred_str, "references",gt_str)
        bleu = bleu_score([pred_str], [gt_str])
        #print("blue",bleu,"wer",wer)
        _accuracy += bleu
        _l2 += wer
        
        
        
    return _accuracy/num_eval_files, _l2/num_eval_files



# Extract Mel-spectrogram

sample_rate = 8000
hop_size = 160
win_size =  480 
fmin= 0
fmax = 4000
n_fft= 480

def extract_mel(x_path,sample_rate=8000, hop_size=160, win_size=480, fmin=0, fmax=4000, n_fft=480, num_mels=80, channel_idx = None):
    x_path = x_path.replace("_hubert_code.npy",".wav").replace(".hubert_code.npy",".wav")
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
