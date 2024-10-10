
from os.path import join
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from glob import glob
from torchaudio import load
import numpy as np
import torch.nn.functional as F
import os
from typing import Optional
import numpy as np
import torchaudio
import onnxruntime as ort
import torchaudio.compliance.kaldi as kaldi
import random
from scipy import signal
from scipy.io import wavfile
import torchaudio.sox_effects as sox_effects
from data_preparation.generate_mel import extract_and_save_mel
from torch.nn.utils.rnn import pad_sequence
from beartype.typing import Tuple, Optional
from transformers import BertTokenizer, BertModel
from covomix.online_feature_extraction import extract_mel, prepare_oracle_data_for_training, prepare_oracle_data_for_training_from_specific_file
from transformers import T5Tokenizer, T5EncoderModel
from g2p_en import G2p
from covomix.conditional_model import CoVoMixModel



##   Load text to semantic model
def load_text2semantic_model(ckpt):
    text2semantic = CoVoMixModel.load_from_checkpoint(ckpt, base_dir='', batch_size=16, num_workers=0)
    text2semantic.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_tokens(['[laughter]'])
    tokenizer.add_tokens(['[spkchange]'])
    tokenizer.add_tokens(['[spka]'])
    tokenizer.add_tokens(['[spkb]'])
    tokenizer.add_tokens(['[partialoverlap]'])
    tokenizer.add_tokens(['[backchannel]'])
    return text2semantic, tokenizer

## Predict semantic tokens
def predict_semantic_tokens(txt, tokenizer, text2semantic, g2p = False, phoneme_to_index = None, ):
    # Tokenize text
    if g2p and predict_semantic_tokens != None:
        g2p_output = global_g2p_with_special_tokens(txt, tokenizer, phoneme_to_index)
        phoneme_input = torch.LongTensor(g2p_output).unsqueeze(0)
    else:
        txt_after_tokenizer = tokenizer([txt], padding=True, truncation=True, return_tensors="pt")
        phoneme_input = txt_after_tokenizer.input_ids
    # Predicted semantic tokens
    semantic_token = text2semantic.synthesis_sample_text2semantic(phoneme_input)
    semantic_token = semantic_token.squeeze().cpu()
    return semantic_token




def get_window(window_type, window_length):
    if window_type == 'sqrthann':
        return torch.sqrt(torch.hann_window(window_length, periodic=True))
    elif window_type == 'hann':
        return torch.hann_window(window_length, periodic=True)
    else:
        raise NotImplementedError(f"Window type {window_type} not implemented!")


class Specs(Dataset):
    def __init__(self, data_dir, subset, dummy, shuffle_spec, num_frames,
            format='default', normalize="noisy", spec_transform=None, only_enhancement="no",
            stft_kwargs=None, train_noisy_data = "mix_both", **ignored_kwargs):

        # Read file paths according to file naming format.
        if format == "default":
            self.mel_files = sorted(glob(join(data_dir, subset) + '/s1/*.wav'))
            if only_enhancement=="yes":
                self.noisy_files = sorted(glob(join(data_dir, subset) + '/mix_single/*.wav'))
                print("use_mix_single_data")
            else: 
                self.noisy_files = sorted(glob(join(data_dir, subset) + '/'+train_noisy_data+'/*.wav'))
        else:
            # Feel free to add your own directory format
            raise NotImplementedError(f"Directory format {format} unknown!")

        self.dummy = dummy
        self.num_frames = num_frames
        self.shuffle_spec = shuffle_spec
        self.normalize = normalize
        self.spec_transform = spec_transform

        assert all(k in stft_kwargs.keys() for k in ["n_fft", "hop_length", "center", "window"]), "misconfigured STFT kwargs"
        self.stft_kwargs = stft_kwargs
        self.hop_length = self.stft_kwargs["hop_length"]
        assert self.stft_kwargs.get("center", None) == True, "'center' must be True for current implementation"

    def __getitem__(self, i):
        x, _ = load(self.mel_files[i])
        y, _ = load(self.noisy_files[i])

        # formula applies for center=True
        target_len = (self.num_frames - 1) * self.hop_length
        current_len = x.size(-1)
        pad = max(target_len - current_len, 0)
        if pad == 0:
            # extract random part of the audio file
            if self.shuffle_spec:
                start = int(np.random.uniform(0, current_len-target_len))
            else:
                start = int((current_len-target_len)/2)
            x = x[..., start:start+target_len]
            y = y[..., start:start+target_len]
        else:
            # pad audio if the length T is smaller than num_frames
            x = F.pad(x, (pad//2, pad//2+(pad%2)), mode='constant')
            y = F.pad(y, (pad//2, pad//2+(pad%2)), mode='constant')

        # normalize w.r.t to the noisy or the clean signal or not at all
        # to ensure same clean signal power in x and y.
        if self.normalize == "noisy":
            normfac = y.abs().max()
        elif self.normalize == "clean":
            normfac = x.abs().max()
        elif self.normalize == "not":
            normfac = 1.0
        x = x / normfac
        y = y / normfac

        X = torch.stft(x, **self.stft_kwargs)
        Y = torch.stft(y, **self.stft_kwargs)

        X, Y = self.spec_transform(X), self.spec_transform(Y)
        return X, Y

    def __len__(self):
        if self.dummy:
            # for debugging shrink the data set size
            return int(len(self.mel_files)/150)
        else:
            return len(self.mel_files)

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




def compress_tensor_sequence(repeated_semantic_sequence):
    # Assuming tensor shape is [T, 2]
    compressed_sequences = []
    counts_sequences = []
    if repeated_semantic_sequence.dim() == 1:
        repeated_semantic_sequence = repeated_semantic_sequence.unsqueeze(-1)
    for dim in range(repeated_semantic_sequence.shape[-1]): # how many spks means how many dim
        dim_seq = repeated_semantic_sequence[:, dim]
        unique_elements, counts = compress_sequence(dim_seq)
        compressed_sequences.append(unique_elements)
        counts_sequences.append(counts)
    
    
    compressed_sequences_tensor = pad_sequence(compressed_sequences, batch_first=True, padding_value=501).permute(1,0)
    counts_sequences_tensor = pad_sequence(counts_sequences, batch_first=True, padding_value=0).permute(1,0)

    return compressed_sequences_tensor, counts_sequences_tensor


def compress_sequence(seq):
    unique_elements = [seq[0]]
    counts = [1]

    for element in seq[1:]:
        if element == unique_elements[-1]:
            counts[-1] += 1
        else:
            unique_elements.append(element)
            counts.append(1)
    unique_elements = torch.LongTensor(unique_elements)
    counts = torch.LongTensor(counts)
    #print("unique_elements",unique_elements,unique_elements.shape,"counts",counts,counts.shape)
    return unique_elements, counts



class CoVoMixMel(Dataset):
    def __init__(self, data_dir, subset, dummy, shuffle_spec, num_frames,
            format='default', normalize="noisy", spec_transform=None,
            stft_kwargs=None, return_time=False, only_enhancement="no", 
            return_prompt="no",return_interference=False, train_noisy_data = "mix_both",
            extend_tasks = False, rir = False, rir_dir=None,  
            frac_lengths_mask: Tuple[float, float] = (0.7, 1.),
            duration_predictor = False, fisher_data = False,
            e3tts=False, max_spk_nums=5,more_training_data = None,
            use_spk_tag = False, repeat_spk = False,
            text2semantic_checkpoint_file = None,  random_mask = False, data_speech_prompt = False,
            **ignored_kwargs):
        self.more_training_data = False
        print("format", format, "data_dir", data_dir, "subset", subset)
        # Read file paths according to file naming format.
        if format == "default" :
            if fisher_data:
                mel_files = sorted(glob(join(data_dir, subset) + '/*/*.mel.npy'))
                phone_list_files = [f.replace('phone_by_frame.npy', 'mel.npy') for f in sorted(glob(join(data_dir, subset) + '/*/*.phone_by_frame.npy'))]
                common_files =set(mel_files) & set(phone_list_files)
                self.mel_files = list(common_files)
            else:
                self.mel_files = sorted(glob(join(data_dir, subset) + '/*/*/*.mel.npy'))
        elif format == "hubert_fisher" :
            mel_files = sorted(glob(join(data_dir, subset) + '/*.mel.npy'))
            phone_list_files = [f.replace('hubert_code.npy', 'mel.npy') for f in sorted(glob(join(data_dir, subset) + '/*.hubert_code.npy'))]
            common_files =set(mel_files) & set(phone_list_files)
            self.mel_files = list(common_files)
            if more_training_data is not None:
                mel_files2 = [f for f in sorted(glob(join(more_training_data,"*-A.mel.npy")))]
                mel_files3 = [f for f in sorted(glob(join(more_training_data,"*-B.mel.npy")))]
                common_files2 =set(mel_files2) & set(mel_files3)
                self.mel_files = list(common_files) + list(common_files2)
        elif format == "hubert_overlap_two_input_two_output" or format == "hubert_overlap_two_input_one_output":
            mel_files = sorted(glob(join(data_dir, subset) + '/*-A.mel.npy'))
            mel_files_utt = [f.replace("-A","").replace("-B","") for f in mel_files]
            phone_list_files = [f.replace('hubert_code.npy', 'mel.npy').replace("-A","").replace("-B","").replace("-16k","")  for f in sorted(glob(join(data_dir, subset) + '/*-A-16k.hubert_code.npy'))]
            common_files =set(mel_files_utt) & set(phone_list_files)
            self.mel_files = list(common_files)
            
            if more_training_data is not None:
                mel_files2 = [f for f in sorted(glob(join(more_training_data,"*.mel.npy")))]
                phone_list_files2 = [f.replace('.mel.npy', '.hubert_code.npy') for f in sorted(glob(more_training_data + '/*.mel.npy'))]
                common_files2 =set(mel_files2) & set(phone_list_files2)
                self.mel_files_prompt = list(common_files2)
            else:
                self.mel_files_prompt = []
        elif format == "text2semantic":
            mel_files = [f for f in sorted(glob(join(data_dir, subset,"*.hubert_code.npy"))) if self.get_hubert_duration(f) <= 2048]
            common_files =set(mel_files) # & set(phone_list_files)
            if more_training_data is not None:
                mel_files2 = [f for f in sorted(glob(join(more_training_data,"*.hubert_code.npy"))) if self.get_hubert_duration(f) <= 2048]
                common_files2 =set(mel_files2) # & set(phone_list_files2)
                self.mel_files = list(common_files) + list(common_files2)
            else:
                self.mel_files = list(common_files)
        elif format == "text2semantic_2output":
            mel_files = []
            for f in sorted(glob(join(data_dir, subset,"*_1.hubert_code.npy"))):
                if self.get_hubert_duration(f) <= 2048 and os.path.exists(f.replace("_1.hubert_code.npy","_2.hubert_code.npy")): 
                    mel_files.append(f)
            self.mel_files = mel_files
            if more_training_data is not None:
                more_training_data_list = more_training_data.split(',')
                mel_file_list = []
                for i in range(len(more_training_data_list)):
                    more_training_data = more_training_data_list[i]
                    mel_files2 = []
                    for f in sorted(glob(join(more_training_data,"*.hubert_code.npy"))):
                        if self.get_hubert_duration(f) <= 2048:
                            mel_files2.append(f)
                    self.more_training_data = True
                    self.mel_files = self.mel_files + mel_files2
                    mel_file_list.append(mel_files2)    
                self.mel_files2 = mel_file_list[0]
        else:
            # Feel free to add your own directory format
            raise NotImplementedError(f"Directory format {format} unknown!")

        self.dummy = dummy
        self.num_frames = num_frames
        self.shuffle_spec = shuffle_spec
        self.normalize = normalize
        self.spec_transform = spec_transform

        assert all(k in stft_kwargs.keys() for k in ["n_fft", "hop_length", "center", "window"]), "misconfigured STFT kwargs"
        self.stft_kwargs = stft_kwargs
        self.hop_length = self.stft_kwargs["hop_length"]
        assert self.stft_kwargs.get("center", None) == True, "'center' must be True for current implementation"
        
        self.return_time = return_time
        self.frac_lengths_mask = frac_lengths_mask
        self.format = format
        self.duration_predictor = duration_predictor
        self.fisher_data = fisher_data
        self.max_spk_nums = max_spk_nums
        self.use_spk_tag = use_spk_tag
        self.repeat_spk = repeat_spk
        self.random_mask = random_mask 
        self.data_speech_prompt =  data_speech_prompt
        print("length of files", len(self.mel_files))
      
    def get_duration(self, file_path):
        # Get the duration of the file
        if os.path.exists(file_path.replace(".mel.npy",".txt")):
            return np.load(file_path).shape[1]
        return 20000
    
    def get_hubert_duration(self, file_path):
        # Get the duration of the file
        file_path1 = file_path.replace(".hubert_code.npy",".txt")
        file_path2 = file_path.replace("-16k.hubert_code.npy",".txt").replace("_1.hubert_code.npy",".txt").replace("_2.hubert_code.npy",".txt")
        if "txt" in file_path1 and os.path.exists(file_path1):
            return np.load(file_path).shape[0]
        elif "txt" in file_path2 and os.path.exists(file_path2):
            return np.load(file_path).shape[0]
        else: 
            return 20000
        
    def create_random_mask(self, seq_len, mask_ratio):
        
        # Calculate the number of elements to mask
        num_elements_to_mask = int(mask_ratio * seq_len)
        
        # Generate a random start index for the mask
        start_index = np.random.randint(0, seq_len - num_elements_to_mask + 1)
        
        # Create a mask where the selected continuous elements are True
        mask = torch.zeros(seq_len)
        mask[start_index:start_index + num_elements_to_mask] = 1
        
        return mask

    

    def __getitem__(self, i):
        #print("i",i,"self.format",self.format)
        mel_gt = None
        if self.duration_predictor:
            if self.format == "default":
                duration = np.load(self.mel_files[i].replace(".mel.npy",".duration_by_phone.npy"))
                phoneme = np.load(self.mel_files[i].replace(".mel.npy",".phone_list_idx.npy"))
                
                duration = torch.tensor(duration).unsqueeze(-1)
                phoneme = torch.LongTensor(phoneme)
                # # formula applies for center=True
                max_len = 500
                current_len = duration.shape[0]
                if current_len > max_len:
                    if self.shuffle_spec:
                        start = int(np.random.uniform(0, current_len-max_len))
                    else:
                        start = int((current_len-max_len)/2)
                    duration = duration[start:start+max_len,:]
                    phoneme = phoneme[start:start+max_len]
                frac_lengths = np.random.uniform(0.1,1.0)
                mask = self.create_random_mask(seq_len = phoneme.shape[0], mask_ratio=frac_lengths)
            
            return duration, phoneme, mask
        else: 
            if self.format == "default":
                mel = np.load(self.mel_files[i])
                phoneme = np.load(self.mel_files[i].replace(".mel.npy",".phone_by_frame.npy"))
                
                mel = torch.tensor(mel).permute(1,0)
                phoneme = torch.LongTensor(phoneme)
                # # formula applies for center=True
                max_len = 1600
                current_len = mel.shape[0]
                if current_len > max_len:
                    if self.shuffle_spec:
                        start = int(np.random.uniform(0, current_len-max_len))
                    else:
                        start = int((current_len-max_len)/2)
                    mel = mel[start:start+max_len,:]
                    phoneme = phoneme[start:start+max_len]

                frac_lengths = np.random.uniform(0.5,1.0)
                mask = self.create_random_mask(seq_len = phoneme.shape[0], mask_ratio=frac_lengths)
            elif self.format == "hubert_fisher":
                mel = np.load(self.mel_files[i])
                phoneme = np.load(self.mel_files[i].replace(".mel.npy",".hubert_code.npy"))
                phoneme = phoneme.astype(int)
                
                equal_len = min(phoneme.shape[0], mel.shape[1])
                mel = mel[:,:equal_len]
                phoneme = phoneme[:equal_len]
                
                mel = torch.tensor(mel).permute(1,0)
                phoneme = torch.LongTensor(phoneme)
                
                # # formula applies for center=True
                max_len = 800
                current_len = mel.shape[0]
                if current_len > max_len:
                    if self.shuffle_spec:
                        start = int(np.random.uniform(0, current_len-max_len))
                    else:
                        start = int((current_len-max_len)/2)
                    mel = mel[start:start+max_len,:]
                    phoneme = phoneme[start:start+max_len]
                
                frac_lengths = np.random.uniform(0.5,1.0)
                mask = self.create_random_mask(seq_len = phoneme.shape[0], mask_ratio=frac_lengths)
            
            elif self.format == "hubert_overlap_two_input_two_output":
                frac_lengths = np.random.uniform(0.3,0.7)
                if os.path.exists(self.mel_files[i].replace(".mel.npy","-A.mel.npy")):
                    channel_1_path = self.mel_files[i].replace(".mel.npy","-A.mel.npy")
                    mel1, phoneme1, mask1, fix_start_point = prepare_oracle_data_for_training_from_specific_file(channel_1_path, shuffle_spec=self.shuffle_spec, frac_lengths = frac_lengths , random_mask = self.random_mask)
                if os.path.exists(self.mel_files[i].replace(".mel.npy","-B.mel.npy")):
                    channel_2_path = self.mel_files[i].replace(".mel.npy","-B.mel.npy")
                    mel2, phoneme2, mask2, fix_start_point = prepare_oracle_data_for_training_from_specific_file(channel_2_path, shuffle_spec=self.shuffle_spec, fix_start_point=fix_start_point, frac_lengths = frac_lengths, random_mask = self.random_mask)
                mel = torch.cat((mel1, mel2),dim=-1)
                mask = mask1
                phoneme = torch.cat((phoneme1.unsqueeze(-1), phoneme2.unsqueeze(-1)),dim=-1)
            elif self.format == "hubert_overlap_two_input_one_output":
                frac_lengths = np.random.uniform(0.3,0.7)
                if os.path.exists(self.mel_files[i].replace(".mel.npy","-A.mel.npy")):
                    channel_1_path = self.mel_files[i].replace(".mel.npy","-A.mel.npy")
                    mel1, phoneme1, mask1, fix_start_point = prepare_oracle_data_for_training_from_specific_file(channel_1_path, shuffle_spec=self.shuffle_spec, frac_lengths = frac_lengths, random_mask = self.random_mask)
                if os.path.exists(self.mel_files[i].replace(".mel.npy","-B.mel.npy")):
                    channel_2_path = self.mel_files[i].replace(".mel.npy","-B.mel.npy")
                    mel2, phoneme2, mask2, fix_start_point= prepare_oracle_data_for_training_from_specific_file(channel_2_path, shuffle_spec=self.shuffle_spec, fix_start_point = fix_start_point, frac_lengths = frac_lengths, random_mask = self.random_mask)
                if os.path.exists(self.mel_files[i]):
                    channel_total_path = self.mel_files[i]
                    mel3, phoneme3, mask3, fix_start_point = prepare_oracle_data_for_training_from_specific_file(channel_total_path, shuffle_spec=self.shuffle_spec,fix_start_point = fix_start_point, frac_lengths = frac_lengths, mix_1channel_mel=True, random_mask = self.random_mask)
                
                mask = mask1
                    
                if mel1.shape[0]!= mel2.shape[0] or mel1.shape[0]!= mel3.shape[0] or mel2.shape[0]!= mel3.shape[0]:
                    equal_len_mel = min(mel1.shape[0], mel2.shape[0], mel3.shape[0])
                    mel1 = mel1[:equal_len_mel,:]
                    mel2 = mel2[:equal_len_mel,:]
                    mel3 = mel3[:equal_len_mel,:]
                
                mel = torch.cat((mel1, mel2, mel3),dim=-1)
                phoneme = torch.cat((phoneme1.unsqueeze(-1), phoneme2.unsqueeze(-1)),dim=-1)
                            
            
            elif self.format == "text2semantic":
                
                mel = np.load(self.mel_files[i]) #mel here is the hubert_code
                mel = mel.astype(int)
                with open(self.mel_files[i].replace("-16k.hubert_code.npy",".txt").replace(".pitch_conversion.hubert_code.npy",".txt").replace(".hubert_code.npy",".txt"), 'r') as file: 
                    phoneme = file.read()
                
                phoneme = [phoneme]        
                mel = torch.tensor(mel) #mel here is the hubert_code
                mask = torch.zeros_like(mel)
                
            elif self.format == "text2semantic_2output":
                random_prob = random.random()
                two_spk = "_1.hubert_code.npy" in self.mel_files[i].split("/")[-1] or "_2.hubert_code.npy" in self.mel_files[i].split("/")[-1]
                if not two_spk:  
                    if random_prob < 0.40:
                        mel1 = np.load(self.mel_files[i]) #mel here is the hubert_code
                        mel1 = torch.tensor(mel1.astype(int)).unsqueeze(-1)
                        with open(self.mel_files[i].replace("-16k.hubert_code.npy",".txt").replace(".hubert_code.npy",".txt"), 'r') as file: 
                            phoneme = file.read()
                        mel2 = torch.ones_like(mel1)*157
                        mel = torch.cat((mel1,mel2),dim=-1)
                        extraction_file = self.mel_files[i]
                        
                        if self.data_speech_prompt :
                            # extract mel from gt wav
                            mel_gt1 = extract_mel(extraction_file.replace("-16k.hubert_code.npy",".wav").replace(".hubert_code.npy",".wav"))
                            mel_gt1 = torch.tensor(mel_gt1).permute(1,0)
                            mel_gt2 = torch.ones_like(mel_gt1)*(-15)
                            mel_gt = torch.cat((mel_gt1,mel_gt2),dim=-1)
                            
                    elif random_prob < 0.80 and random_prob >= 0.40:
                        mel2 = np.load(self.mel_files[i]) #mel here is the hubert_code
                        extraction_file = self.mel_files[i]
                        mel2 = torch.tensor(mel2.astype(int)).unsqueeze(-1)
                        with open(self.mel_files[i].replace("-16k.hubert_code.npy",".txt").replace(".hubert_code.npy",".txt"), 'r') as file: 
                            phoneme = file.read()
                        phoneme = ' [spkchange] '+phoneme
                        mel1 = torch.ones_like(mel2)*157
                        mel = torch.cat((mel1,mel2),dim=-1)
                        
                        if self.data_speech_prompt :    
                            mel_gt2 = extract_mel(extraction_file.replace("-16k.hubert_code.npy",".wav").replace(".hubert_code.npy",".wav"))
                            mel_gt2 = torch.tensor(mel_gt2).permute(1,0)
                            mel_gt1 = torch.ones_like(mel_gt2)*(-15)
                            mel_gt = torch.cat((mel_gt1,mel_gt2),dim=-1)
                    else: 
                        mel1 = np.load(self.mel_files[i]) #mel here is the hubert_code
                        mel1 = torch.tensor(mel1.astype(int)).unsqueeze(-1)
                        mel2_empty = torch.ones_like(mel1)*157
                        another_file = random.choice(self.mel_files2) # extract from short utterance list
                        mel2 = np.load(another_file) #mel here is the hubert_code
                        mel2 = torch.tensor(mel2.astype(int)).unsqueeze(-1)
                        mel1_empty = torch.ones_like(mel2)*157
                        with open(self.mel_files[i].replace("-16k.hubert_code.npy",".txt").replace(".hubert_code.npy",".txt"), 'r') as file: 
                            phoneme1 = file.read()
                        with open(another_file.replace("-16k.hubert_code.npy",".txt").replace(".hubert_code.npy",".txt"), 'r') as file: 
                            phoneme2 = file.read()
                        phoneme = phoneme1 + ' [spkchange] '+phoneme2
                        mel1 = torch.cat((mel1,mel1_empty),dim=0)
                        mel2 = torch.cat((mel2_empty,mel2),dim=0)
                        mel = torch.cat((mel1,mel2),dim=-1)
                        extraction_file = self.mel_files[i]
                        
                        if self.data_speech_prompt :
                            mel_gt1 = extract_mel(extraction_file.replace("-16k.hubert_code.npy",".wav").replace(".hubert_code.npy",".wav"))
                            mel_gt1 = torch.tensor(mel_gt1).permute(1,0)
                            mel_gt2 = extract_mel(another_file.replace("-16k.hubert_code.npy",".wav").replace(".hubert_code.npy",".wav"))
                            mel_gt2 = torch.tensor(mel_gt2).permute(1,0)
                            min_len = min(mel_gt1.shape[0],mel_gt2.shape[0])
                            mel_gt = torch.cat((mel_gt1[:min_len,:],mel_gt2[:min_len,:]),dim=-1)
                else: # Normal 2 spk utt
                    mel1 = np.load(self.mel_files[i]) #mel here is the hubert_code
                    mel1 = torch.tensor(mel1.astype(int)).unsqueeze(-1)
                    mel2 = np.load(self.mel_files[i].replace("_1.hubert_code.npy","_2.hubert_code.npy")) #mel here is the hubert_code
                    mel2 = torch.tensor(mel2.astype(int)).unsqueeze(-1)
                    mel = torch.cat((mel1,mel2),dim=-1)
                    with open(self.mel_files[i].replace("_1.hubert_code.npy",".txt"), 'r') as file: 
                        phoneme = file.read()
                    extraction_file = self.mel_files[i]
                    
                    if self.data_speech_prompt :
                        mel_gt1 = extract_mel(extraction_file.replace("-16k.hubert_code.npy",".wav").replace(".hubert_code.npy",".wav"))
                        mel_gt1 = torch.tensor(mel_gt1).permute(1,0)
                        mel_gt2 = extract_mel(extraction_file.replace("_1.hubert_code.npy","_2.hubert_code.npy").replace("-16k.hubert_code.npy",".wav").replace(".hubert_code.npy",".wav"))
                        mel_gt2 = torch.tensor(mel_gt2).permute(1,0)
                        mel_gt = torch.cat((mel_gt1,mel_gt2),dim=-1)
                if self.use_spk_tag:
                    phoneme = self.transform_text(phoneme)
                phoneme = [phoneme]        
                mask = torch.zeros_like(mel)
            
            if self.data_speech_prompt and mel_gt is not None:
                return mel, phoneme, mask, mel_gt
            return mel, phoneme, mask
        
    def transform_text(self, input_text):
        # Split the text at each [spkchange]
        segments = input_text.split("[spkchange]")

        # Initialize an empty list to hold the transformed segments
        transformed_segments = []

        # Loop through each segment and prepend the appropriate speaker tag
        for i, segment in enumerate(segments):
            speaker_tag = " [spka]" if i % 2 == 0 else " [spkb]"
            transformed_segments.append(speaker_tag + " " + segment.strip())

        # Join the transformed segments into a single string
        return " ".join(transformed_segments)
    

    def __len__(self):
        if self.dummy:
            # for debugging shrink the data set size
            return int(len(self.mel_files)/150)
        else:
            return len(self.mel_files)
        







class SpecsDataModule(pl.LightningDataModule):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--base_dir", type=str, required=True, help="The base directory of the trainining dataset.")
        parser.add_argument("--format", type=str, default="default", help="Read file paths according to file naming format.")
        parser.add_argument("--batch_size", type=int, default=8, help="The batch size. 8 by default.")
        parser.add_argument("--n_fft", type=int, default=510, help="Number of FFT bins. 510 by default.")   # to assure 256 freq bins
        parser.add_argument("--hop_length", type=int, default=128, help="Window hop length. 128 by default.")
        parser.add_argument("--num_frames", type=int, default=256, help="Number of frames for the dataset. 256 by default.")
        parser.add_argument("--window", type=str, choices=("sqrthann", "hann"), default="hann", help="The window function to use for the STFT. 'hann' by default.")
        parser.add_argument("--num_workers", type=int, default=4, help="Number of workers to use for DataLoaders. 4 by default.")
        parser.add_argument("--dummy", action="store_true", help="Use reduced dummy dataset for prototyping.")
        parser.add_argument("--spec_factor", type=float, default=0.15, help="Factor to multiply complex STFT coefficients by. 0.15 by default.")
        parser.add_argument("--spec_abs_exponent", type=float, default=0.5, help="Exponent e for the transformation abs(z)**e * exp(1j*angle(z)). 0.5 by default.")
        parser.add_argument("--normalize", type=str, choices=("clean", "noisy", "not"), default="noisy", help="Normalize the input waveforms by the clean signal, the noisy signal, or not at all.")
        parser.add_argument("--transform_type", type=str, choices=("exponent", "log", "none"), default="exponent", help="Spectogram transformation for input representation.")
        parser.add_argument("--condition_on_spkemb", type=str, choices=("no", "yes"), default="no", help="no for Spec, yes for ConditionalSpec")
        parser.add_argument("--return_time", action="store_true", help="Return the waveform instead of the STFT")
        parser.add_argument("--only_enhancement", type=str, choices=("no", "yes"), default="no", help="training and testing using mix_single")
        parser.add_argument("--return_prompt", type=str, choices=("no", "yes"), default="no", help="return prompt stft for dataloader")
        parser.add_argument("--return_interference", action="store_true", help="Return the interference speech")
        parser.add_argument("--train_subset", type=str,  default="train-360", help="Return the interference speech")
        parser.add_argument("--extend_tasks", action="store_true", help="Extend tasks to non-personalized se and non target extraction")
        parser.add_argument("--train_noisy_data", type=str, choices=("mix_both", "mix_clean","mix_single"), default="mix_both", help="Extend tasks to non-personalized se and non target extraction")
        parser.add_argument("--rir_dir", type=str, help="The base directory of the rir dataset")
        parser.add_argument("--rir", action="store_true", help="Extend tasks to non-personalized se and non target extraction")
        parser.add_argument("--dev_base_dir", type=str, default="LibriTTS_R",  help="The base directory of the dev dataset.")
        parser.add_argument("--dev_subset", type=str, default="dev-clean-textgrid-ouralignment/mfa_output", help="The name of the dev directory")
        parser.add_argument("--duration_predictor", action="store_true", help="Use phoneme by frame or duration per phoneme")
        parser.add_argument("--fisher_data", action="store_true", help="Use phoneme by frame or duration per phoneme")
        parser.add_argument("--e3tts", action="store_true", help="Use Unet similar to e3tts")
        parser.add_argument("--text2semantic_data", action="store_true", help="data prep for text2semantic ")
        parser.add_argument("--text2semantic_remove_uh", type=float, default=-1.0, help="remove uh for text2semantic")
        parser.add_argument("--max_spk_num", type=int, default=5, help="max speaker number")
        parser.add_argument("--laughter_tokenizer", action="store_true", help="Use laughter tokenizer")
        parser.add_argument("--more_training_data", type=str,help="Use more data to train the model")
        parser.add_argument("--bert_init_emb", action="store_true", help="Use bert embedding instead of nn.Embedding")
        parser.add_argument("--t5_init_emb", action="store_true", help="Use T5 small embedding instead of nn.Embedding")
        parser.add_argument("--data_pred_duration", action="store_true", help="text2semantic predict duration and non-repeated code simultaneously")
        parser.add_argument("--g2p", action="store_true", help="text2semantic regard g2p as tokenizer")
        parser.add_argument("--use_spk_tag", action="store_true", help="text2semantic use [spkA] and [spkB] instead of [spkchange]")
        parser.add_argument("--repeat_prompt",  action='store_true',  help="acoustic model does mask prompt, but repeat it as a new channel")
        parser.add_argument("--text2semantic_checkpoint_file", type=str,  default="", help="text2semantic_checkpoint_file")
        parser.add_argument("--random_mask",  action='store_true',  help="acoustic model random mask instead of fix mask in the beginning")
        parser.add_argument("--data_speech_prompt",  action='store_true',  help="text2semantic model get extra input of ground truth mel-spectrogram")

    
        return parser

    def __init__(
        self, base_dir, format='default', batch_size=8,
        n_fft=510, hop_length=128, num_frames=256, window='hann',
        num_workers=4, dummy=False, spec_factor=0.15, spec_abs_exponent=0.5,
        gpu=True, normalize='noisy', transform_type="exponent",condition_on_spkemb="no", 
        return_time=False, only_enhancement="no", train_subset = "train-360",
        train_noisy_data = "mix_both",dev_base_dir = "LibriTTS_R", 
        dev_subset="dev-clean-textgrid-ouralignment/mfa_output",
        duration_predictor = False, fisher_data=False, e3tts = False,
        text2semantic_data = False, text2semantic_remove_uh = -1.0, max_spk_num =5,
        laughter_tokenizer = False, more_training_data = None,bert_init_emb = False, t5_init_emb = False,
        t5_finetune_only_decoder = False,data_pred_duration= False, g2p = False,
        use_spk_tag = False, repeat_prompt =False, text2semantic_checkpoint_file="",
        random_mask = False,data_speech_prompt=False,
        **kwargs
    ):
        super().__init__()
        self.base_dir = base_dir
        #print("self.base_dir", self.base_dir)
        self.format = format
        self.batch_size = batch_size
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_frames = num_frames
        self.window = get_window(window, self.n_fft)
        self.windows = {}
        self.num_workers = num_workers
        self.dummy = dummy
        self.spec_factor = spec_factor
        self.spec_abs_exponent = spec_abs_exponent
        self.gpu = gpu
        self.normalize = normalize
        self.transform_type = transform_type
        self.kwargs = kwargs
        self.condition_on_spkemb = condition_on_spkemb
        self.return_time= return_time
        self.only_enhancement=only_enhancement
        self.train_subset=train_subset
        self.train_noisy_data = train_noisy_data
        self.dev_base_dir=dev_base_dir
        self.dev_subset = dev_subset
        self.duration_predictor = duration_predictor
        self.fisher_data = fisher_data
        self.text2semantic_data = text2semantic_data
        self.e3tts = e3tts 
        self.max_spk_num = max_spk_num
        self.laughter_tokenizer = laughter_tokenizer
        self.bert_init_emb  = bert_init_emb 
        self.t5_init_emb = t5_init_emb
        self.t5_finetune_only_decoder = t5_finetune_only_decoder
        self.data_pred_duration = data_pred_duration
        self.g2p = g2p
        self.use_spk_tag = use_spk_tag
        self.text2semantic_checkpoint_file = text2semantic_checkpoint_file
        self.data_speech_prompt = data_speech_prompt

        if self.e3tts:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        elif self.text2semantic_data:
            if not self.t5_init_emb and not self.g2p:
                self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                if self.laughter_tokenizer:
                    self.tokenizer.add_tokens(['[laughter]'])
                    self.tokenizer.add_tokens(['[spkchange]'])
                    self.tokenizer.add_tokens(['[partialoverlap]'])
                    self.tokenizer.add_tokens(['[backchannel]'])
                    self.tokenizer.add_tokens(['[spka]'])
                    self.tokenizer.add_tokens(['[spkb]'])
                if self.bert_init_emb:
                    self.bert_model = BertModel.from_pretrained('bert-base-uncased')
                    self.bert_model.resize_token_embeddings(len( self.tokenizer))
            elif self.g2p:
                self.tokenizer = G2p()
                self.tokenizer.phonemes.append('[laughter]')
                self.tokenizer.phonemes.append('[spkchange]')
                self.tokenizer.phonemes.append('[partialoverlap]')
                self.tokenizer.phonemes.append('[backchannel]')
                self.tokenizer.phonemes.append('[SEP]')
                self.tokenizer.phonemes.append('[spka]')
                self.tokenizer.phonemes.append('[spkb]')
                self.phoneme_to_index = {phoneme: idx for idx, phoneme in enumerate(self.tokenizer.phonemes)}
                print("self.phoneme_to_index",self.phoneme_to_index)

            elif self.t5_finetune_only_decoder:
                self.t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
                special_tokens_dict = {'additional_special_tokens': ['[laughter]']}
                self.prefix =  "translate english to semantic tokens: "
                # self.t5_tokenizer.add_special_tokens(special_tokens_dict)
                # self.t5_model = T5EncoderModel.from_pretrained("t5-small")
                # self.t5_model.resize_token_embeddings(len(self.t5_tokenizer))
            else:
                self.t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
                special_tokens_dict = {'additional_special_tokens': ['[laughter]']}
                self.t5_tokenizer.add_special_tokens(special_tokens_dict)
                self.t5_model = T5EncoderModel.from_pretrained("t5-small")
                self.t5_model.resize_token_embeddings(len(self.t5_tokenizer))

        self.text2semantic_remove_uh = text2semantic_remove_uh
        self.uh_list = [' uh ', ' um ', ' mm ', ' hm ', ' mhm ', 
                        ' hmm ', ' hmm ', ' huh ', ' uhh ', ' umh ', ' ummm ', ' umm ', ' ummm ', 
                        ' em ', ' eh', ' ehh ', ' ehm ', ' ehmm ', ' ehm ',
                        ' ah ', ' ahh ', ' ahm ', ' ahmm ', ' ahm ', ' ahah ']
        self.more_training_data = more_training_data
        self.repeat_prompt = repeat_prompt
        self.random_mask = random_mask


    def setup(self, stage=None):
        specs_kwargs = dict(
            stft_kwargs=self.stft_kwargs, num_frames=self.num_frames,
            spec_transform=self.spec_fwd, **self.kwargs
        )
        if stage == 'fit' or stage is None:
            self.train_set = CoVoMixMel(data_dir=self.base_dir, subset=self.train_subset,
                dummy=self.dummy, shuffle_spec=True, format=self.format,only_enhancement=self.only_enhancement,
                normalize=self.normalize, return_time = self.return_time, train_noisy_data = self.train_noisy_data,
                duration_predictor = self.duration_predictor, fisher_data = self.fisher_data, max_spk_nums=self.max_spk_num, 
                more_training_data =self.more_training_data, use_spk_tag = self.use_spk_tag, repeat_prompt = self.repeat_prompt,
                text2semantic_checkpoint_file =self.text2semantic_checkpoint_file, random_mask = self.random_mask,
                data_speech_prompt=self.data_speech_prompt,
                **specs_kwargs)
            self.valid_set = CoVoMixMel(data_dir=self.dev_base_dir, subset=self.dev_subset,
                dummy=True, shuffle_spec=False, return_time = self.return_time,format=self.format,
                only_enhancement=self.only_enhancement, train_noisy_data = self.train_noisy_data,
                duration_predictor = self.duration_predictor,fisher_data = self.fisher_data,  max_spk_nums=self.max_spk_num, 
                use_spk_tag = self.use_spk_tag, repeat_prompt = self.repeat_prompt, random_mask = self.random_mask,
                normalize=self.normalize, 
                text2semantic_checkpoint_file =self.text2semantic_checkpoint_file,
                data_speech_prompt=self.data_speech_prompt,  
                **specs_kwargs)
            
        if stage == 'validate' or stage is None:
            self.valid_set = CoVoMixMel(data_dir=self.dev_base_dir, subset=self.dev_subset,
                dummy=True, shuffle_spec=False, return_time = self.return_time,format=self.format,
                only_enhancement=self.only_enhancement, train_noisy_data = self.train_noisy_data,
                duration_predictor = self.duration_predictor,fisher_data = self.fisher_data, max_spk_nums=self.max_spk_num,
                use_spk_tag = self.use_spk_tag, repeat_prompt = self.repeat_prompt, 
                text2semantic_checkpoint_file =self.text2semantic_checkpoint_file,
                data_speech_prompt=self.data_speech_prompt,
                normalize=self.normalize, **specs_kwargs)
        

    def spec_fwd(self, spec):
        if self.transform_type == "exponent":
            if self.spec_abs_exponent != 1:
                # only do this calculation if spec_exponent != 1, otherwise it's quite a bit of wasted computation
                # and introduced numerical error
                e = self.spec_abs_exponent
                spec = spec.abs()**e * torch.exp(1j * spec.angle())
            spec = spec * self.spec_factor
        elif self.transform_type == "log":
            spec = torch.log(1 + spec.abs()) * torch.exp(1j * spec.angle())
            spec = spec * self.spec_factor
        elif self.transform_type == "none":
            spec = spec
        return spec

    def spec_back(self, spec):
        if self.transform_type == "exponent":
            spec = spec / self.spec_factor
            if self.spec_abs_exponent != 1:
                e = self.spec_abs_exponent
                spec = spec.abs()**(1/e) * torch.exp(1j * spec.angle())
        elif self.transform_type == "log":
            spec = spec / self.spec_factor
            spec = (torch.exp(spec.abs()) - 1) * torch.exp(1j * spec.angle())
        elif self.transform_type == "none":
            spec = spec
        return spec

    @property
    def stft_kwargs(self):
        return {**self.istft_kwargs, "return_complex": True}

    @property
    def istft_kwargs(self):
        return dict(
            n_fft=self.n_fft, hop_length=self.hop_length,
            window=self.window, center=True
        )

    def _get_window(self, x):
        """
        Retrieve an appropriate window for the given tensor x, matching the device.
        Caches the retrieved windows so that only one window tensor will be allocated per device.
        """
        window = self.windows.get(x.device, None)
        if window is None:
            window = self.window.to(x.device)
            self.windows[x.device] = window
        return window

    def stft(self, sig):
        window = self._get_window(sig)
        return torch.stft(sig, **{**self.stft_kwargs, "window": window})

    def istft(self, spec, length=None):
        window = self._get_window(spec)
        return torch.istft(spec, **{**self.istft_kwargs, "window": window, "length": length})

    def train_dataloader(self):
        return DataLoader(
            self.train_set, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=self.gpu, shuffle=True,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_set, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=self.gpu, shuffle=False,
            collate_fn=self.collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=self.gpu, shuffle=False,
            collate_fn=self.collate_fn
        )
        
    def collate_fn(self, batch):
        condition_batch_padded = None    
        # Unpack the batch into separate lists for mel and pho
        if self.data_speech_prompt:
            mel_batch, pho_batch, mask_batch, mel_gt_batch = zip(*batch)
        else:
            mel_batch, pho_batch, mask_batch = zip(*batch)
                        
        # Pad the mel spectrograms to the maximum sequence length in the batch
        if  self.data_speech_prompt:
            mel_gt_batch_padded = pad_sequence(mel_gt_batch, batch_first=True, padding_value=-15)
        else: 
            mel_gt_batch_padded = pad_sequence(mel_batch, batch_first=True, padding_value=-15)
        
        
        if self.duration_predictor:
            mel_batch_padded = pad_sequence(mel_batch, batch_first=True, padding_value=0)
        elif self.text2semantic_data and (not self.data_pred_duration): #The mel here is the hubert code
            #print("mel_batch,",mel_batch[0].shape)
            mel_batch_padded = pad_sequence(mel_batch, batch_first=True, padding_value=501)
        elif self.text2semantic_data and  self.data_pred_duration:
            unrepeated_token_batch = []
            duration_batch = []
            for mel_batch_i in mel_batch:
                unrepeated_token_tensor, duration_tensor = compress_tensor_sequence(mel_batch_i)
                unrepeated_token_batch.append(unrepeated_token_tensor)
                duration_batch.append(duration_tensor)
            unrepeated_token_batch_padded = pad_sequence(unrepeated_token_batch, batch_first=True, padding_value=501)
            duration_list_padded = pad_sequence(duration_batch, batch_first=True, padding_value=0)
            mel_batch_padded = (unrepeated_token_batch_padded, duration_list_padded)
        elif self.repeat_prompt:
            pho_batch_new = []
            mel_batch_new = [] # target
            cond_batch_new = []
            mask_batch_new = []
            for i in range(len(mel_batch)):
                mel_batch_i = mel_batch[i]
                pho_batch_i = pho_batch[i]
                total_length = mel_batch_i.shape[0]
                prompt_length = random.randint(45, min(int(total_length*0.4),400))
                target = mel_batch_i[prompt_length:,:]
                mel_batch_new.append(target)
                cond = self.repeat_and_trim_tensor(mel_batch_i[:prompt_length,:-80],target.shape[0] )
                pho_batch_new.append(pho_batch_i[prompt_length:])
                cond_batch_new.append(cond)
                mask_batch_new.append(torch.ones_like(pho_batch_i[prompt_length:,0]))
            mel_batch_padded = pad_sequence(mel_batch_new, batch_first=True, padding_value=-15)
            condition_batch_padded = pad_sequence(cond_batch_new, batch_first=True, padding_value=-15)
            pho_batch = pho_batch_new
            mask_batch = mask_batch_new
        else: 
            mel_batch_padded = pad_sequence(mel_batch, batch_first=True, padding_value=-15)
        
        # Pad the pho sequences to the maximum sequence length in the batch
        if self.duration_predictor:
            pho_batch_padded = pad_sequence(pho_batch, batch_first=True, padding_value=100)
        elif self.e3tts:
            texts = []
            for pho in pho_batch:
                if self.text2semantic_remove_uh > 0.0 and random.random() < self.text2semantic_remove_uh:
                    for word in self.uh_list:
                        input_string = pho[0].replace(word," ")
                    texts.append(input_string)
                else: 
                    texts.append(pho[0])
            
            with torch.no_grad():
                pho_emb_inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
                pho_emb_outputs = self.bert_model(**pho_emb_inputs)
                embeddings = pho_emb_outputs.last_hidden_state
                pho_input_mask_expanded = pho_emb_inputs['attention_mask'].unsqueeze(-1).expand(embeddings.size()).float()
                pho_batch_padded = [embeddings,pho_input_mask_expanded]
        elif  self.text2semantic_data:
            texts = []
            for pho in pho_batch:
                if self.text2semantic_remove_uh > 0.0 and random.random() < self.text2semantic_remove_uh:
                    for word in self.uh_list:
                        input_string = pho[0].replace(word," ")
                    texts.append(input_string)
                else: 
                    texts.append(pho[0])
            
            if self.bert_init_emb: 
                with torch.no_grad():
                    pho_emb_inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
                    pho_emb_outputs = self.bert_model(**pho_emb_inputs)
                    embeddings = pho_emb_outputs.last_hidden_state
                    pho_batch_padded = [embeddings, pho_emb_inputs['attention_mask'].bool()]
            elif self.t5_init_emb:
                with torch.no_grad():
                    encoding = self.t5_tokenizer(texts, padding="longest",max_length=2048,truncation=True,return_tensors="pt")
                    input_ids, attention_mask = encoding.input_ids, encoding.attention_mask
                    outputs = self.t5_model(input_ids=input_ids, attention_mask=attention_mask)
                    embeddings = outputs.last_hidden_state
                    pho_batch_padded = [embeddings, attention_mask.bool()]
            elif self.t5_finetune_only_decoder:
                text_with_prefix = [self.prefix + text for text in texts]
                with torch.no_grad():
                    encoding = self.t5_tokenizer(text_with_prefix, padding="longest",max_length=2048,truncation=True,return_tensors="pt")
            elif self.g2p:
                pho_batch = []
                for text in  texts:
                    g2p_output = self.g2p_with_special_tokens(text,self.tokenizer, self.phoneme_to_index)
                    pho_batch.append(g2p_output)
                pho_batch_padded = pad_sequence(pho_batch, batch_first=True, padding_value=0)
            else:
                with torch.no_grad():
                    pho_emb_inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
                    pho_batch_padded = pho_emb_inputs.input_ids
        elif "hubert" in self.format:
            pho_batch_padded = pad_sequence(pho_batch, batch_first=True, padding_value=501)
        else: 
            pho_batch_padded = pad_sequence(pho_batch, batch_first=True, padding_value=92)
        
        mask_batch_padded = pad_sequence(mask_batch, batch_first=True, padding_value=0)
        mask_batch_padded = mask_batch_padded.bool()
            
        return mel_batch_padded, pho_batch_padded, mask_batch_padded, condition_batch_padded, mel_gt_batch_padded  # [B,T,80] for mel, [B,T] for mask and phoneme
        
    
    def repeat_and_trim_tensor(self, tensor, T2):
        T1, D = tensor.shape
        # Calculate the total repeat factor (how many times to repeat the entire T1 dimension)
        repeat_factor = -(-T2 // T1)  # Ceiling division
        # Repeat the tensor
        extended_tensor = tensor.repeat(repeat_factor, 1)
        # Trim the tensor to the desired T2 length
        trimmed_tensor = extended_tensor[:T2, :]

        return trimmed_tensor
            
    def g2p_with_special_tokens(self, txt, g2p_tokenizer, phoneme_to_index):
        return global_g2p_with_special_tokens(txt, g2p_tokenizer, phoneme_to_index)

        
        
def reverberate(audio, rir_audio_file):
        audio = audio.squeeze().numpy()
        audio = audio.astype(np.float32)
        audio_len = audio.shape[0]
        
        rir_audio, _ = load(rir_audio_file) 
        rir_audio = rir_audio.squeeze().numpy()

        rir_audio = rir_audio.astype(np.float32)
        rir_audio = rir_audio / np.sqrt(np.sum(rir_audio ** 2))

        return signal.convolve(audio, rir_audio, mode='full')[:audio_len]




def global_g2p_with_special_tokens(txt, g2p_tokenizer, phoneme_to_index):
    normalize_string = ',.!?;:()' + '"'+ "'"+'-'+' '+'\n'+ '\t'+ '\r'+ '\x0b'+ '\x0c'+ '\ufeff'
    for i in normalize_string:
        txt = txt.replace(i,' ')
                
    tokens = txt.split(' ')

    # Process the tokens
    output_tokens = []
    for token in tokens:
        if token == '[laughter]' or token == '[spkchange]':
            output_tokens.append(token)
        else:
            # If it's a regular word, apply g2p_en conversion
            phonemes = g2p_tokenizer(token)
            output_tokens.extend(phonemes)
        output_tokens.append('[SEP]')
    try:
        indices = [phoneme_to_index[phoneme] for phoneme in output_tokens]
    except:
        print("txt",txt,"output_tokens",output_tokens)
        print("phoneme_to_index",phoneme_to_index)
    indices = torch.LongTensor(indices)
    return indices

    