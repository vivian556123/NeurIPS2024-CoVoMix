from scipy.io.wavfile import read, write
import torchaudio
import torch
from librosa.util import normalize
from librosa.filters import mel as librosa_mel_fn
import numpy as np
import librosa
import argparse
import librosa.display
from tqdm import tqdm
import os
import soundfile as sf
import matplotlib.pyplot as plt
import parser
import glob
import wespeakerruntime as wespeaker

#%matplotlib inline

MAX_WAV_VALUE = 32768.0

def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate

def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)

def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output

mel_basis = {}
hann_window = {}

def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True,return_complex=False)

    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec

def extract_and_save_mel(data_path, save_path, wav_files):
    for file_name in tqdm(wav_files[:]):
        
        try:
            wav, sr = librosa.load(os.path.join(data_path, file_name), sr=22050)
        except:
            print("error file",os.path.join(data_path, file_name))
            continue
        wav = np.clip(wav, -1, 1)
        x = torch.FloatTensor(wav)
        # print(len(x))
        x = mel_spectrogram(x.unsqueeze(0), n_fft=1024, num_mels=80, sampling_rate=8000,
                        hop_size=160, win_size=480, fmin=0, fmax=4000)
        #print("x.shape",x.shape)
        spec = x.cpu().numpy()[0]
        #print("spec.shape",spec.shape)
        wav = wav * MAX_WAV_VALUE
        wav = wav.astype('int16')
        save_file_name = os.path.basename(file_name)
        print("path", os.path.join(save_path, "mel",save_file_name.replace(".wav", ".npy")))
        write(os.path.join(save_path, "wav",save_file_name), 22050, wav)
        np.save(os.path.join(save_path, "mel",save_file_name.replace(".wav", ".npy")), spec)

def extract_and_save_spk_emb(data_path, save_path, wav_files,speaker):
    for file_name in tqdm(wav_files[:]):
        try:
            wav, sr = librosa.load(os.path.join(data_path, file_name), sr=16000)
        except:
            print("error file",os.path.join(data_path, file_name))
            continue
        spk_emb=speaker.extract_embedding(os.path.join(data_path, file_name))

        np.save(os.path.join(save_path, "spkemb", file_name.replace(".wav", ".npy")), spk_emb)

def stft(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=True):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=False,return_complex=True)

    return spec

def extract_and_save_stft(data_path, save_path, wav_files):
    for file_name in tqdm(wav_files[:]):
        try:
            wav, sr = librosa.load(os.path.join(data_path, file_name), sr=16000)
        except:
            print("error file",os.path.join(data_path, file_name))
            continue
        wav = np.clip(wav, -1, 1)
        x = torch.FloatTensor(wav)
        # print(len(x))
        spec = stft(x.unsqueeze(0), n_fft=512, num_mels=80, sampling_rate=16000,
                        hop_size=256, win_size=512, fmin=0, fmax=8000, center=True)
        #print("x.shape",x.shape)
        #spec = x.cpu().numpy()[0]
        #print("spec.shape",spec.shape)
        spec = spec.cpu().numpy()
        np.save(os.path.join(save_path, "stft_512", file_name.replace(".wav", ".npy")), spec)
        

def save_wav(data_path, save_path, wav_files):
    for file_name in tqdm(wav_files[:]):
        try:
            wav, sr = librosa.load(os.path.join(data_path, file_name), sr=16000)
        except:
            print("error file",os.path.join(data_path, file_name))
            continue
        wav = wav - np.mean(wav)
        wav = wav / np.max(np.abs(wav))
        #wav = np.clip(wav, -1, 1)
        #wav = wav * MAX_WAV_VALUE
        #wav = wav.astype('int16')
        write(os.path.join(save_path, file_name), 16000, wav)


# main function
if __name__ == "__main__":
    # input parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="/home/v-leyzhang/audioset-processing/output/bird")
    parser.add_argument('--save_path', type=str, default="/home/v-leyzhang/audioset-processing/output/bird_mel")
    args = parser.parse_args()
    data_path = args.data_path
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(os.path.join(save_path, "wav")):
        os.makedirs(os.path.join(save_path, "wav"))
    if not os.path.exists(os.path.join(save_path, "mel")):
        os.makedirs(os.path.join(save_path, "mel"))

        
    wav_files = os.listdir(data_path)
    wav_files = glob.glob(os.path.join(data_path,"*.wav"))
    print("wav_files",len(wav_files))
    #extract_and_save_mel(data_path, save_path, wav_files)
    #speaker = wespeaker.Speaker(lang='en')
    #extract_and_save_stft(data_path, save_path, wav_files)
    extract_and_save_mel(data_path, save_path, wav_files)
    
    # Process LibriSpeech:  
    # wav_files = glob.glob(os.path.join(data_path, "*/*"))
    # for sub_data_path in wav_files:
    #     wav_files = os.listdir(sub_data_path)
    #     extract_and_save_mel(sub_data_path, save_path, wav_files)