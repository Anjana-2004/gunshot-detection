import librosa
import numpy as np
import torch

def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y, sr=sr, n_mfcc=40)
    mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
    mfcc = mfcc[np.newaxis, np.newaxis, :, :40]
    return torch.tensor(mfcc, dtype=torch.float32)