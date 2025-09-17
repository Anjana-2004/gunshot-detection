import librosa
import numpy as np

def read_audio(path, sr=16000, duration=2.5):
    y, _ = librosa.load(path, sr=sr, duration=duration)
    return y

def batch_extract_features(file_list, feature_fn):
    feats = []
    for f in file_list:
        feats.append(feature_fn(f))
    return np.stack(feats)
