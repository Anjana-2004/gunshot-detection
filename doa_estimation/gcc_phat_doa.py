import numpy as np
from scipy.signal import correlate
from scipy.io import wavfile

def gcc_phat(sig1, sig2, fs):
    n = len(sig1)
    corr = correlate(sig1, sig2, 'full')
    delay = np.argmax(corr) - n + 1
    tdoa = delay / fs
    angle = np.degrees(np.arcsin(tdoa * 343 / 0.2))
    return angle

def estimate_doa(audio_file="simulated_stereo.wav"):
    fs, data = wavfile.read(audio_file)
    sig1, sig2 = data[:, 0], data[:, 1]
    return gcc_phat(sig1, sig2, fs)