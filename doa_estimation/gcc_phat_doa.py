import numpy as np

def gcc_phat(sig, refsig, fs=1, max_tau=None, interp=16):
    # Compute cross-spectrum
    n = sig.shape[0] + refsig.shape[0]
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)
    cc = np.fft.irfft(R / (np.abs(R) + np.finfo(float).eps), n=(interp * n))
    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))
    shift = np.argmax(np.abs(cc)) - max_shift
    tau = shift / float(interp * fs)
    return tau, cc

def estimate_doa(signals, mic_distance, fs=16000, sound_velocity=343):
    # Here, signals: list or array of microphone signals; for two-mic, signals = [sig1, sig2]
    tau, cc = gcc_phat(signals[0], signals[1], fs=fs, max_tau=mic_distance/sound_velocity)
    doa = np.arcsin(tau * sound_velocity / mic_distance) * 180 / np.pi
    return doa, cc

# Example usage on synthetic signals for test/demo
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    fs = 16000
    duration = 1.0
    mic_distance = 0.08  # 8 cm
    sound_velocity = 343
    angle = 30  # degrees

    t = np.linspace(0, duration, int(duration * fs))
    source = np.sin(2 * np.pi * 400 * t)

    delay = mic_distance * np.sin(np.deg2rad(angle)) / sound_velocity
    n_samples_delay = int(delay * fs)

    sig1 = source
    sig2 = np.roll(source, n_samples_delay)

    doa, cc = estimate_doa([sig1, sig2], mic_distance, fs, sound_velocity)
    print(f"Estimated DOA: {doa:.2f} degrees (true: {angle})")

    plt.plot(cc)
    plt.title('GCC-PHAT Cross-correlation')
    plt.show()
