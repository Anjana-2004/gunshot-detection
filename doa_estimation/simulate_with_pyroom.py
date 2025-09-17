import pyroomacoustics as pra
import numpy as np
from gcc_phat_doa import estimate_doa

def simulate_room_and_estimate_doa(angle=45, room_dim=[5, 4], fs=16000, mic_distance=0.08):
    # Create shoebox room
    room = pra.ShoeBox(room_dim, fs=fs, max_order=15)
    mic_locs = np.c_[
        [2.5-mic_distance/2, 2],  # Mic 1
        [2.5+mic_distance/2, 2]   # Mic 2
    ]  # shape: 2x2 (x,y), so transpose

    room.add_microphone_array(mic_locs.T)

    # Place source at specific DOA
    radius = 2
    src_x = 2.5 + radius * np.cos(np.deg2rad(angle))
    src_y = 2 + radius * np.sin(np.deg2rad(angle))
    room.add_source([src_x, src_y])

    # Run simulation
    room.compute_rir()
    signal = np.random.randn(int(fs))  # white noise
    room.add_source([src_x, src_y], signal=signal)
    room.simulate()

    mic_signals = room.mic_array.signals
    doa, cc = estimate_doa(mic_signals, mic_distance, fs)
    return doa

if __name__ == "__main__":
    est = simulate_room_and_estimate_doa(45)
    print(f"Estimated DOA in simulated room: {est:.2f} degrees")
