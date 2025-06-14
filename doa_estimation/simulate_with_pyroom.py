import pyroomacoustics as pra
import numpy as np
import soundfile as sf

room_dim = [10, 10]
source_loc = [2, 3]
mic_locs = np.array([[4, 4], [4.2, 4]])

room = pra.ShoeBox(room_dim, fs=16000, max_order=1)
room.add_microphone_array(pra.MicrophoneArray(mic_locs.T, room.fs))
room.add_source(source_loc, signal=sf.read("gunshot.wav")[0])
room.simulate()

sf.write("simulated_stereo.wav", room.mic_array.signals.T, 16000)