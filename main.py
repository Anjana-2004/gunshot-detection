from gunshot_detection.predict import predict
from doa_estimation.gcc_phat_doa import estimate_doa

AUDIO_FILE = "simulated_stereo.wav"
label = predict(AUDIO_FILE)
print("Detected Sound:", label)

if label == "Gunshot":
    angle = estimate_doa(AUDIO_FILE)
    print("Estimated DoA:", round(angle, 2), "degrees")