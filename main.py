from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Your audio file:
AUDIO_FILE = "simulated_stereo.wav"
label = predict(AUDIO_FILE)
print("Detected Sound:", label)

if label == "Gunshot":
    # estimate the angle
    angle = estimate_doa(AUDIO_FILE)
    print("Estimated DoA:", round(angle, 2), "degrees")

    
   
    true_angle = 45.0  

   
    mae = mean_absolute_error([true_angle], [angle])
    rmse = np.sqrt(mean_squared_error([true_angle], [angle]))

    print(f"MAE: {mae:.2f} degrees")
    print(f"RMSE: {rmse:.2f} degrees")

    # Optional: show +/- 10 degree precision claim
    if abs(angle - true_angle) <= 10:
        print("Within ±10° precision ✅")
    else:
        print("Outside ±10° precision ❌")
