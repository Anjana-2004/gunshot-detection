import librosa
import numpy as np
from model import load_model
from train import extract_features

def predict_file(audio_file, model_path='gunshot_model.pkl'):
    model = load_model(model_path)
    features = extract_features(audio_file)
    result = model.predict([features])
    prob = model.predict_proba([features])[result]
    print(f'Prediction: {"gunshot" if result else "other"} (confidence: {100*prob:.1f}%)')
    return result

if __name__ == '__main__':
    import sys
    audio_path = sys.argv[1]
    predict_file(audio_path)
