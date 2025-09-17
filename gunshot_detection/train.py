import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from model import get_model, save_model

def extract_features(path, n_mfcc=20):
    y, sr = librosa.load(path, sr=16000, duration=2.5)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    return np.concatenate([np.mean(mfcc,axis=1), np.mean(chroma,axis=1), np.mean(spec_contrast,axis=1)])

def load_dataset(data_root):
    # Expects directories data_root/gunshot and data_root/other
    files, labels = [], []
    for label in ['gunshot', 'other']:
        folder = os.path.join(data_root, label)
        for fname in os.listdir(folder):
            if fname.endswith('.wav'):
                files.append(os.path.join(folder, fname))
                labels.append(1 if label == 'gunshot' else 0)
    X = [extract_features(f) for f in files]
    return np.stack(X), np.array(labels)

def main():
    X, y = load_dataset('data/UrbanSound8K')
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y)
    model = get_model('svm')
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    print(classification_report(y_te, y_pred))
    save_model(model, 'gunshot_model.pkl')

if __name__ == '__main__':
    main()
