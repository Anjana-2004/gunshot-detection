from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import joblib

def get_model(model_type='svm'):
    if model_type == 'svm':
        return SVC(kernel='rbf', probability=True)
    elif model_type == 'rf':
        return RandomForestClassifier(n_estimators=100)
    else:
        raise ValueError("Unknown model_type")

def save_model(model, filename):
    joblib.dump(model, filename)

def load_model(filename):
    return joblib.load(filename)
