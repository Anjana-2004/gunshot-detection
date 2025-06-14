import torch
from model import GunshotCNN
from utils import extract_mfcc

def predict(file_path):
    model = GunshotCNN()
    model.load_state_dict(torch.load("gunshot_cnn.pth"))
    model.eval()
    feat = extract_mfcc(file_path)
    with torch.no_grad():
        output = model(feat)
        prediction = torch.argmax(output).item()
    return "Gunshot" if prediction == 1 else "Not Gunshot"