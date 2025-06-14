import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from model import GunshotCNN
from utils import extract_mfcc

# Load data
data_dir = "data/UrbanSound8K/audio"
meta_file = "data/UrbanSound8K/metadata/UrbanSound8K.csv"

meta = pd.read_csv(meta_file)
meta = meta[meta["class"].isin(["gun_shot", "dog_bark"])]
X, y = [], []

label_map = {"gun_shot": 1, "dog_bark": 0}
for _, row in meta.iterrows():
    file_path = os.path.join(data_dir, f"fold{row.fold}", row.slice_file_name)
    try:
        feat = extract_mfcc(file_path)
        X.append(feat)
        y.append(label_map[row["class"]])
    except:
        continue

X = torch.cat(X)
y = torch.tensor(y)

model = GunshotCNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(5):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

torch.save(model.state_dict(), "gunshot_cnn.pth")