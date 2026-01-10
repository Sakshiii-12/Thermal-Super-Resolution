import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import torch
from tqdm import tqdm
from models.backbone.thermal_sr import ThermalSR

device = "cuda" if torch.cuda.is_available() else "cpu"

model = ThermalSR().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.L1Loss()

RGB_DIR = "data/processed/rgb_edges"
LR_DIR = "data/processed/thermal_lr"
HR_DIR = "data/processed/thermal_hr"

files = sorted(os.listdir(HR_DIR))

for epoch in range(3):  # ONLY 3 EPOCHS
    epoch_loss = 0

    for f in tqdm(files, desc=f"Epoch {epoch+1}"):
        lr = cv2.imread(os.path.join(LR_DIR, f), 0) / 255.0
        hr = cv2.imread(os.path.join(HR_DIR, f), 0) / 255.0
        edge = cv2.imread(os.path.join(RGB_DIR, f), 0) / 255.0

        lr = torch.tensor(lr).unsqueeze(0).unsqueeze(0).to(device).float()
        hr = torch.tensor(hr).unsqueeze(0).unsqueeze(0).to(device).float()
        edge = torch.tensor(edge).unsqueeze(0).unsqueeze(0).to(device).float()

        pred = model(lr, edge)
        loss = criterion(pred, hr)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {epoch_loss/len(files):.4f}")

torch.save(model.state_dict(), "thermal_sr.pth")
print("✅ Training complete. Model saved.")
