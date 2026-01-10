import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import torch
import numpy as np
from models.backbone.thermal_sr import ThermalSR

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model = ThermalSR().to(device)
model.load_state_dict(torch.load("thermal_sr.pth", map_location=device))
model.eval()

# Pick one sample
fname = os.listdir("data/processed/thermal_lr")[0]

# Load images
lr = cv2.imread(f"data/processed/thermal_lr/{fname}", 0) / 255.0
hr = cv2.imread(f"data/processed/thermal_hr/{fname}", 0) / 255.0
edge = cv2.imread(f"data/processed/rgb_edges/{fname}", 0) / 255.0

# Prepare tensors
lr_t = torch.tensor(lr).unsqueeze(0).unsqueeze(0).to(device).float()
edge_t = torch.tensor(edge).unsqueeze(0).unsqueeze(0).to(device).float()

with torch.no_grad():
    pred = model(lr_t, edge_t)

pred = pred.squeeze().cpu().numpy()

# Normalize for display
def norm(x):
    x = (x - x.min()) / (x.max() - x.min() + 1e-6)
    return (x * 255).astype(np.uint8)

lr_disp = norm(lr)
pred_disp = norm(pred)
hr_disp = norm(hr)

# Concatenate horizontally
combined = cv2.hconcat([lr_disp, pred_disp, hr_disp])

cv2.putText(combined, "Thermal LR", (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
cv2.putText(combined, "Enhanced Thermal", (lr_disp.shape[1] + 20, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
cv2.putText(combined, "Thermal HR (GT)", (2 * lr_disp.shape[1] + 20, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)

cv2.imshow("Thermal Super-Resolution Demo", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
