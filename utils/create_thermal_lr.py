import cv2
import os

HR_DIR = "data/processed/thermal_hr"
LR_DIR = "data/processed/thermal_lr"

os.makedirs(LR_DIR, exist_ok=True)

for fname in os.listdir(HR_DIR):
    hr_path = os.path.join(HR_DIR, fname)
    lr_path = os.path.join(LR_DIR, fname)

    img = cv2.imread(hr_path, cv2.IMREAD_GRAYSCALE)

    # Step 1: Downsample to simulate cheap sensor
    lr = cv2.resize(img, (160, 120), interpolation=cv2.INTER_AREA)

    # Step 2: Upsample back (blocky effect)
    lr = cv2.resize(lr, img.shape[::-1], interpolation=cv2.INTER_NEAREST)

    cv2.imwrite(lr_path, lr)

print("✅ Low-resolution thermal images created")
print("Count:", len(os.listdir(LR_DIR)))
