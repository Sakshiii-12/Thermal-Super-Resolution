import os
import shutil

SRC = "data/raw/LLVIP"
DST = "data/processed"

rgb_dst = os.path.join(DST, "rgb")
thermal_dst = os.path.join(DST, "thermal_hr")

os.makedirs(rgb_dst, exist_ok=True)
os.makedirs(thermal_dst, exist_ok=True)

rgb_src = os.path.join(SRC, "visible", "train")
thermal_src = os.path.join(SRC, "infrared", "train")

files = sorted(os.listdir(rgb_src))[:400]  # LIMIT FOR HACKATHON

for f in files:
    shutil.copy(os.path.join(rgb_src, f), os.path.join(rgb_dst, f))
    shutil.copy(os.path.join(thermal_src, f), os.path.join(thermal_dst, f))

print("✅ LLVIP processed data prepared successfully")
print("RGB images:", len(os.listdir(rgb_dst)))
print("Thermal images:", len(os.listdir(thermal_dst)))
