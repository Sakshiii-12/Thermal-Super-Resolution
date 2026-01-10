import cv2
import os

RGB_DIR = "data/processed/rgb"
EDGE_DIR = "data/processed/rgb_edges"

os.makedirs(EDGE_DIR, exist_ok=True)

for fname in os.listdir(RGB_DIR):
    rgb_path = os.path.join(RGB_DIR, fname)
    edge_path = os.path.join(EDGE_DIR, fname)

    img = cv2.imread(rgb_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Canny edge detection (fast + effective)
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)

    cv2.imwrite(edge_path, edges)

print("✅ RGB edge maps created")
print("Count:", len(os.listdir(EDGE_DIR)))
