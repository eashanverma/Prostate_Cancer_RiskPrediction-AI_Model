import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

base_id = 'slide001_core049'
image_path = f'./Train Imgs/{base_id}.jpg'
map_dirs = [f'./Maps{i}_T/' for i in range(1, 7)]

image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image not found: {image_path}")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

color_map = {
    0: [0, 0, 0],       # background
    1: [0, 255, 0],     # benign
    2: [255, 255, 0],   # Gleason 3
    3: [255, 165, 0],   # Gleason 4
    4: [255, 0, 0],     # Gleason 5
}

plt.figure(figsize=(20, 10))
plt.subplot(2, 4, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis('off')

for i, map_dir in enumerate(map_dirs, 1):
    mask_path = os.path.join(map_dir, f"{base_id}_classimg_nonconvex.png")
    if not os.path.exists(mask_path):
        print(f"❌ Mask not found: {mask_path}")
        continue
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"⚠️ Cannot load: {mask_path}")
        continue

    mask_rgb = np.zeros_like(image)
    for k, v in color_map.items():
        mask_rgb[mask == k] = v

    overlay = cv2.addWeighted(image, 0.7, mask_rgb, 0.3, 0)
    plt.subplot(2, 4, i+1)
    plt.imshow(overlay)
    plt.title(f'Map {i}')
    plt.axis('off')

plt.tight_layout()
plt.show()