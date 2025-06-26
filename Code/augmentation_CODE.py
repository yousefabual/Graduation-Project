import cv2
import numpy as np
import os
from pathlib import Path

# ----------- CONFIG -----------
SOURCE_DIR = "C:/Users/Yousef/Desktop/Year4Sem2/Graduation projec/CroppedTanks"
DEST_DIR = "C:/Users/Yousef/Desktop/Year4Sem2/Graduation projec/Cropped-Augmented"

# ----------- AUGMENTATION FUNCTION -----------
def augment_image(img):
    # Random flip
    img_flip = cv2.flip(img, 1)  # Horizontal flip

    # Random rotation
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle=15, scale=1.0)
    img_rotated = cv2.warpAffine(img, M, (w, h))

    # Brightness adjustment
    img_bright = cv2.convertScaleAbs(img, alpha=1.1, beta=30)

    return [img_flip, img_rotated, img_bright]

# ----------- MAIN SCRIPT -----------
def augment_dataset(source_dir, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)

    for tank_name in os.listdir(source_dir):
        tank_folder = os.path.join(source_dir, tank_name)
        if not os.path.isdir(tank_folder):
            continue

        dest_tank_folder = os.path.join(dest_dir, tank_name)
        os.makedirs(dest_tank_folder, exist_ok=True)

        for filename in os.listdir(tank_folder):
            file_path = os.path.join(tank_folder, filename)
            img = cv2.imread(file_path)

            if img is None:
                print(f"[WARNING] Couldn't load image: {file_path}")
                continue

            # Save original image
            cv2.imwrite(os.path.join(dest_tank_folder, filename), img)

            # Generate augmentations
            aug_imgs = augment_image(img)
            base_name = Path(filename).stem
            ext = Path(filename).suffix

            for idx, aug_img in enumerate(aug_imgs):
                aug_name = f"{base_name}_aug{idx+1}{ext}"
                cv2.imwrite(os.path.join(dest_tank_folder, aug_name), aug_img)

            print(f"[INFO] Augmented {filename} → {len(aug_imgs)} files created")

    print("\n✅ Augmentation completed. Output saved to:", dest_dir)

# ----------- RUN -----------
if __name__ == '__main__':
    augment_dataset(SOURCE_DIR, DEST_DIR)
