from pathlib import Path

import cv2

import os
 
def crop_and_save(image_path, label_path, output_dir):

    img = cv2.imread(str(image_path))

    if img is None:

        print(f"Could not read {image_path}")

        return
 
    height, width = img.shape[:2]
 
    with open(label_path, 'r', encoding='utf-8') as f:

        lines = f.readlines()
 
    filename = image_path.stem

    count = 0
 
    for line in lines:

        parts = line.strip().split()

        if len(parts) != 5:

            continue
 
        _, x_center, y_center, w, h = map(float, parts)

        x_center *= width

        y_center *= height

        w *= width

        h *= height
 
        x1 = int(x_center - w / 2)

        y1 = int(y_center - h / 2)

        x2 = int(x_center + w / 2)

        y2 = int(y_center + h / 2)
 
        x1, y1 = max(0, x1), max(0, y1)

        x2, y2 = min(width - 1, x2), min(height - 1, y2)
 
        crop = img[y1:y2, x1:x2]

        if crop.size == 0:

            continue
 
        output_dir.mkdir(parents=True, exist_ok=True)

        crop_filename = f"{filename}_{count}.jpg"

        cv2.imwrite(str(output_dir / crop_filename), crop)

        count += 1
 
def process_dataset(input_root, output_root):

    input_root = Path(input_root)

    output_root = Path(output_root)
 
    for tank_folder in input_root.iterdir():

        if not tank_folder.is_dir():

            continue
 
        for file in tank_folder.iterdir():

            if file.suffix.lower() in ['.jpg', '.png', '.jpeg']:

                label_path = file.with_suffix('.txt')

                if label_path.exists():

                    output_dir = output_root / tank_folder.name

                    crop_and_save(file, label_path, output_dir)
 
if __name__ == "__main__":

    input_dataset = "C:/Users/Yousef/Desktop/Year4Sem2/Graduation projec/Final_dataset/Tanks"

    output_dataset = "C:/Users/Yousef/Desktop/Year4Sem2/Graduation projec/CroppedTanks"

    process_dataset(input_dataset, output_dataset)

    print("✅ Done cropping all tanks.")

 