import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import Counter
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Paths
DATASET_PATH = "C:/Users/Yousef/Desktop/Year4Sem2/Graduation projec/Cropped-Augmented"
IMAGE_PATH = "C:/Users/Yousef/Desktop/Year4Sem2/Graduation projec/Images for testing/Russia/T34.jpg"
FEATURES_FILE = "orb_features.npz"
TOP_K = 10

# Load FLAN-T5 model
print("[INFO] Loading FLAN-T5...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "google/flan-t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

# ORB extractor
orb = cv2.ORB_create(nfeatures=500)

# Step 1: Extract & Save Features
if not os.path.exists(FEATURES_FILE):
    print("[INFO] Extracting ORB features and saving...")

    all_descriptors, all_labels, all_paths = [], [], []

    for class_name in tqdm(sorted(os.listdir(DATASET_PATH)), desc="Classes"):
        class_path = os.path.join(DATASET_PATH, class_name)
        if not os.path.isdir(class_path):
            continue

        for fname in os.listdir(class_path):
            fpath = os.path.join(class_path, fname)
            image = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            keypoints, descriptors = orb.detectAndCompute(image, None)
            if descriptors is not None:
                all_descriptors.append(descriptors)
                all_labels.append(class_name)
                all_paths.append(fpath)

    np.savez(FEATURES_FILE,
             descriptors=np.array(all_descriptors, dtype=object),
             labels=np.array(all_labels),
             paths=np.array(all_paths))

    print(f"[INFO] Saved {len(all_labels)} features to {FEATURES_FILE}")
else:
    print("[INFO] Loading saved ORB features...")

# Step 2: Load saved features
data = np.load(FEATURES_FILE, allow_pickle=True)
descriptors = data["descriptors"]
labels = data["labels"]
paths = data["paths"]
database = list(zip(descriptors, labels, paths))

# Step 3: ORB feature matching for query image
print("[INFO] Processing query image with ORB...")
query_img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
query_kp, query_des = orb.detectAndCompute(query_img, None)

# Brute-force matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

print("[INFO] Matching using ORB...")
match_scores = []

for des, label, path in database:
    if des is None or len(des) == 0:
        continue
    matches = bf.match(query_des, des)
    distance = sum(m.distance for m in matches) / (len(matches) + 1e-5)
    match_scores.append((distance, label, path))

# Sort by lowest distance
match_scores.sort(key=lambda x: x[0])
top_matches = match_scores[:TOP_K]

# Step 4: Use FLAN-T5 to describe the matched images
print("\n[INFO] Generating labels with FLAN-T5...")
predicted_labels = []

for _, _, path in top_matches:
    prompt = f"What is in this image: {os.path.basename(path).replace('_', ' ')}"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    output_ids = model.generate(input_ids, max_length=10)
    label = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    predicted_labels.append(label.strip())

# Step 5: Voting
print("\n[INFO] Voting result:")
label_counts = Counter(predicted_labels)
for label, count in label_counts.most_common():
    print(f"{label}: {count} votes")

predicted = label_counts.most_common(1)[0][0]
print(f"\n✅ Predicted Class (by voting): {predicted}")