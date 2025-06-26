import os
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import faiss
from collections import Counter

# Paths
DATASET_PATH = "C:/Users/Yousef/Desktop/Year4Sem2/Graduation projec/Cropped-Augmented"
IMAGE_PATH = "C:/Users/Yousef/Desktop/Year4Sem2/Graduation projec/Images for testing/Merkava/A.jpg"
FEATURES_FILE = "clip_faiss_features.npz"
TOP_K = 10

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP
print("[INFO] Loading CLIP...")
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)

def extract_features(image: Image.Image):
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
    return outputs.cpu().numpy()[0]

# Step 1: Extract & save features
if not os.path.exists(FEATURES_FILE):
    print("[INFO] Extracting and saving features...")
    all_features, all_labels, all_paths = [], [], []

    for class_name in tqdm(sorted(os.listdir(DATASET_PATH)), desc="Classes"):
        class_path = os.path.join(DATASET_PATH, class_name)
        if not os.path.isdir(class_path):
            continue
        for fname in os.listdir(class_path):
            fpath = os.path.join(class_path, fname)
            try:
                image = Image.open(fpath).convert("RGB")
                feat = extract_features(image)
                all_features.append(feat)
                all_labels.append(class_name)
                all_paths.append(fpath)
            except Exception as e:
                print(f"[WARNING] Failed to process {fpath}: {e}")

    all_features = np.vstack(all_features).astype("float32")
    all_labels = np.array(all_labels)
    all_paths = np.array(all_paths)

    np.savez(FEATURES_FILE,
             features=all_features,
             labels=all_labels,
             paths=all_paths)

    print(f"[INFO] Saved {len(all_labels)} features to {FEATURES_FILE}")
else:
    print("[INFO] Features already extracted. Loading...")

# Step 2: Load features
data = np.load(FEATURES_FILE)
features = data["features"].astype("float32")
labels = data["labels"]
paths = data["paths"]

# Step 3: Create FAISS index
print("[INFO] Creating FAISS index...")
index = faiss.IndexFlatL2(features.shape[1])
index.add(features)

# Step 4: Extract features for the query image
print("[INFO] Extracting features for query image...")
query_image = Image.open(IMAGE_PATH).convert("RGB")
query_feat = extract_features(query_image).astype("float32").reshape(1, -1)

# Step 5: Search
print("[INFO] Searching FAISS index...")
distances, indices = index.search(query_feat, TOP_K)

# Step 6: Poll Voting for Prediction
votes = [labels[idx] for idx in indices[0]]
vote_counter = Counter(votes)

# Get the most common vote (tank class)
predicted_tank, vote_count = vote_counter.most_common(1)[0]
confidence = vote_count / TOP_K  # Confidence based on vote proportion

print(f"\nPredicted Tank: {predicted_tank} (Confidence: {confidence * 100:.2f}%)")

# Display top K matches
print(f"\nTop {TOP_K} Matches:")
for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), start=1):
    print(f"{rank}. {labels[idx]} ({paths[idx]}) | Distance: {dist:.4f}")
