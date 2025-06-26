# 2_clip_faiss_inference.ipynb

import torch
import faiss
import pickle
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load index and labels
index = faiss.read_index("clip_faiss.index")
with open("clip_labels.pkl", "rb") as f:
    labels = pickle.load(f)

# Prediction function
def predict(image_path, k=1):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        features = model.get_image_features(**inputs)
        features /= features.norm(p=2, dim=-1, keepdim=True)
    features_np = features.cpu().numpy()
    D, I = index.search(features_np, k)
    for i in range(k):
        print(f"Top {i+1}: {labels[I[0][i]]} (Distance: {D[0][i]:.4f})")
    return labels[I[0][0]]

# Example usage
test_img = "C:/Users/Yousef/Desktop/Year4Sem2/Graduation projec/Images for testing/M1/M.jpg"
predict(test_img)