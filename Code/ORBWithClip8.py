import os
import cv2
import pickle
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizerFast, CLIPFeatureExtractor

# CONFIG
DATASET_PATH = "C:/Users/Yousef/Desktop/Year4Sem2/Graduation projec/Cropped-Augmented"
IMAGE_PATH = "C:/Users/Yousef/Desktop/Year4Sem2/Graduation projec/Images for testing/Centurion/OIP.jpeg"
FEATURE_DB_PATH = "orb_features.pkl"
MODEL_NAME = "openai/clip-vit-base-patch32"
TOP_K = 5


def precompute_orb_features(dataset_path, save_path):
    if os.path.exists(save_path):
        print("[INFO] ORB features already computed.")
        return

    print("[INFO] Computing ORB features...")
    orb = cv2.ORB_create(nfeatures=500)
    feature_db = {}

    for class_name in os.listdir(dataset_path):
        class_folder = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_folder):
            continue

        for fname in os.listdir(class_folder):
            fpath = os.path.join(class_folder, fname)
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            keypoints, descriptors = orb.detectAndCompute(img, None)
            feature_db[fpath] = {
                "class": class_name,
                "descriptors": descriptors
            }

    with open(save_path, "wb") as f:
        pickle.dump(feature_db, f)

    print(f"[INFO] Saved ORB features to {save_path}")


def match_orb_features(image_path, feature_db_path, top_k=5):
    print("[INFO] Matching using ORB...")
    orb = cv2.ORB_create(nfeatures=500)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    with open(feature_db_path, "rb") as f:
        feature_db = pickle.load(f)

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    keypoints, descriptors = orb.detectAndCompute(img, None)

    matches_list = []
    for path, data in feature_db.items():
        if data["descriptors"] is None or descriptors is None:
            continue
        matches = bf.match(descriptors, data["descriptors"])
        if len(matches) == 0:
            continue
        score = sum([m.distance for m in matches]) / len(matches)
        matches_list.append((path, data["class"], score))

    top_matches = sorted(matches_list, key=lambda x: x[2])[:top_k]

    print(f"\n[INFO] Top-{top_k} ORB Matches:")
    for i, (path, cls, score) in enumerate(top_matches):
        print(f"  {i+1}. Class: {cls}, Image: {os.path.basename(path)}, Score: {score:.2f}")

    return top_matches


def clip_classify_from_top_matches(image_path, top_matches, model_name):
    print("\n[INFO] Running CLIP on top matches...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = CLIPTokenizerFast.from_pretrained(model_name)
    feature_extractor = CLIPFeatureExtractor.from_pretrained(model_name)
    processor = CLIPProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)
    model = CLIPModel.from_pretrained(model_name).to(device)

    tank_names = [match[1] for match in top_matches]
    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        text=tank_names,
        images=image,
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)

    predicted_index = probs.argmax().item()
    predicted_tank = tank_names[predicted_index]
    confidence = probs[0, predicted_index].item()

    print(f"\n✅ Predicted tank: {predicted_tank} (Confidence: {confidence:.2%})")
    return predicted_tank, confidence


# Run the full pipeline
if __name__ == "__main__":
    precompute_orb_features(DATASET_PATH, FEATURE_DB_PATH)
    top_matches = match_orb_features(IMAGE_PATH, FEATURE_DB_PATH, TOP_K)
    clip_classify_from_top_matches(IMAGE_PATH, top_matches, MODEL_NAME)
