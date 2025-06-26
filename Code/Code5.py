import os
import cv2
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image

# Setup
CACHE_DIR = 'cached_features'
os.makedirs(CACHE_DIR, exist_ok=True)

def extract_resnet_feature(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x, verbose=0)
    return features.flatten()

def load_or_cache_resnet_features(tanks_path, model):
    cache_path = os.path.join(CACHE_DIR, 'resnet_features_only.pkl')
    
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            print("✅ Loaded ResNet features from cache.")
            return pickle.load(f)
    
    dataset_features = {}
    for tank in os.listdir(tanks_path):
        tank_path = os.path.join(tanks_path, tank)
        if not os.path.isdir(tank_path):
            continue
        
        dataset_features[tank] = []
        for img_name in os.listdir(tank_path):
            img_path = os.path.join(tank_path, img_name)
            feat = extract_resnet_feature(img_path, model)
            dataset_features[tank].append(feat)
    
    with open(cache_path, 'wb') as f:
        pickle.dump(dataset_features, f)
    print("📦 ResNet features cached.")
    return dataset_features

def predict_tank_resnet(img_path, dataset_features, model):
    input_feat = extract_resnet_feature(img_path, model)
    max_sim = -1
    best_tank = None
    
    for tank, feats in dataset_features.items():
        for feat in feats:
            sim = cosine_similarity([input_feat], [feat])[0][0]
            if sim > max_sim:
                max_sim = sim
                best_tank = tank
    
    print(f"🧠 Predicted Tank (ResNet): {best_tank} | Similarity: {max_sim:.4f}")
    return best_tank

# === Run ===
tanks_folder = "C:/Users/Yousef/Desktop/Year4Sem2/Graduation projec/Cropped-Augmented"
input_image = "C:/Users/Yousef/Desktop/Year4Sem2/Graduation projec/Images for testing/Challenger/Capture.PNG"

resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
dataset_feats = load_or_cache_resnet_features(tanks_folder, resnet_model)
result = predict_tank_resnet(input_image, dataset_feats, resnet_model)
