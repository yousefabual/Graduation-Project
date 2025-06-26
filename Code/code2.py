import cv2
import os
import numpy as np
from skimage.feature import hog
from skimage import color
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import models, transforms
import torch
from PIL import Image

# ORB feature extraction
def extract_orb(image):
    orb = cv2.ORB_create(nfeatures=1000)
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return descriptors

def match_orb(desc1, desc2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1, desc2)
    return sorted(matches, key=lambda x: x.distance)

# HOG feature extraction
def extract_hog(image):
    image = cv2.resize(image, (128, 128))
    gray_image = color.rgb2gray(image)
    features, _ = hog(gray_image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return features.reshape(1, -1)

def match_cosine(desc1, desc2):
    return cosine_similarity(desc1, desc2)[0][0]

# ResNet feature extraction
def extract_resnet(image):
    model = models.resnet18(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    img_tensor = preprocess(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))).unsqueeze(0)
    with torch.no_grad():
        features = model(img_tensor).squeeze().numpy()
    return features.reshape(1, -1)

# EfficientNet feature extraction
def extract_efficientnet(image):
    model = models.efficientnet_b0(pretrained=True)
    model.classifier = torch.nn.Identity()
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    img_tensor = preprocess(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))).unsqueeze(0)
    with torch.no_grad():
        features = model(img_tensor).squeeze().numpy()
    return features.reshape(1, -1)

# Framework to test all methods
def load_database_features(database_path, feature_method):
    database = {}
    for label in os.listdir(database_path):
        class_dir = os.path.join(database_path, label)
        if os.path.isdir(class_dir):
            descriptors = []
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    desc = feature_method(img)
                    if desc is not None:
                        descriptors.append(desc)
            database[label] = descriptors
    return database

def find_best_match(input_image_path, database_path, feature_method, match_method):
    input_img = cv2.imread(input_image_path)
    input_desc = feature_method(input_img)

    database = load_database_features(database_path, feature_method)

    best_score = -1 if match_method == match_cosine else float('inf')
    best_label = None

    for label, descriptors_list in database.items():
        scores = []
        for db_desc in descriptors_list:
            if match_method == match_orb:
                matches = match_method(input_desc, db_desc)
                distance = sum(m.distance for m in matches) / len(matches) if matches else float('inf')
                scores.append(distance)
            else:
                score = match_method(input_desc, db_desc)
                scores.append(score)

        avg_score = np.mean(scores)

        if (match_method == match_cosine and avg_score > best_score) or (match_method == match_orb and avg_score < best_score):
            best_score = avg_score
            best_label = label

    return best_label

if __name__ == "__main__":
    input_image = "C:/Users/Yousef/Desktop/Year4Sem2/Graduation projec/Images for testing/M60/M60.jpg"
    database_dir = "C:/Users/Yousef/Desktop/Year4Sem2/Graduation projec/CroppedTanks"

    feature_methods = {
        'ORB': (extract_orb, match_orb),
        'HOG': (extract_hog, match_cosine),
        'ResNet': (extract_resnet, match_cosine),
        'EfficientNet': (extract_efficientnet, match_cosine)
    }

    for method_name, (feature_fn, match_fn) in feature_methods.items():
        match = find_best_match(input_image, database_dir, feature_fn, match_fn)
        print(f"Best match using {method_name}: {match}")