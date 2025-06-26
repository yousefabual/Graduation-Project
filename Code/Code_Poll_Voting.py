import cv2
import numpy as np
import torch
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import os
from torch import nn
from torchvision import transforms
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, EfficientNet_B0_Weights
from collections import Counter

# ORB feature extraction
def extract_orb(image):
    orb = cv2.ORB_create()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return descriptors

# HOG feature extraction
def extract_hog(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray, (256, 256))
    hog = cv2.HOGDescriptor()
    h = hog.compute(resized_image)
    return h

# ResNet feature extraction
def extract_resnet(image):
    resnet_model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    resnet_model.eval()
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        features = resnet_model(input_batch)
    return features.flatten().numpy()

# EfficientNet feature extraction
def extract_efficientnet(image):
    efficientnet_model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    efficientnet_model.eval()
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        features = efficientnet_model(input_batch)
    return features.flatten().numpy()

# Cosine Similarity match
def match_cosine(input_desc, db_desc):
    return cosine_similarity(input_desc.reshape(1, -1), db_desc.reshape(1, -1))[0][0]

# ORB match using Hamming distance
def match_orb(input_desc, db_desc):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(input_desc, db_desc)
    return matches

# Load the database of features from files
def load_database_features(database_path, feature_method):
    database = {}
    for label in os.listdir(database_path):
        label_path = os.path.join(database_path, label)
        if os.path.isdir(label_path):
            descriptors_list = []
            for img_name in os.listdir(label_path):
                img_path = os.path.join(label_path, img_name)
                image = cv2.imread(img_path)
                if feature_method == extract_orb:
                    descriptors = feature_method(image)
                    if descriptors is not None:
                        descriptors_list.append(descriptors)
                else:
                    descriptors = feature_method(image)
                    descriptors_list.append(descriptors)
            database[label] = descriptors_list
    return database

# Find the best match from the database
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

    return best_label, best_score

# Poll Voting mechanism
def poll_voting(input_image_path, database_path, feature_methods, iterations=5):
    votes = []
    for _ in range(iterations):
        method_votes = []
        for method_name, (feature_fn, match_fn) in feature_methods.items():
            match_label, match_score = find_best_match(input_image_path, database_path, feature_fn, match_fn)
            method_votes.append(match_label)
        # Tally the votes for this iteration
        most_common_vote = Counter(method_votes).most_common(1)[0][0]  # Most common label
        votes.append(most_common_vote)
    # Final result is the most common vote across all iterations
    final_vote = Counter(votes).most_common(1)[0][0]
    return final_vote

# Main execution logic
if __name__ == "__main__":
    input_image = "C:/Users/Yousef/Desktop/Year4Sem2/Graduation projec/Images for testing/M60/60.jpg"
    database_dir = "C:/Users/Yousef/Desktop/Year4Sem2/Graduation projec/CroppedTanks"

    feature_methods = {
        'ORB': (extract_orb, match_orb),
        'HOG': (extract_hog, match_cosine),
        'ResNet': (extract_resnet, match_cosine),
        'EfficientNet': (extract_efficientnet, match_cosine)
    }

    # Execute poll voting
    best_match_label = poll_voting(input_image, database_dir, feature_methods, iterations=5)
    
    print(f"The predicted tank name (after poll voting): {best_match_label}")
