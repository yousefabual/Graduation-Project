import cv2
import numpy as np
import os
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import tensorflow as tf

def extract_features_orb(img):
    orb = cv2.ORB_create()
    kp, des = orb.detectAndCompute(img, None)
    return kp, des

def extract_features_sift(img):
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    return kp, des

def extract_features_brisk(img):
    brisk = cv2.BRISK_create()
    kp, des = brisk.detectAndCompute(img, None)
    return kp, des

def match_features(des1, des2, method='bf', norm=cv2.NORM_HAMMING):
    if des1 is None or des2 is None:
        return 0
    bf = cv2.BFMatcher(norm, crossCheck=True)
    matches = bf.match(des1, des2)
    return len(matches)

def extract_resnet_feature(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x, verbose=0)
    return features.flatten()

def load_dataset_features(tanks_path, resnet_model):
    dataset_features = {'orb': {}, 'sift': {}, 'brisk': {}, 'resnet': {}}
    for tank in os.listdir(tanks_path):
        tank_path = os.path.join(tanks_path, tank)
        if not os.path.isdir(tank_path):
            continue

        dataset_features['orb'][tank] = []
        dataset_features['sift'][tank] = []
        dataset_features['brisk'][tank] = []
        dataset_features['resnet'][tank] = []

        for img_name in os.listdir(tank_path):
            img_path = os.path.join(tank_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            _, orb_des = extract_features_orb(img)
            _, sift_des = extract_features_sift(img)
            _, brisk_des = extract_features_brisk(img)
            resnet_feat = extract_resnet_feature(img_path, resnet_model)

            dataset_features['orb'][tank].append(orb_des)
            dataset_features['sift'][tank].append(sift_des)
            dataset_features['brisk'][tank].append(brisk_des)
            dataset_features['resnet'][tank].append(resnet_feat)
    return dataset_features

def predict_tank(img_path, dataset_features, resnet_model):
    input_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    orb_kp, orb_des = extract_features_orb(input_img)
    sift_kp, sift_des = extract_features_sift(input_img)
    brisk_kp, brisk_des = extract_features_brisk(input_img)
    resnet_feat = extract_resnet_feature(img_path, resnet_model)

    predictions = []

    for method, (des, norm) in {'orb': (orb_des, cv2.NORM_HAMMING), 'sift': (sift_des, cv2.NORM_L2), 'brisk': (brisk_des, cv2.NORM_HAMMING)}.items():
        max_matches = 0
        best_tank = None
        for tank, desc_list in dataset_features[method].items():
            for db_des in desc_list:
                matches = match_features(des, db_des, norm=norm)
                if matches > max_matches:
                    max_matches = matches
                    best_tank = tank
        predictions.append(best_tank)
        print(f"🔍 {method.upper()} + BF ({'Hamming' if norm==cv2.NORM_HAMMING else 'L2'}):\t{best_tank}")

    # ResNet prediction
    max_sim = -1
    best_tank_resnet = None
    for tank, feats in dataset_features['resnet'].items():
        for feat in feats:
            sim = cosine_similarity([resnet_feat], [feat])[0][0]
            if sim > max_sim:
                max_sim = sim
                best_tank_resnet = tank
    predictions.append(best_tank_resnet)
    print(f"🧠 ResNet50 Embedding:\t{best_tank_resnet}")

    # Voting
    vote_result = Counter(predictions).most_common(1)[0]
    print(f"\n✅ Final Result (Voting):\t{vote_result[0]}\n🗳️ Votes: {dict(Counter(predictions))}")
    return vote_result[0]


tanks_folder ="C:/Users/Yousef/Desktop/Year4Sem2/Graduation projec/Cropped-Augmented" # your root folder containing subfolders for each tank
input_image = "C:/Users/Yousef/Desktop/Year4Sem2/Graduation projec/Images for testing/M60/60.jpg"

resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
dataset_feats = load_dataset_features(tanks_folder, resnet_model)
result = predict_tank(input_image, dataset_feats, resnet_model)
