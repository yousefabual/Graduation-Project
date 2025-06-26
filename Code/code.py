import cv2
import os
import numpy as np
 
def extract_orb_features(image):
    orb = cv2.ORB_create(nfeatures=1000)
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors
 
def match_features(desc1, desc2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches
 
def load_database_features(database_path):
    database = {}
    for tank_type in os.listdir(database_path):
        tank_dir = os.path.join(database_path, tank_type)
        if os.path.isdir(tank_dir):
            image_descriptors = []
            for img_name in os.listdir(tank_dir):
                img_path = os.path.join(tank_dir, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    _, desc = extract_orb_features(img)
                    if desc is not None:
                        image_descriptors.append(desc)
            database[tank_type] = image_descriptors
    return database
 
def find_best_match(input_image_path, database_path):
    input_img = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    _, input_desc = extract_orb_features(input_img)
 
    database = load_database_features(database_path)
 
    best_score = float('inf')
    best_match = None
 
    for tank_type, descriptors_list in database.items():
        total_distance = 0
        matches_count = 0
        for db_desc in descriptors_list:
            matches = match_features(input_desc, db_desc)
            distance = sum(m.distance for m in matches)
            total_distance += distance
            matches_count += len(matches)
 
        if matches_count > 0:
            avg_distance = total_distance / matches_count
            if avg_distance < best_score:
                best_score = avg_distance
                best_match = tank_type
 
    return best_match
 
# Example usage
if __name__ == "__main__":
    input_image = "C:/Users/Yousef/Desktop/Year4Sem2/Graduation projec/T-34_2.jpg"  # Path to input tank image
    database_dir = "C:/Users/Yousef/Desktop/Year4Sem2/Graduation projec/CroppedTanks"          # Folder containing tank folders
    match = find_best_match(input_image, database_dir)
    print(f"Best match: {match}")