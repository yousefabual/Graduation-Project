import streamlit as st
import os
import random
import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
import torchvision.transforms as transforms

# === Static Paths ===
BASE_PATH = r"C:\Users\Yousef\Desktop\Year4Sem2\Graduation projec\simulationn\Simulation_files\Simulation_files"
BG_PATH = os.path.join(BASE_PATH, "backgrounds")
TANK_PATH = os.path.join(BASE_PATH, "test_tanks")
MODEL_DET_PATH = os.path.join(BASE_PATH, "models", "v11n_100epoch_best.pt")
MODEL_CLF_PATH = os.path.join(BASE_PATH, "models", "FULL_efficeintNetv3_tanks_15epochs_0.0001lr_300size.pth")

# === Load Models ===
st.title("Tank Detection & Classification Simulator")
st.success("Models and folders loaded from disk.")

det_model = YOLO(MODEL_DET_PATH)
clf_model = torch.load(MODEL_CLF_PATH, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
clf_model.eval()

class_names = ['Al Mared', 'Al-wahsh', 'Centurion', 'Challenger2', 'China_Type99', 'Leopard 2',
               'M1_Abrams', 'M60', 'Magach', 'Merkava IV', 'Russia T-34', 'T90']

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# === Overlay Function ===
def overlay_image(bg, fg, x, y):
    h, w = fg.shape[:2]
    bg_h, bg_w = bg.shape[:2]
    x_start, y_start = max(0, x), max(0, y)
    x_end, y_end = min(bg_w, x + w), min(bg_h, y + h)
    fg_slice = fg[y_start - y:y_end - y, x_start - x:x_end - x]
    bg_slice = bg[y_start:y_end, x_start:x_end]
    if fg_slice.shape[:2] != bg_slice.shape[:2] or fg_slice.size == 0 or bg_slice.size == 0:
        return bg
    alpha = fg_slice[:, :, 3] / 255.0
    for c in range(3):
        bg[y_start:y_end, x_start:x_end, c] = (alpha * fg_slice[:, :, c] +
                                               (1 - alpha) * bg_slice[:, :, c])
    return bg

# === Generate Synthetic Frame ===
def simulate_frame():
    bg_file = random.choice([f for f in os.listdir(BG_PATH) if f.endswith((".png", ".jpg", ".jpeg"))])
    tank_file = random.choice([f for f in os.listdir(TANK_PATH) if f.endswith((".png", ".jpg", ".jpeg"))])

    bg = cv2.imread(os.path.join(BG_PATH, bg_file))
    bg = cv2.resize(bg, (640, 480))

    tank = cv2.imread(os.path.join(TANK_PATH, tank_file), cv2.IMREAD_UNCHANGED)
    tank = cv2.resize(tank, (150, 150))

    x = random.randint(0, 640 - tank.shape[1])
    y = random.randint(0, 480 - tank.shape[0])
    return overlay_image(bg.copy(), tank, x, y)

# === Detection + Classification ===
def run_simulation():
    frame = simulate_frame()
    if frame is None:
        st.error("Simulation failed.")
        return

    results = det_model.predict(source=frame, conf=0.15, verbose=False)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            input_tensor = transform(crop).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")

            with torch.no_grad():
                output = clf_model(input_tensor)
                probs = torch.nn.functional.softmax(output, dim=1)
                confidence, pred = torch.max(probs, 1)
                label = f"{class_names[pred.item()]} ({confidence.item():.2f})"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    return frame[..., ::-1]  # Convert BGR to RGB

# === Streamlit UI ===
if st.button("Run Simulation"):
    st.info("Running...")
    output = run_simulation()
    if output is not None:
        st.image(output, caption="Detection & Classification", use_column_width=True)
