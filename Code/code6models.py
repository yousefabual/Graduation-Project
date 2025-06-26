import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

def classify_tank_image(image_path, dataset_path, model_name="openai/clip-vit-base-patch32"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load CLIP model and processor
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    # Extract tank class names from subfolder names
    tank_names = sorted([
        name for name in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, name))
    ])

    # Load and preprocess the input image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(
        text=tank_names,
        images=image,
        return_tensors="pt",
        padding=True
    ).to(device)

    # Run the model
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # shape: [1, num_classes]
        probs = logits_per_image.softmax(dim=1)

    # Get top prediction
    predicted_index = probs.argmax().item()
    predicted_tank = tank_names[predicted_index]
    confidence = probs[0, predicted_index].item()

    print(f"Predicted tank: {predicted_tank} (Confidence: {confidence:.2%})")
    return predicted_tank, confidence

dataset_path ="C:/Users/Yousef/Desktop/Year4Sem2/Graduation projec/Cropped-Augmented" # your root folder containing subfolders for each tank
image_path = "C:/Users/Yousef/Desktop/Year4Sem2/Graduation projec/Images for testing/M1/M.jpg"
classify_tank_image(image_path, dataset_path, model_name="openai/clip-vit-large-patch14")

