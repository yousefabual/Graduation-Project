import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizerFast, CLIPFeatureExtractor

def classify_tank_image(image_path, dataset_path, model_name="openai/clip-vit-base-patch32"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Use fast tokenizer
    tokenizer = CLIPTokenizerFast.from_pretrained(model_name)
    feature_extractor = CLIPFeatureExtractor.from_pretrained(model_name)
    processor = CLIPProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)

    # Load model
    model = CLIPModel.from_pretrained(model_name).to(device)

    # Tank class names from folder names
    tank_names = sorted([
        name for name in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, name))
    ])

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(
        text=tank_names,
        images=image,
        return_tensors="pt",
        padding=True
    ).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)

    # Prediction
    predicted_index = probs.argmax().item()
    predicted_tank = tank_names[predicted_index]
    confidence = probs[0, predicted_index].item()

    print(f"Predicted tank: {predicted_tank} (Confidence: {confidence:.2%})")
    return predicted_tank, confidence

# Set paths
dataset_path = "C:/Users/Yousef/Desktop/Year4Sem2/Graduation projec/Cropped-Augmented"
image_path = "C:/Users/Yousef/Desktop/Year4Sem2/Graduation projec/Images for testing/M1/M.jpg"

# Run with CLIP ViT-L/14
classify_tank_image(image_path, dataset_path, model_name="openai/clip-vit-base-patch32")
