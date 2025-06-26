import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizerFast, CLIPFeatureExtractor

def load_model_and_text_features(dataset_path, model_name="openai/clip-vit-large-patch14", use_compile=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load fast tokenizer and processor
    tokenizer = CLIPTokenizerFast.from_pretrained(model_name)
    feature_extractor = CLIPFeatureExtractor.from_pretrained(model_name)
    processor = CLIPProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)

    # Load model and move to GPU
    model = CLIPModel.from_pretrained(model_name).to(device)
    if use_compile and hasattr(torch, 'compile'):
        model = torch.compile(model)

    model = model.eval().half()  # Use FP16

    # Get tank class names
    tank_names = sorted([
        name for name in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, name))
    ])

    # Pre-compute and cache text features
    with torch.no_grad():
        text_inputs = tokenizer(tank_names, return_tensors="pt", padding=True).to(device)
        text_features = model.get_text_features(**text_inputs).half()
        text_features /= text_features.norm(p=2, dim=-1, keepdim=True)

    return model, processor, text_features, tank_names, device


def classify_image(image_path, model, processor, text_features, tank_names, device):
    # Load and resize image for speed
    image = Image.open(image_path).convert("RGB").resize((224, 224))
    
    # Preprocess image
    image_input = processor(images=image, return_tensors="pt")["pixel_values"].to(device).half()

    # Get image features
    with torch.no_grad():
        image_features = model.get_image_features(image_input)
        image_features /= image_features.norm(p=2, dim=-1, keepdim=True)

        # Cosine similarity with class (text) features
        logits = image_features @ text_features.T
        probs = logits.softmax(dim=1)

    # Prediction
    predicted_index = probs.argmax().item()
    predicted_tank = tank_names[predicted_index]
    confidence = probs[0, predicted_index].item()

    print(f"🖼️ Predicted tank: {predicted_tank} (Confidence: {confidence:.2%})")
    return predicted_tank, confidence


# --- USAGE ---
dataset_path = "C:/Users/Yousef/Desktop/Year4Sem2/Graduation projec/Cropped-Augmented"
image_path = "C:/Users/Yousef/Desktop/Year4Sem2/Graduation projec/Images for testing/M1/M.jpg"

# Load model and cache text features (only once)
model, processor, text_features, tank_names, device = load_model_and_text_features(
    dataset_path,
    model_name="openai/clip-vit-large-patch14",
    use_compile=True  # Only works if you have PyTorch 2.0+
)

# Run prediction
classify_image(image_path, model, processor, text_features, tank_names, device)
