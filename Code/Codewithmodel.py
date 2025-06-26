import torch
from torchvision import models, transforms
from torch import nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

# Define your dataset class
class TankDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return image, label

# Preprocessing pipeline for ResNet input
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load pre-trained ResNet18 model
model = models.resnet18(pretrained=True)

# Modify the final layer of ResNet18 to match the number of tank types (e.g., 5 types of tanks)
num_classes = 5  # Replace this with the actual number of tank types in your dataset
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Freeze all layers except the final one (for transfer learning)
for param in model.parameters():
    param.requires_grad = False
model.fc.requires_grad = True

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

# Helper function to load images and labels
def load_data(image_folder_path, label_file_path):
    image_paths = []
    labels = []
    
    with open(label_file_path, 'r') as f:
        for line in f:
            # Assuming the label file has image paths and corresponding tank type labels
            img_path, label = line.strip().split(',')
            image_paths.append(os.path.join(image_folder_path, img_path))
            labels.append(int(label))  # Ensure label is integer
    return image_paths, labels

# Load your dataset
image_folder_path = 'C:/Users/Yousef/Desktop/Year4Sem2/Graduation projec/augmented_images'  # Path where your tank images are stored
label_file_path = 'C:/Users/Yousef/Desktop/Year4Sem2/Graduation projec/augmented_labels'  # File containing image paths and labels

image_paths, labels = load_data(image_folder_path, label_file_path)

# Split data into training and validation sets (80% training, 20% validation)
train_size = int(0.8 * len(image_paths))
train_image_paths = image_paths[:train_size]
train_labels = labels[:train_size]
val_image_paths = image_paths[train_size:]
val_labels = labels[train_size:]

# Create Dataset objects
train_dataset = TankDataset(train_image_paths, train_labels, transform=transform)
val_dataset = TankDataset(val_image_paths, val_labels, transform=transform)

# Create DataLoaders for batching
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Train the model (simplified training loop)
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to('cuda'), labels.to('cuda')
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Validation phase
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to('cuda'), labels.to('cuda')
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Validation Accuracy: {100 * correct / total:.2f}%')

# Save the model after training
torch.save(model.state_dict(), 'tank_classifier.pth')

# Function to predict tank type from a new image
def predict_tank_type(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to('cuda')  # Add batch dimension and move to GPU

    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted_class = torch.max(output, 1)
    
    return predicted_class.item()

# Example usage for predicting tank type
new_image_path = 'path_to_new_image.jpg'  # Replace with the path to the image you want to classify
predicted_class = predict_tank_type(new_image_path)
print(f'Tank Type Predicted: {predicted_class}')
