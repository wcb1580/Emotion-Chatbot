import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataset import CustomImageDataset
import os
# Define the training function
def train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=1000):
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0

        # Iterate over training data
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Track the loss
            running_loss += loss.item()

        # Calculate average training loss for the epoch
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Evaluate on validation data
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                val_outputs = model(val_inputs)
                val_loss += criterion(val_outputs, val_labels).item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Print statistics for the epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Calculate accuracy
        correct = 0
        total = 0
        with torch.no_grad():  # Disable gradient computation for efficiency during evaluation
            for images, labels in tqdm(val_loader):  # Use val_loader instead of test_loader
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                predicted = torch.argmax(outputs.data, 1)  # Get the index of the max log-probability
                total += labels.size(0)
                correct += (predicted == labels).sum().item()  # Direct comparison without argmax

        # Print accuracy
        print(f"Accuracy on validation set: {(correct / total) * 100:.2f}%")
                    # Save model checkpoint
        save_path = "checkpoints"
        os.makedirs(save_path, exist_ok=True)
        torch.save(model, save_path + os.sep + str(epoch) + ".pt")

    print("Training Complete")
    return train_losses, val_losses


# Function to plot the training and validation loss
def plot_loss(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Cross-Entropy Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Set device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # Load ResNet18 model
    num_classes = 8  # Replace with the number of classes in your dataset
    model = timm.create_model('resnet18', pretrained=False, num_classes=num_classes)
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Define data transformations with augmentations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),            # Resize for consistent input size
        transforms.RandomResizedCrop(224),        # Randomly crop to 224x224
        transforms.RandomHorizontalFlip(p=0.5),   # Randomly flip horizontally
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the dataset
        # Create datasets and data loaders
    train_dataset = CustomImageDataset(root_dir="C:\\Users\\pw335\\Downloads\\emotion_recognition_CNN\\emotion_recognition_CNN\\train_images", transform=transform)
    test_dataset = CustomImageDataset(root_dir="C:\\Users\\pw335\\Downloads\\emotion_recognition_CNN\\emotion_recognition_CNN\\test_images", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    # Train the model
    train_losses, val_losses = train(model, train_loader, test_loader, criterion, optimizer, device, num_epochs=1000)

    # Plot the loss
            
    plot_loss(train_losses, val_losses)
