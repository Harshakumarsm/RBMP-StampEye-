import torch
from csrnet.model import CSRNet
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Initialize the model
    model = CSRNet()
    model.eval()  # Set to evaluation mode

    # Create a sample transform pipeline
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
    ])

    # Load and transform a sample image
    try:
        # Replace 'sample.jpg' with your actual image path
        img = Image.open('sample.jpg').convert('RGB')
        img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

        # Run inference
        with torch.no_grad():
            output = model(img_tensor)
            
        # Convert output density map to count
        pred_count = output.sum().item()
        
        print(f"Predicted count: {pred_count:.2f}")

        # Visualize the density map
        density_map = output.squeeze().numpy()
        plt.imshow(density_map, cmap='jet')
        plt.colorbar()
        plt.title(f'Density Map (Total Count: {pred_count:.2f})')
        plt.savefig('density_map.png')
        plt.close()

    except FileNotFoundError:
        print("Please place a sample image named 'sample.jpg' in the root directory")
        print("or modify the image path in the script to point to your test image.")

if __name__ == "__main__":
    main() 