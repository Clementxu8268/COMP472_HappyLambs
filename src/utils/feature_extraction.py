import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np
from sklearn.decomposition import PCA

def prepare_data():
    """
    Process CIFAR-10，only use 500 training samples and 100 testing samples for each class
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image as 224x224 for suiting ResNet-18
        transforms.ToTensor(),  # For Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load CIFAR-10
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    train_data = torch.utils.data.Subset(dataset, range(5000))
    test_data = torch.utils.data.Subset(dataset, range(1000))
    return train_data, test_data

def load_pretrained_resnet():
    """
    Load pre-trained ResNet-18, remove the last layer, and only preserve the part of feature extraction
    """
    weights = ResNet18_Weights.IMAGENET1K_V1
    resnet = resnet18(weights=weights)
    resnet.fc = torch.nn.Identity()  # Remove the last layer
    return resnet

def extract_features(data, model, device):
    """
    Use ResNet-18 to extract feature from image
    """
    model.eval()  # Set evaluation mode
    features, labels = [], []
    loader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=False, num_workers=2) # Set batch size, and not shuffle

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            outputs = model(inputs).cpu().numpy()  # Extraction
            features.extend(outputs)
            labels.extend(targets.numpy())

    return np.array(features), np.array(labels)

def apply_pca(features, n_components=50):
    """
    Use PCA to decrease dimension from 512 to 50
    """
    pca = PCA(n_components=n_components)
    return pca.fit_transform(features)