import torch
import torch.nn as nn
from tqdm import tqdm

class VGG11Experiment(nn.Module):
    def __init__(self, num_classes=10, conv_configs=None):
        super(VGG11Experiment, self).__init__()
        if conv_configs is None:
            conv_configs = [
                (3, 64, 3, 1, 1),  # Block 1
                (64, 128, 3, 1, 1),  # Block 2
                (128, 256, 3, 1, 1), (256, 256, 3, 1, 1),  # Block 3
                (256, 512, 3, 1, 1), (512, 512, 3, 1, 1),  # Block 4
                (512, 512, 3, 1, 1), (512, 512, 3, 1, 1),  # Block 5
            ]

        layers = []
        self.input_resolution = 32  # CIFAR-10 image size

        for in_channels, out_channels, kernel_size, stride, padding in conv_configs:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            self.input_resolution = self._compute_output_resolution(self.input_resolution, kernel_size, stride, padding)

            if self.input_resolution >= 2:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                self.input_resolution //= 2

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Linear(512 * self.input_resolution * self.input_resolution, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    @staticmethod
    def _compute_output_resolution(input_size, kernel_size, stride, padding):
        return (input_size - kernel_size + 2 * padding) // stride + 1

def train_vgg11_experiment(model, train_loader, test_loader, device, epochs):
    """
    Train the experimental VGG11 model with dynamic progress display.
    Args:
        model (torch.nn.Module): VGG11 experiment model.
        train_loader (DataLoader): Training data loader.
        test_loader (DataLoader): Validation data loader.
        device (torch.device): Device to use for training.
        epochs (int): Number of epochs.
    Returns:
        torch.nn.Module: Trained model.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")

        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")
        validate_vgg11_experiment(model, test_loader, device)  # Optional validation step

    return model

def validate_vgg11_experiment(model, test_loader, device):
    """
    Validate the experimental VGG11 model with summary results.
    Args:
        model (torch.nn.Module): Model to validate.
        test_loader (DataLoader): Validation data loader.
        device (torch.device): Device to use for validation.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Validating", unit="batch"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Validation Accuracy: {accuracy:.2f}%")
    return accuracy