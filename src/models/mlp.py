import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class MLP(nn.Module):
    def __init__(self, input_size=50, hidden_sizes=[512, 512], output_size=10):
        super(MLP, self).__init__()
        layers = []
        in_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))
            in_size = hidden_size
        layers.append(nn.Linear(in_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def train_mlp(train_features, train_labels, epochs=10, batch_size=64, device=None, hidden_sizes=[512, 512]):
    """
    Train an MLP model with specified hidden layer sizes and show training progress.
    """
    # Move features and labels to device
    train_features = torch.tensor(train_features, dtype=torch.float32).to(device)
    train_labels = torch.tensor(train_labels, dtype=torch.long).to(device)

    # Create DataLoader
    train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, loss, and optimizer
    model = MLP(input_size=50, hidden_sizes=hidden_sizes, output_size=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Train the model
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        # Initialize tqdm progress bar
        with tqdm(total=len(loader), desc=f"Epoch {epoch + 1}/{epochs}") as pbar:
            for inputs, labels in loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                # Update progress bar
                pbar.set_postfix({"Loss": running_loss / (pbar.n + 1)})
                pbar.update(1)

        print(f"Epoch {epoch + 1}/{epochs} completed, Avg Loss: {running_loss / len(loader)}")

    return model
