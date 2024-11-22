import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle
import json
import numpy as np
import os

def evaluate_model(model, test_features, test_labels, model_type="sklearn", test_data_loader=None):
    """
    Evaluate the model, calculate the accuracy, f1 and so on
    Based on model type to process sklearn or torch type
    """
    # Check model type
    if model_type == "sklearn":
        predictions = model.predict(test_features)
    elif model_type == "torch":
        model.eval()
        with torch.no_grad():
            # For tensor
            inputs = torch.tensor(test_features, dtype=torch.float32)
            labels = torch.tensor(test_labels, dtype=torch.long)
            # Get output of model
            outputs = model(inputs)
            # Get prediction
            _, predicted = torch.max(outputs, 1)
            predictions = predicted.cpu().numpy()

    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions, average='macro')
    recall = recall_score(test_labels, predictions, average='macro')
    f1 = f1_score(test_labels, predictions, average='macro')
    cm = confusion_matrix(test_labels, predictions)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }

def save_model_and_results(model, results, model_name, model_type="sklearn"):
    """
    Save model and result of evaluation
    """

    # Create checkpoints folder if it is not there
    os.makedirs('checkpoints', exist_ok=True)

    # Save mode
    if model_type == "sklearn":
        # Save scikit-learn model as .pkl file
        model_path = f'checkpoints/{model_name}.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    elif model_type == "torch":
        # Save PyTorch model as .pth file
        model_path = f'checkpoints/{model_name}.pth'
        torch.save(model.state_dict(), model_path)

    # Convert numpy.ndarray to list for JSON
    results_serializable = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in
                            results.items()}

    # Save result as JSON
    results_path = f'checkpoints/{model_name}_results.json'
    with open(results_path, 'w') as f:
        json.dump(results_serializable, f, indent=4)

    print(f"Model and results saved as {model_name}.pkl (or .pth) and {model_name}_results.json in checkpoints/ directory.")