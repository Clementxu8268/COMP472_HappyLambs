import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle
import json
import numpy as np
import os
from .plot_utils import plot_confusion_matrix
import pandas as pd  # for CSV file

def evaluate_model(model, test_features, test_labels, model_type="sklearn", test_data_loader=None):
    #Evaluate models' performance
    if model_type == "sklearn":
        # Use Scikit-learn model to predict
        predictions = model.predict(test_features)
    elif model_type == "custom":
        # Use custom model to predict
        predictions = model.predict(np.array(test_features))
    elif model_type == "torch":
        if test_data_loader is None:
            raise ValueError("Torch models require a DataLoader for evaluation.")
        model.eval()
        predictions = []
        with torch.no_grad():
            for inputs, _ in test_data_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        predictions = np.array(predictions)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Calculate indicators
    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions, average="macro")
    recall = recall_score(test_labels, predictions, average="macro")
    f1 = f1_score(test_labels, predictions, average="macro")
    conf_matrix = confusion_matrix(test_labels, predictions)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": conf_matrix.tolist()  # Make sure for JSON
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

def print_saved_results(checkpoints_dir, results_dir, show_confusion_matrix=False):
    """
    Print last saved models and results

    Args:
        checkpoints_dir (str): models' file path。
        results_dir (str): results' file path。
        show_confusion_matrix (bool): whether to show confusion matrix。
    """
    print("\n=== Saved Models and Results ===")

    # Show checkpoints folder content
    print("\n[Saved Models and Results in Checkpoints]")
    if os.path.exists(checkpoints_dir):
        print(f"Checking checkpoints directory: {os.path.abspath(checkpoints_dir)}")
        model_files = os.listdir(checkpoints_dir)
        if model_files:
            for model_file in model_files:
                print(f"- {model_file}")
                if model_file.endswith(".json"):
                    # Try to load JSON files
                    try:
                        with open(os.path.join(checkpoints_dir, model_file), "r") as f:
                            result_data = json.load(f)
                            print(f"  Summary for {model_file}:")
                            print(f"    - Accuracy: {result_data['accuracy']:.4f}")
                            print(f"    - F1-Score: {result_data['f1_score']:.4f}")
                            print(f"    - Precision: {result_data['precision']:.4f}")
                            print(f"    - Recall: {result_data['recall']:.4f}")

                            # Optional display of confusion matrix
                            if show_confusion_matrix and "confusion_matrix" in result_data:
                                conf_matrix = result_data["confusion_matrix"]
                                if conf_matrix:  # Check confusion matrix existing
                                    plot_confusion_matrix(conf_matrix, model_file.replace("_results.json", ""))
                    except Exception as e:
                        print(f"  Error reading {model_file}: {e}")
        else:
            print("No saved models or results found in checkpoints.")
    else:
        print("Checkpoints directory does not exist.")

    # Show results from dir
    print("\n[Saved General Results in Results]")
    if os.path.exists(results_dir):
        print(f"Checking results directory: {os.path.abspath(results_dir)}")
        result_files = os.listdir(results_dir)
        if result_files:
            for result_file in result_files:
                print(f"- {result_file}")
                result_path = os.path.join(results_dir, result_file)
                try:
                    if result_file.endswith(".txt"):
                        # Print content of files
                        with open(result_path, "r") as f:
                            print(f"  Content of {result_file}:\n{f.read()}")
                    elif result_file.endswith(".csv"):
                        # Load and print CSV files
                        df = pd.read_csv(result_path)
                        print(f"  Preview of {result_file}:")
                        print(df.head())  # Print first five lines
                except Exception as e:
                    print(f"  Error reading {result_file}: {e}")
        else:
            print("No saved results found in results.")
    else:
        print("Results directory does not exist.")

def evaluate_mlp(model, test_loader, device):
    """
    Evaluate the performance of an MLP model.

    Args:
        model (torch.nn.Module): The trained MLP model.
        test_loader (DataLoader): DataLoader containing the test dataset.
        device (torch.device): The device (CPU or GPU) to perform evaluation.

    Returns:
        dict: A dictionary containing accuracy, precision, recall, F1-score, and confusion matrix.
    """
    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    predictions = np.array(predictions)
    true_labels = np.array(true_labels)

    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average="macro")
    recall = recall_score(true_labels, predictions, average="macro")
    f1 = f1_score(true_labels, predictions, average="macro")
    conf_matrix = confusion_matrix(true_labels, predictions)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": conf_matrix.tolist()  # Ensure JSON serializability
    }