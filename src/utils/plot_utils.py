import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_results(model_name):
    """
    Load certain evaluation result from JSON file
    """
    results_path = f'checkpoints/{model_name}_results.json'
    with open(results_path, 'r') as f:
        results = json.load(f)
    return results

def plot_confusion_matrix(conf_matrix, model_name):
    """
    Draw confusion matrix
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(10), yticklabels=np.arange(10))
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
