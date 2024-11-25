import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import joblib
from utils.eval_utils import evaluate_model
from utils.plot_utils import plot_confusion_matrix
import pandas as pd

def evaluate_saved_models(test_features, test_labels):
    models = {
        "DecisionTree_Python": "checkpoints/DecisionTree_Python.pkl",
        "DecisionTree_Sklearn": "checkpoints/DecisionTree_Sklearn.pkl",
    }

    # Save results
    results = []

    for model_name, model_path in models.items():
        print(f"Evaluating {model_name}...")
        # Load models
        model = joblib.load(model_path)

        # Evaluate models
        model_type = "custom" if "Python" in model_name else "sklearn"
        eval_results = evaluate_model(model, test_features, test_labels, model_type=model_type)

        # Draw confusion matrix
        plot_confusion_matrix(eval_results["confusion_matrix"], model_name)

        # Save evaluation results
        results.append({
            "Model": model_name,
            "Accuracy": eval_results["accuracy"],
            "Precision": eval_results["precision"],
            "Recall": eval_results["recall"],
            "F1-score": eval_results["f1_score"]
        })

    # Create evaluation results
    results_df = pd.DataFrame(results)
    print("\nEvaluation Results:")
    print(results_df)
    results_df.to_csv("results/evaluation_summary.csv", index=False)