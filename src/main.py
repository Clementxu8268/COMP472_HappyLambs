# Import required libraries
import torch
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Import related modules
from utils.feature_extraction import prepare_data, load_pretrained_resnet, extract_features, apply_pca
from models.naive_bayes import NaiveBayes, train_naive_bayes_sklearn
from utils.eval_utils import evaluate_model, save_model_and_results, print_saved_results, evaluate_mlp
from utils.report_utils import save_results_as_text
from models.decision_tree import DecisionTree, train_sklearn_decision_tree
from evaluate import evaluate_saved_models
from models.mlp import train_mlp

# Get root path (main.py's path)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define path
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Wrapped by if __name__ == '__main__': ，make sure it is well run on Windows
if __name__ == "__main__":
    # Check available CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Checkpoints directory: {os.path.abspath(CHECKPOINTS_DIR)}")
    print(f"Results directory: {os.path.abspath(RESULTS_DIR)}")

    # Print last saved models and results
    print_saved_results(CHECKPOINTS_DIR, RESULTS_DIR, True)

    # Data prepared and feature extraction
    train_data, test_data = prepare_data()

    # Prompt retrain or not
    retrain = input("\nDo you want to retrain the models? (yes/no): ").strip().lower()

    if retrain == "yes":
        print("\n=== Retraining Models ===")
        # Load pre-trained ResNet-18 model and remove the last layer of it
        resnet = load_pretrained_resnet()
        resnet = resnet.to(device)
        print("ResNet-18 loaded successfully!")

        # Extract train set and test set
        train_features, train_labels = extract_features(train_data, resnet, device)
        test_features, test_labels = extract_features(test_data, resnet, device)

        # PCA decrease dimension
        train_features = apply_pca(train_features)
        test_features = apply_pca(test_features)

        # Save the results
        results = {}

        # MLP part
        # Train and evaluate MLP
        print("Training MLP Model...")
        mlp_model = train_mlp(train_features, train_labels, epochs=10, device=device)

        # Create DataLoader for test set
        test_dataset = TensorDataset(
            torch.tensor(test_features, dtype=torch.float32),
            torch.tensor(test_labels, dtype=torch.long)
        )
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        # Evaluate MLP using the dedicated function
        print("Evaluating MLP Model...")
        mlp_results = evaluate_mlp(mlp_model, test_loader, device=device)

        # Save results and model
        save_model_and_results(mlp_model, mlp_results, "MLP")
        results["MLP"] = mlp_results

        # Experiment with different MLP configurations
        hidden_layer_configs = [
            [512, 512],  # Default configuration
            [512, 256],  # Smaller layers
            [1024, 512, 256],  # Deeper network
        ]

        for config in hidden_layer_configs:
            print(f"\nTraining MLP with hidden layer sizes: {config}...")
            mlp_model = train_mlp(train_features, train_labels, epochs=10, device=device, hidden_sizes=config)

            # Evaluate MLP Model
            print(f"Evaluating MLP Model with hidden layer sizes: {config}...")
            test_dataset = TensorDataset(
                torch.tensor(test_features, dtype=torch.float32).to(device),
                torch.tensor(test_labels, dtype=torch.long).to(device)
            )
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
            mlp_results = evaluate_mlp(mlp_model, test_loader, device=device)
            save_model_and_results(mlp_model, mlp_results, f"MLP_Hidden_{'_'.join(map(str, config))}")

        # Decision tree part
        # Decision tree training and evaluation
        print("Training Decision Tree (Python implementation)...")
        dt_python = DecisionTree(max_depth=50)
        dt_python.fit(train_features, train_labels)
        dt_python_results = evaluate_model(dt_python, test_features, test_labels, model_type="custom")
        save_model_and_results(dt_python, dt_python_results, "DecisionTree_Python")
        results["DecisionTree_Python"] = dt_python_results

        print("Training Decision Tree (Scikit-learn implementation)...")
        dt_sklearn = train_sklearn_decision_tree(train_features, train_labels, max_depth=50)
        dt_sklearn_results = evaluate_model(dt_sklearn, test_features, test_labels, model_type="sklearn")
        save_model_and_results(dt_sklearn, dt_sklearn_results, "DecisionTree_Sklearn")
        results["DecisionTree_Sklearn"] = dt_sklearn_results

        # Experiment different depth
        for depth in [10, 20, 30, 40, 50]:
            print(f"Training Scikit-learn Decision Tree with max depth {depth}...")
            dt_sklearn = train_sklearn_decision_tree(train_features, train_labels, max_depth=depth)
            dt_sklearn_results = evaluate_model(dt_sklearn, test_features, test_labels, model_type="sklearn")
            save_model_and_results(dt_sklearn, dt_sklearn_results, f"DecisionTree_Sklearn_Depth_{depth}")

        # Naive Bayes part
        # Train and evaluate Naive Bayes Python and Numpy edition
        print("Training Naive Bayes (Python and Numpy implementation)...")
        nb_model_python = NaiveBayes()
        nb_model_python.fit(train_features, train_labels)
        nb_results_python = evaluate_model(nb_model_python, test_features, test_labels, model_type="sklearn")
        save_model_and_results(nb_model_python, nb_results_python, "NaiveBayes_Python")
        save_results_as_text(nb_results_python, "NaiveBayes_Python")

        # Train and evaluate Naive Bayes Scikit-learn edition
        print("Training Naive Bayes (Scikit-learn implementation)...")
        nb_model_sklearn = train_naive_bayes_sklearn(train_features, train_labels)
        nb_results_sklearn = evaluate_model(nb_model_sklearn, test_features, test_labels, model_type="sklearn")
        save_model_and_results(nb_model_sklearn, nb_results_sklearn, "NaiveBayes_Sklearn")
        save_results_as_text(nb_results_sklearn, "NaiveBayes_Sklearn")

        # Print current saved models and results
        print_saved_results(CHECKPOINTS_DIR, RESULTS_DIR, True)
    else:
        print("\nSkipping retraining. Evaluating saved models...")
        # Load feature
        resnet = load_pretrained_resnet()
        resnet = resnet.to(device)
        test_features, test_labels = extract_features(test_data, resnet, device)
        test_features = apply_pca(test_features)

        # Evaluation
        evaluate_saved_models(test_features, test_labels)