# Import required libraries
import torch
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Import related modules
from utils.feature_extraction import prepare_data, load_pretrained_resnet, extract_features, apply_pca
from src.models.naive_bayes import NaiveBayes, train_naive_bayes_sklearn
from utils.eval_utils import evaluate_model, save_model_and_results
from utils.plot_utils import load_results, plot_confusion_matrix
from utils.report_utils import save_results_as_text

# Wrapped by if __name__ == '__main__': ，make sure it is well run on Windows
if __name__ == "__main__":
    # Check available CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data prepared and feature extraction
    train_data, test_data = prepare_data()

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

    # Load and show Naive Bayes Python and Numpy edition
    nb_results_python = load_results("NaiveBayes_Python")
    print("Naive Bayes Python Model Results:")
    print(nb_results_python)
    plot_confusion_matrix(nb_results_python['confusion_matrix'], "NaiveBayes_Python")

    # Load and show Naive Bayes Scikit-learn edition  版本的评估结果
    nb_results_sklearn = load_results("NaiveBayes_Sklearn")
    print("Naive Bayes Scikit-learn Model Results:")
    print(nb_results_sklearn)
    plot_confusion_matrix(nb_results_sklearn['confusion_matrix'], "NaiveBayes_Sklearn")


