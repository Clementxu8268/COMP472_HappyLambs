# COMP472 Group HappyLambs' Final Project

## Authors
- **Ke Xu**  
  - **ID**: 40253950  
  - **Email**: kxu8268@gmail.com  
- **Jingyuan Zhang**  
  - **ID**: 40257596  
  - **Email**: jingyuanzhangzjy@gmail.com  

---

## Overview
This project is the final submission for COMP472. It implements various machine learning models, including decision trees, Naive Bayes, Multi-Layer Perceptrons (MLPs), and Convolutional Neural Networks (CNNs). The program is built with Python and follows a modular structure for ease of understanding and extensibility.

---

## Features
1. **Decision Trees**: Implemented using both Python and Scikit-learn, with experiments on varying depths.  
2. **Naive Bayes**: Compared Python/Numpy implementations with Scikit-learn’s version.  
3. **Multi-Layer Perceptron (MLP)**: Trained on CIFAR-10 features with customizable architecture and hidden layers.  
4. **VGG11 CNN**: Trained directly on CIFAR-10 images, with experiments on depth and kernel sizes.  
5. **Modular Evaluation**: Unified evaluation framework for metrics like accuracy, precision, recall, and F1-score.  

---
## Documents
For more details, refer to the generated pdoc documentation located in the docs/ folder. Open index.html in any browser for an organized overview of the project's modules.

---
## Prerequisites
Ensure you have the following installed:
- **Python**: Version 3.8 or higher
- **PyTorch**: Version 1.11.0 or higher with CUDA support (optional)
---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Clementxu8268/COMP472_HappyLambs.git
   cd COMP472_HappyLambs
2. Other dependencies:
   ```bash
   pip install -r requirements.txt
3. Run main.py for whole interactional way to check details. 

## Interaction with main.py
### Main Function Explanation:

This `main` function orchestrates the training, evaluation, and experimentation of various machine learning models. Here's a breakdown of its interactive branches and processes:

---

#### **Startup**
1. **CUDA Check**: The program detects whether a GPU (CUDA) is available. The device being used (CPU/GPU) is printed to the console.  
   - Example Output: `Using device: cuda` or `Using device: cpu`.

2. **Directory Information**: Prints the paths where model checkpoints and results will be stored.  
   - Example Output:  
     ```
     Checkpoints directory: /path/to/checkpoints
     Results directory: /path/to/results
     ```

3. **Dataset Preparation**: Prepares the CIFAR-10 dataset, including transformations, normalization, and creation of DataLoaders for training and testing.

---

#### **Interactive Prompt**
- **Prompt**: `Do you want to retrain the models? (yes/no):`  
- **Input Options**:
  - **`yes`**: Proceeds with training and experimentation on models.
  - **`no`**: Skips retraining and evaluates saved models directly.

---

### **Branch: Retraining Models**
If the user selects `yes`, the program follows these steps:

#### **1. VGG11 Experiments with Kernel Sizes**
- **Goal**: Test the impact of different kernel sizes on performance.  
- **Action**: A custom `VGG11KernelExperiment` model is trained using various kernel configurations (`5x5`, `7x7`, `2x2`, etc.).  
- **Output**:  
  - Progress through epochs, validation accuracy, and training loss.  
  - The trained model and results are saved to the specified directories.

---

#### **2. VGG11 Experiments with Depth**
- **Goal**: Explore how changing the number of convolutional layers impacts performance.
- **Action**: The `VGG11Experiment` model is trained with three configurations:
  - **Default**: Standard VGG11 architecture.  
  - **Reduced**: Fewer convolutional layers.  
  - **Extended**: More convolutional layers.  
- **Output**:  
  - Training progress for each configuration.  
  - Results (accuracy) and model checkpoints for each depth configuration.

---

#### **3. VGG11 Basic Model**
- **Goal**: Train the standard VGG11 model on CIFAR-10.  
- **Action**: The predefined VGG11 model is trained for 10 epochs.  
- **Output**:  
  - Training progress, validation accuracy, and confusion matrix.  
  - Final model and evaluation results saved to directories.

---

#### **4. Non-CNN Models**
- **Pre-trained ResNet-18**: Feature extraction is performed using ResNet-18.
- **PCA**: Dimensionality reduction is applied to extracted features.
- **MLP**: Multi-Layer Perceptron is trained and evaluated with variations in hidden layers.
- **Decision Trees**: Experiments with Python and Scikit-learn implementations, including varying depths.
- **Naive Bayes**: Comparisons between Python/Numpy and Scikit-learn implementations.

---

### **Branch: Skip Retraining**
If the user selects `no`, the program:
- Loads pre-trained models and previously extracted features.
- Evaluates the models using saved data and generates reports.

---

### **Interactive Outputs**
- **During Training**: Epoch-by-epoch progress, including loss and accuracy. Uses progress bars for better readability.  
- **Model Checkpoints**: Trained models are saved to `checkpoints/`.  
- **Evaluation Results**: Metrics and results saved to `results/`.

---

This flow ensures that all experiments and evaluations align with the project requirements and allow flexibility for future additions or modifications.
