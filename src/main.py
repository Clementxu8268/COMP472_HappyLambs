# Import required libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import time

# Check your device whether supports CUDA
cuda_available = torch.cuda.is_available()
print("CUDA available:", cuda_available)

# If it supports CUDA, print CPU's information
if cuda_available:
    print("GPU device name:", torch.cuda.get_device_name(0))
    print("Number of GPUs available:", torch.cuda.device_count())
    print("Current CUDA device:", torch.cuda.current_device())
else:
    print("CUDA is not supported on this system.")