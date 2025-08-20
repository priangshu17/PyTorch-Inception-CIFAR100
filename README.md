# PyTorch-Inception-CIFAR100

# PyTorch InceptionNet for CIFAR-100 Classification

This repository provides a from-scratch implementation of a GoogLeNet-style Inception network in PyTorch. The model is designed and trained for image classification on the challenging CIFAR-100 dataset.

## 📝 Overview

The goal of this project is to offer a clear, step-by-step guide to building a multi-branch convolutional neural network. The Inception architecture, famous for its computational efficiency and performance, is constructed from basic PyTorch modules in a single, well-documented Jupyter Notebook.

### Key Features:
* **Inception Module from Scratch**: The core `InceptionModule` is built from the ground up, showcasing its parallel 1x1, 3x3, 5x5, and max-pooling convolutional branches.
* **GoogLeNet-Style Architecture**: A deep network is constructed by stacking multiple Inception modules, adapted for the 32x32 image size of CIFAR-100.
* **CIFAR-100 Dataset**: The model is trained on the 100-class CIFAR-100 dataset.
* **Complete Training Pipeline**: The notebook includes a reusable pipeline for data loading, transformation, training, validation, and testing.
* **Performance Visualization**: Training history for loss and accuracy is plotted with `matplotlib` for easy analysis.

## 📈 Model Performance

The notebook trains the model for 10 epochs and includes code to plot the training history and calculate the final test accuracy. The performance metrics can be reproduced by running the notebook.

🏗️ Code Structure
The Jupyter Notebook py_inception.ipynb is organized as follows:

Imports and Setup: Imports required libraries and sets up the device (CPU/GPU).

Data Handling: Loads and preprocesses the CIFAR-100 dataset, splitting it into training, validation, and test sets using DataLoader.

Model Architecture:

InceptionModule: A class defining the core multi-branch Inception block.

BasicInceptionNet: The main class that assembles the InceptionModule blocks into a full GoogLeNet-style network.

Training Pipeline:

Defines the loss function (CrossEntropyLoss) and optimizer (Adam).

A training_loop function manages the epoch-based training and validation.

An evaluate_model helper function computes loss and accuracy.

Evaluation and Visualization:

A plot_metrics function generates plots for training, validation, and test loss/accuracy.

The final cell reports the model's accuracy on the unseen test data.
