# mnist-cnn-pytorch

# Handwritten Digit Classification using CNN (PyTorch)

A Convolutional Neural Network (CNN) implementation in PyTorch for classifying handwritten digits (0–9) from the MNIST dataset.

The model achieves ~99% test accuracy.

---

## 🚀 Project Overview

This project demonstrates:

- Deep Learning fundamentals
- Convolutional Neural Networks (CNN)
- Multi-class classification
- Model training & evaluation
- GPU training with PyTorch
- Regularization using Dropout

Dataset used:
- MNIST (60,000 training images, 10,000 test images)

---

## 🏗️ Model Architecture

Input: 1 × 28 × 28 grayscale image

Conv2d (1 → 64, kernel=3, padding=1)  
ReLU  
Conv2d (64 → 128, kernel=3, padding=1)  
MaxPool2d (2×2)  
ReLU  
Dropout (0.2)  
Flatten  
Linear (128×14×14 → 10)

Loss Function: CrossEntropyLoss  
Optimizer: Adam (lr=0.001)

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Training Accuracy | ~99% |
| Test Accuracy | ~98–99% |
| Epochs | 10 |
| Batch Size | 64 |

The model generalizes well with minimal overfitting.

---

## 🧠 Key Learnings

- Understanding CNN layers and feature maps
- Handling tensor shape mismatches
- Proper use of CrossEntropyLoss (no manual softmax)
- Using argmax for multi-class prediction
- Preventing overfitting with Dropout
- Debugging model training errors

---

## 📂 Project Structure

