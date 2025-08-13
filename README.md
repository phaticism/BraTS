# 2D Medical Image Segmentation with U-Net and Federated Learning

This directory contains a comprehensive implementation of 2D medical image segmentation using U-Net architecture and Federated Learning (FL) for brain tumor segmentation on the BraTS dataset.

## Project Overview

The project implements a 2D U-Net model for medical image segmentation with support for both centralized training and federated learning approaches. It's designed to work with brain tumor segmentation datasets (BraTS 2017, BraTS 2021) and includes preprocessing, training, evaluation, and visualization capabilities.

## Core Components

### 1. Model Architecture (`model.py`)
- **U-Net 2D Implementation**: 2D U-Net architecture for medical image segmentation
- **Loss Functions**: Dice coefficient loss, binary cross-entropy, and combined losses
- **Metrics**: Dice coefficient, sensitivity, specificity calculations
- **Training Utilities**: Model training, validation, and inference functions

### 2. Data Management (`dataloader.py`)
- **Data Loading**: Efficient loading of medical imaging data
- **Preprocessing**: Image normalization, augmentation, and cropping
- **Data Splitting**: Train/validation/test dataset partitioning

### 3. Federated Learning (`flwrclient.py`, `flwrserver.py`)
- **FL Client**: Individual node implementation using Flower framework
- **FL Server**: Central server for coordinating federated training
- **Multi-Node Support**: Configurable for multiple participating nodes

### 4. Configuration (`settings.py`)
- **Model Parameters**: Feature maps, learning rate, batch size, epochs
- **Data Paths**: Dataset and output directory configurations
- **Training Settings**: Dropout, augmentation, and optimization parameters
- **Hardware Settings**: Threading and memory management configurations

## Workflow Notebooks

### Preprocessing (`0 – Preprocessing.ipynb`)
- Data loading and validation
- Image preprocessing and normalization
- Dataset preparation for training

### Training (`1 – Training.ipynb`)
- Centralized model training
- Model checkpointing and saving

### Evaluation (`2 – Evaluation.ipynb`)
- Model performance assessment
- Metrics calculation

### Visualization (`3 – Visualization.ipynb`)
- Results analysis and plotting

### FL Training (`5 – FL.ipynb`)
- FL training implementation
- Model checkpointing and saving

### FL Evaluation (`6 – FL Evaluation.ipynb`)
- Federated model assessment
- Metrics calculation

### FL Visualization (`7 – FL Visualization.ipynb`)
- Results analysis and plotting

## Key Features

- **2D U-Net Architecture**: Optimized for medical image segmentation
- **Federated Learning**: Privacy-preserving distributed training
- **Comprehensive Evaluation**: Multiple metrics and visualization tools
- **Flexible Configuration**: Easy parameter tuning via settings.py
- **Multi-Dataset Support**: BraTS 2017, 2021.
- **Production Ready**: Includes model saving, loading, and inference capabilities

## Dependencies

The project requires the following key packages:
- **TensorFlow 2.15.0**: Deep learning framework
- **Flower (flwr)**: Federated learning framework
- **Medical Imaging**: nibabel for medical image formats
- **Data Processing**: numpy, matplotlib for analysis and visualization

## Usage

1. **Setup**: Install dependencies from `requirements.txt`
2. **Data Preparation**: Run preprocessing notebook to prepare datasets
3. **Training**: Choose between centralized (`1 – Training.ipynb`) or federated (`5 – FL.ipynb`) training
4. **Evaluation**: Assess model performance using evaluation notebooks
5. **Visualization**: Generate insights and plots using visualization notebooks
