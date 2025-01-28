# Traffic Sign Recognition using Deep Learning Models

This repository contains code for traffic sign recognition using the German Traffic Sign Recognition Benchmark (GTSRB) dataset. The project implements and compares the performance of three popular deep learning architectures: VGG16, AlexNet, and DenseNet.

Alternatively this project can be checked from following colab link:
https://colab.research.google.com/drive/1ekJcZyiYg5K11c0IFNngiVKq2cwDlUTr?usp=sharing

## Table of Contents

    Project Overview
    Dataset
    Models
    Requirements
    Usage
    Results
    References
    License

## Project Overview

The goal of this project is to classify traffic signs using deep learning models. The German Traffic Sign Recognition Benchmark (GTSRB) dataset is used for training and evaluation. Three models—VGG16, AlexNet, and DenseNet—are implemented and compared for their performance in traffic sign recognition.

## Dataset

The German Traffic Sign Recognition Benchmark (GTSRB) dataset is used for this project. It contains over 50,000 images of 43 different traffic sign classes. The dataset is split into training and testing sets, with images resized to 32x32 pixels.

    Dataset Source: GTSRB Dataset
    Number of Classes: 43
    Image Size: 32x32 pixels

## Models

The following deep learning models are implemented and trained on the GTSRB dataset:

    VGG16:

        A pre-trained VGG16 model is used with fine-tuning.

        Custom layers are added for classification.

    AlexNet:

        A custom implementation of AlexNet is used.

        The model consists of convolutional, pooling, and fully connected layers.

    DenseNet:

        A pre-trained DenseNet model is used with fine-tuning.

        Custom layers are added for classification.

## Requirements

To run the code, you need the following dependencies:

    Python 3.x
    TensorFlow 2.x
    Keras
    NumPy
    Matplotlib
    OpenCV
    scikit-learn

You can install the required packages using the following command:
pip install tensorflow numpy matplotlib opencv-python scikit-learn

## Usage

    Clone the Repository:
    git clone https://github.com/musayevff/traffic-sign-recognition.git
    cd traffic-sign-recognition

    Download the Dataset:

        Download the GTSRB dataset from here.
        Extract the dataset into the data/ directory.

    Run the Code:
        Download and Run ANN_Project_Traffic_Sign_Classification.ipynb

    Colab Link:
        Alternatively this code can be checked from the colab link below:
        https://colab.research.google.com/drive/1ekJcZyiYg5K11c0IFNngiVKq2cwDlUTr?usp=sharing

        Model predictions on the test set are also saved.
