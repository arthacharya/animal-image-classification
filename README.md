# Animal Image Classification

This repository contains a Jupyter Notebook (`Animal_Image_Classification.ipynb`) that performs image classification on a dataset of animal images. The notebook is designed to preprocess, train, and evaluate a machine learning model to classify images of different animals.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Dataset](#dataset)

## Introduction
The goal of this project is to classify images of animals into different categories using a machine learning model. The notebook walks through the entire process, from loading and preprocessing the data to training and evaluating the model.

## Dataset
The dataset used in this project can be accessed [here](https://drive.google.com/file/d/14p1ekDW2YKqEY1kUufKjyJJTvlKVpEJZ/view?usp=sharing). It contains images of various animals, each categorized into folders named after the animal (e.g., `dog`, `cat`, `horse`, etc.). The dataset includes images of the following animals:
- Dog
- Horse
- Elephant
- Butterfly
- Chicken
- Cat
- Cow
- Sheep
- Squirrel
- Spider

## Preprocessing
The notebook includes code to preprocess the images, which involves:
1. **Renaming Folders**: The folders containing the images are renamed from Italian to English for easier understanding.
2. **Image Resizing**: Images are resized to a consistent dimension to ensure uniformity in the input data.
3. **Data Augmentation**: Techniques such as rotation, flipping, and zooming are applied to increase the diversity of the training data.

## Model Training
The notebook uses a Convolutional Neural Network (CNN) to classify the images. The model is built using TensorFlow and Keras. The training process includes:
- **Model Architecture**: The CNN consists of multiple convolutional layers followed by max-pooling layers, and fully connected layers.
- **Compilation**: The model is compiled with an appropriate loss function and optimizer.
- **Training**: The model is trained on the preprocessed dataset for a specified number of epochs.

## Evaluation
After training, the model's performance is evaluated on a separate test set. The evaluation metrics include:
- **Accuracy**: The percentage of correctly classified images.
- **Confusion Matrix**: A detailed breakdown of the model's predictions.

## Usage
To use this notebook:
1. Clone the repository to your local machine.
2. Ensure you have the required dependencies installed (see [Dependencies](#dependencies)).
3. Open the `Animal_Image_Classification.ipynb` notebook in Jupyter.
4. Run the cells sequentially to preprocess the data, train the model, and evaluate its performance.

## Dependencies
The following Python libraries are required to run the notebook:
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- OpenCV
- Scikit-learn

## Dataset Link
https://drive.google.com/file/d/14p1ekDW2YKqEY1kUufKjyJJTvlKVpEJZ/view?usp=sharing

You can install the dependencies using pip:
```bash
pip install tensorflow keras numpy pandas matplotlib opencv-python scikit-learn


