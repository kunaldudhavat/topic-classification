# Topic Classification with Linear Classifier

## Introduction

This repository contains my project on training a linear classifier for topic classification. The goal of this project is to classify documents into different topics using a linear model. The implementation is done in Python using Jupyter Notebook.

## Table of Contents

- [Introduction](#introduction)
- [Project Description](#project-description)
- [Dataset](#dataset)
- [Approach](#approach)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)


## Project Description

### Linear Classifier

The linear classifier in this project has parameters `W` (weight matrix) and `b` (bias vector), where `d` is the dimension of the input and `K` is the number of label types. Given a document represented as `x`, the model defines a distribution over `{1, ..., K}` using the softmax function:

<p align="center">
  <img src="https://github.com/kunaldudhavat/topic-classification/blob/main/images/softmax.png" alt="softmax" title="Softmax">
</p>

where `w_y` is the `y`-th column of `W`.

### Training

The training objective is to minimize the cross-entropy loss between the true labels and the predicted labels. This is achieved using stochastic gradient descent (SGD).

## Dataset

The dataset used in this project is the AG News dataset. A version of AG is made easily available by huggingface, so we will use that. Since this dataset comes in only a train-test split, we will create a validation set by setting aside a random subset of the training portion. The dataset consists of documents labeled with different topics. 

## Approach

The approach taken in this project includes the following steps:

1. **Data Preprocessing**: Loading and preprocessing the dataset, including text cleaning and vectorization.
2. **Model Implementation**: Implementing a linear classifier with parameters `W` and `b`.
3. **Gradient Calculation**: Implementing the code for accumulating gradients required for model training.
4. **SGD Optimizer**: Implementing the stochastic gradient descent (SGD) optimizer for updating model parameters.
5. **Hyperparameter Tuning**: Tuning hyperparameters such as learning rate and batch size to optimize model performance.
6. **Model Evaluation**: Evaluating the trained model on the test set and visualizing the results.

## Requirements

- Python 3.8+
- Jupyter Notebook
- NumPy
- Matplotlib
- Pandas
- Scikit-learn

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/topic-classification.git
   cd topic-classification

## Usage

1. Running the notebook:
   ```sh
   jupyter notebook
  
2. Open the Topic-classification.ipynb file and run all the cells


## Contributing
Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.
