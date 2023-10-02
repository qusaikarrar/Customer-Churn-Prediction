# Customer Churn Prediction

![Churn Prediction](https://miro.medium.com/v2/resize:fit:844/1*MyKDLRda6yHGR_8kgVvckg.png)

## Overview

This project focuses on predicting customer churn using a neural network-based machine learning model. Customer churn, also known as customer attrition, is a vital business metric that gauges the rate at which customers leave or discontinue using a service. Accurate churn prediction allows businesses to proactively identify at-risk customers and implement strategies to retain them. Leveraging data on customer demographics, geographical information, and banking-related metrics, this project aims to build a predictive model that helps businesses optimize their customer retention efforts. By utilizing state-of-the-art techniques in deep learning and data preprocessing, we aim to provide a valuable tool for businesses seeking to reduce churn rates and enhance customer satisfaction.


## Table of Contents

- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Data Preparation](#data-preparation)
- [Model Building](#model-building)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)

## Dataset

We use the Churn_Modelling dataset, which contains customer information, including geographical and demographic data, as well as banking-related metrics. The goal is to predict whether a customer will churn or not based on this data.

The dataset used for this project can be found on Kaggle: [Churn_Modelling Dataset](https://www.kaggle.com/datasets/shubh0799/churn-modelling/data).

Please download the dataset and place it in the project directory before running the code.


## Installation

1. Clone the repository:

```bash
git clone https://github.com/qusaikarrar/churn-prediction.git
cd churn-prediction
```

## Usage

To use this project, follow these steps:

1. Make sure you have completed the installation steps.

2. Download the dataset from Kaggle and place it in the project directory.

3. Open the Jupyter Notebook (`Customer_Churn_Prediction.ipynb`) to view and run the code.

## Data Preparation

In this section, we outline the data preparation steps performed before building and training the predictive model.

- **Data Cleaning:** We check for missing values, outliers, and any data inconsistencies. If necessary, we perform data cleaning operations to ensure the dataset's quality.

- **Feature Engineering:** We engineer new features or transform existing ones to improve the model's performance. This may include creating derived features or encoding categorical variables.

- **Train-Test Split:** We split the dataset into training and testing subsets to evaluate the model's performance on unseen data. We typically use an 80-20 or 70-30 split ratio.

- **Data Standardization:** To ensure consistency, we standardize or normalize the data, bringing all features to a common scale, which is essential for neural network models.

These data preparation steps are crucial to ensure the data is suitable for training and evaluating the machine learning model. The details of each step can be found in the project's Jupyter Notebook (`Customer_Churn_Prediction.ipynb`).

## Model Building

In this section, we describe the architecture and components of the neural network model used for predicting customer churn.

- **Model Architecture:** We build a deep neural network with multiple layers, including input, hidden, and output layers.

- **Loss Function:** We use binary cross-entropy as the loss function and optimize the model using the Adam optimizer.

- **Regularization:** Dropout and batch normalization layers are added to prevent overfitting and improve model generalization.

## Model Training

We train the model on the prepared dataset and monitor its performance during training. The training process includes epochs, loss, and accuracy metrics.

## Evaluation

We evaluate the trained model's performance using various metrics, including accuracy and loss on the validation set. The model's summary and architecture are also provided.


## Model Summary


**Model Architecture:**
- Input Layer: 10 neurons
- Hidden Layers: Multiple dense layers with 256, 256, and 128 neurons, respectively.
- Output Layer: 2 neurons (binary classification)
- Activation Function: ReLU (Rectified Linear Unit) for hidden layers and Sigmoid for the output layer.

**Training Metrics:**
- Loss Function: Binary Cross-Entropy
- Optimizer: Adam
- Regularization: Dropout and Batch Normalization

The model was trained for 100 epochs and achieved an accuracy of 86.72% on the validation set.

For a detailed model summary with layer shapes and parameters, refer to the Jupyter Notebook (`Customer_Churn_Prediction.ipynb`).


Total params: 101912 (398.09 KB)
Trainable params: 101892 (398.02 KB)
Non-trainable params: 20 (80.00 Byte)
Accuracy: 86.72%


## Contributing
Feel free to contribute to this project, discuss potential future improvements, additional features, or optimizations that could enhance the model's performance or usability.

## License
This project is free licensed.


