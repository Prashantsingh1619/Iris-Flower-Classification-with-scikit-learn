# Iris Species Classification

This repository contains a Jupyter Notebook (`iris_classification.ipynb`) that demonstrates a machine learning pipeline for classifying iris flowers. The goal is to predict the species of an iris based on its sepal and petal measurements.

## Project Structure

- `iris_classification.ipynb`: The main Jupyter Notebook that contains all the code for data loading, exploration, model training, and evaluation.
- `README.md`: This file, which provides an overview of the project.

## Workflow

The notebook follows a standard machine learning workflow:

1.  **Data Loading**: The `scikit-learn` library's built-in Iris dataset is loaded into a pandas DataFrame.
2.  **Exploratory Data Analysis (EDA)**: The data is visualized to understand the relationships between the different features (sepal length, sepal width, petal length, and petal width) and the three iris species (`setosa`, `versicolor`, and `virginica`).
3.  **Model Training**:
    - The dataset is split into training and testing sets.
    - A **Logistic Regression** model is chosen for classification.
    - The model is trained on the training data.
4.  **Model Evaluation**:
    - The trained model is used to make predictions on the test set.
    - The model's performance is evaluated using a **classification report** and a **confusion matrix**.

## Getting Started

### Prerequisites

You will need to have Python and the following libraries installed:

-   `numpy`
-   `pandas`
-   `scikit-learn`
-   `matplotlib`
