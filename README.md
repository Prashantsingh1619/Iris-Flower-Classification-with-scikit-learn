
Iris Species Classification
This Jupyter Notebook demonstrates a machine learning workflow for classifying iris flowers into their respective species using the Iris dataset and a Logistic Regression model.

Overview
The notebook is divided into four main sections:

Loading the data: The necessary libraries are imported and the Iris dataset is loaded into a pandas DataFrame.

Understanding the data: This section includes data visualization to help understand the relationship between different features, such as sepal and petal dimensions, and how they relate to the iris species.

Train/Test Split & Model Training: The dataset is split into training and testing sets, and a Logistic Regression model is trained using the training data.

Evaluation of Model: The trained model is evaluated on the test set, and its performance is assessed using a classification report and a confusion matrix.

Notebook Structure
Loading the data:

Imports libraries: numpy, pandas, matplotlib.pyplot, sklearn.datasets, sklearn.model_selection, sklearn.linear_model, sklearn.metrics.

Loads the Iris dataset using load_iris(as_frame=True).

Creates a pandas DataFrame and adds a species column for better readability.

Understanding the data:

Generates two scatter plots:

Sepal Length vs. Sepal Width

Petal Length vs. Petal Width

These plots show that petal dimensions are a better feature for distinguishing between species, particularly separating setosa from versicolor and virginica.

Train/Test Split & Model Training:

Separates the feature data (X) from the target labels (y).

Splits the data into a 70% training set and a 30% testing set using train_test_split with a random_state for reproducibility.

Initializes a LogisticRegression model and trains it using the X_train and y_train data.

Evaluation of Model:

The model predicts species for the X_test data.

A classification report is printed, showing the precision, recall, and f1-score for each species.

A confusion matrix is plotted to visualize the model's performance, showing the number of correct and incorrect predictions for each class.

Requirements
The code in this notebook was developed using Python and requires the following libraries:

numpy

pandas

scikit-learn

matplotlib

You can install these libraries using pip:

Bash

pip install numpy pandas scikit-learn matplotlib
How to Run
Clone this repository.

Ensure you have Jupyter Notebook installed. If not, you can install it via pip: pip install jupyter.

Navigate to the cloned repository's directory in your terminal.

Run the command jupyter notebook.

Open the iris_classification.ipynb file and execute the cells in order.
