# K-Nearest Neighbors (KNN) Classifier

This repository contains a notebook that implements the **K-Nearest Neighbors (KNN)** algorithm for classifying data based on two features: age and estimated salary. The goal is to predict whether a user will purchase a product based on these features.

## Project Overview

K-Nearest Neighbors (KNN) is a simple, yet powerful, supervised machine learning algorithm. This project demonstrates how KNN can be used for classification tasks. The algorithm is trained on a dataset, and then it predicts the class of a new data point based on the majority vote of its nearest neighbors.

## Dataset

The dataset used in this project includes two key features:
- **Age**
- **Estimated Salary**

The target variable represents whether a user has purchased a product (binary classification).

## Workflow

1. **Data Preprocessing**:
   - The dataset is loaded and divided into independent variables (`X`) and the target variable (`y`).
   - The dataset is split into training and test sets.
   - Feature scaling is applied using `StandardScaler` to normalize the age and salary values.

2. **Model Building**:
   - A **K-Nearest Neighbors (KNN)** classifier is built using the `KNeighborsClassifier` from the Scikit-learn library.
   - The model is trained on the scaled training data.

3. **Model Evaluation**:
   - Predictions are made on the test set, and the performance is evaluated using a **confusion matrix**.
   - The confusion matrix helps in understanding the accuracy, precision, recall, and overall performance of the classifier.

4. **Visualization**:
   - Decision boundaries are plotted using Matplotlib to show how the KNN model classifies the test set data.
   - Scatter plots depict the test data points, with colors indicating the predicted classes.

## Results

The performance of the KNN classifier is visualized with a confusion matrix and a graph of decision boundaries for the test set. The accuracy of the model depends on the number of neighbors (`k`), which can be fine-tuned for optimal results.

### Confusion Matrix
The confusion matrix is used to summarize the performance of the classifier.

### Test Set Visualization
A decision boundary is plotted to show the regions where the classifier predicts each class. Test data points are plotted as red or green, depending on the actual class labels.


