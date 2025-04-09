# 💳 Loan Default Risk Prediction  
**A Deep Learning-Based Binary Classifier for Financial Risk Assessment**

> Predicting whether a loan applicant is at risk of default using demographic, professional, and financial data.  
> Built using TensorFlow, Keras, and Scikit-learn.  
> [🔗 View Colab Notebook](https://colab.research.google.com/drive/1fGx5YdtYVK5bgiS6YLnGzHDs5bz6UywE)
---
## 🧠 Project Summary

This project builds a **binary classification model** to predict **loan default risk**, helping financial institutions automate the decision-making process when approving or rejecting loans. It uses a **multi-layer perceptron (MLP)** trained on user-specific attributes like marital status, car ownership, income, profession, and more.

---
## 🔍 Problem Statement

Given a dataset of applicants with both **numerical** and **categorical** features, determine whether the individual is a **potential loan defaulter (Risk_Flag = 1)** or a **safe borrower (Risk_Flag = 0)** using machine learning.

---
## 🚀 Key Features

| Feature Type     | Examples                                          |
|------------------|--------------------------------------------------|
| Demographic      | Age, Marital Status, Car Ownership               |
| Financial        | Monthly Income, Work Experience                  |
| Categorical Data | House Ownership, Profession, City, State         |
| Target Variable  | Risk_Flag (0 = Low Risk, 1 = High Risk)          |

---
## 📁 Dataset

- 📄 `Training Data.csv` (manually uploaded via Google Colab)
- ✅ Preprocessed with label encoding and one-hot encoding
- 📊 Features after encoding: Mix of numerical + categorical
- 🧮 Number of records: Assumed to be in the thousands

--- 
🧹 Data Preprocessing
- Removed unnecessary ID column as it doesn't contribute to model learning.
- Encoded binary categorical columns (Married/Single, Car_Ownership) using Label Encoding.
- Applied One-Hot Encoding to multi-category columns (House_Ownership, Profession, CITY, STATE) to convert them into numerical format suitable for machine learning.
- Separated features and labels: X contains the predictors, while y contains the target variable Risk_Flag.
- Standardized all numerical features using Z-score normalization to ensure uniform feature scaling.
- Split the dataset into training and testing sets using an 80-20 ratio for model evaluation.

---
🧠 Model Architecture (Deep Neural Network)
- This project uses a Multi-Layer Perceptron (MLP) built with TensorFlow/Keras for binary classification to predict loan default risk.
🔧 Architecture Overview
- Input Layer: Matches the number of features after preprocessing.
- Hidden Layers:
- Layer 1: 32 neurons, ReLU activation
- Layer 2: 16 neurons, ReLU activation
- These layers help capture complex patterns and interactions in the data.
- Output Layer: 1 neuron with sigmoid activation to output the probability of default (1 = high risk, 0 = low risk).
⚙️ Configuration
- Loss Function: Binary Crossentropy
- Optimizer: Adam
- Metric: Accuracy

---
🛠️ How to Use
- To run this project and evaluate the loan default prediction model, follow these steps:
1️⃣ Clone the Repository
 - Start by cloning the GitHub repository to your local machine. This will give you access to the Jupyter notebook and associated project files.
2️⃣ Upload the Dataset
 - The dataset file, named Training Data.csv, is required for training and evaluation.
 - If you're using Google Colab, use the file upload utility within the notebook to upload the dataset.
 - If you're using a local Jupyter Notebook, place the dataset file in the same directory as the notebook.
3️⃣ Install Required Libraries
 - Ensure that Python is installed on your machine. Then, install the necessary libraries, which include:
 - pandas and numpy for data manipulation
 - scikit-learn for preprocessing and model evaluation
 - tensorflow for building and training the neural network
 These libraries can be installed via pip (Python package manager).
4️⃣ Run the Notebook
 - Once everything is set up:
   - Open the notebook file (Loan_Default_Risk_Prediction.ipynb) in Jupyter or Google Colab.
   - Run all the cells sequentially:
   - This will preprocess the data
   - Build and train the deep learning model
   - Evaluate the model on a test set and display the accuracy

---

