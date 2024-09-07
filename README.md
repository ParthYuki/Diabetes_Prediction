# Diabetes Prediction Model

## Overview

This repository contains a machine learning project for predicting diabetes outcomes based on various health features. The project uses a deep learning model built with Keras and TensorFlow to classify whether a patient has diabetes or not based on their medical data.

## Dataset

The dataset used for this project is the **Diabetes Dataset**. It includes the following features:

- **Pregnancies**: Number of pregnancies
- **Glucose**: Plasma glucose concentration after 2 hours in an oral glucose tolerance test
- **BloodPressure**: Diastolic blood pressure (mm Hg)
- **SkinThickness**: Triceps skin fold thickness (mm)
- **Insulin**: 2-Hour serum insulin (mu U/ml)
- **BMI**: Body mass index (weight in kg/(height in m)^2)
- **DiabetesPedigreeFunction**: Diabetes pedigree function
- **Age**: Age (years)
- **Outcome**: Class label (0 or 1, where 1 indicates diabetes)

The dataset file is named `diabetes.csv`.

## Installation

Ensure you have the following packages installed:

- pandas
- numpy
- scikit-learn
- imbalanced-learn
- tensorflow (keras)

You can install the required packages using pip:

```bash
pip install pandas numpy scikit-learn imbalanced-learn tensorflow
``` 
## Usage
1. Load the Dataset: Load the diabetes.csv file into your notebook.  
2. Feature Scaling: Apply feature scaling to normalize the data.  
3. Train-Test Split: Split the data into training and testing sets.  
4. Handle Class Imbalance: Use SMOTE to resample the training data to address class imbalance.  
5. Compute Class Weights: Calculate class weights to handle imbalance during training.  
6. Define and Train the Model: Use the provided notebook to define and train a deep learning model with optimizations.  
7. Evaluate the Model: Evaluate the model on the test set and obtain performance metrics.  
8. Find Optimal Threshold: Apply the optimal threshold for classification to improve metrics.  

## Results
The model achieved the following metrics:  
• Accuracy: Approximately 0.77  
• Precision, Recall, F1-Score: Detailed in the classification report  
## Files in This Repository
• diabetes.csv: The dataset used for training and evaluating the model.  
•Diabetes_Prediction_Model.ipynb: Jupyter Notebook containing the code for data processing, model training, and evaluation.  
## Future Work
• Hyperparameter Tuning: Explore different hyperparameters to further improve model performance.  
• Feature Engineering: Investigate additional feature engineering techniques.  
• Model Comparison: Compare with other models or approaches for diabetes prediction.  
