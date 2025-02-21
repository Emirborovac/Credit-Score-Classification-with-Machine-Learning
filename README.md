# Credit Score Prediction System

## Overview
This project implements a machine learning system for predicting credit scores using Random Forest classification. The system analyzes various financial metrics to categorize credit scores into multiple classes, providing a robust tool for financial risk assessment.

## Project Structure
```
Credit-Score-Data/
├── Credit Score Data/
│   ├── train.csv
│   ├── processed/
│   │   ├── X_train.npy
│   │   ├── X_test.npy
│   │   ├── y_train.npy
│   │   ├── y_test.npy
│   │   ├── train_split.csv
│   │   └── test_split.csv
│   └── models/
│       └── random_forest_model.pkl
└── scripts/
    ├── 1_Data_Inspection.py
    ├── 2_Feature_Exploration.py
    ├── 3_Data_Preprocessing.py
    ├── 4_Training.py
    ├── 5_testdrive.py
    └── 6_Accuracy.py
```

## Technical Requirements
- Python 3.8+
- Dependencies:
  - numpy
  - pandas
  - scikit-learn
  - seaborn
  - matplotlib
  - panel
  - plotly
  - logging

## Installation & Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/Emirborovac/Credit-Score-Classification-with-Machine-Learning
   cd Credit-Score-Classification-with-Machine-Learning
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage Guide

### 1. Data Inspection
Run initial data analysis to understand the dataset structure:
```bash
python 1_Data_Inspection.py
```
This script provides:
- Dataset overview
- Data shape and structure
- Null value analysis
- Target variable distribution

### 2. Feature Exploration
Analyze feature relationships and importance:
```bash
python 2_Feature_Exploration.py
```
Launches an interactive dashboard for exploring:
- Feature distributions
- Correlation analysis
- Credit score relationships with financial metrics

### 3. Data Preprocessing
Prepare data for model training:
```bash
python 3_Data_Preprocessing.py
```
Performs:
- Categorical variable encoding
- Feature scaling
- Train/test split generation
- Processed data storage

### 4. Model Training
Train the Random Forest classifier:
```bash
python 4_Training.py
```
Includes:
- Model initialization
- Cross-validation
- Model persistence

### 5. Manual Testing
Test the model with custom inputs:
```bash
python 5_testdrive.py
```
Features:
- Interactive input interface
- Real-time predictions
- Multiple prediction capability

### 6. Model Evaluation
Assess model performance:
```bash
python 6_Accuracy.py
```
Provides:
- Classification metrics
- Confusion matrix
- Feature importance analysis
- Performance visualizations

## Model Features
The system analyzes the following financial metrics:
- Annual Income
- Monthly Inhand Salary
- Number of Bank Accounts
- Number of Credit Cards
- Interest Rate
- Number of Loans
- Payment Delay Statistics
- Outstanding Debt
- Credit History Age
- Monthly Balance

## Performance Metrics
The model's performance is evaluated using:
- Classification Accuracy
- Precision, Recall, F1-Score
- Class-wise Performance Analysis
- Feature Importance Rankings

## Best Practices & Notes
1. Always run scripts in numerical order for proper setup
2. Regularly backup the trained model
3. Monitor system logs for potential issues
4. Validate input data quality before predictions
5. Periodically retrain the model with new data

