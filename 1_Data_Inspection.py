import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "plotly_white"

# Read the dataset
data = pd.read_csv("./Credit-Score-Data/Credit Score Data/train.csv")

# Display the first few rows of the dataset
print("\n" + "="*50)
print("First 5 rows of the dataset:\n")
print(data.head())
print("="*50 + "\n")

# Display dataset structure and information
print("\n" + "="*50)
print("Dataset Information:\n")
data.info()
print("="*50 + "\n")

# Check for null values in the dataset
print("\n" + "="*50)
print("Null Values in Each Column:\n")
print(data.isnull().sum())
print("="*50 + "\n")

# Check the distribution of the target variable
print("\n" + "="*50)
print("Target Variable ('Credit_Score') Distribution:\n")
print(data["Credit_Score"].value_counts())
print("="*50 + "\n")
