import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# --- Diabetes Dataset ---

# Load the diabetes dataset
diabetes_file_path = '/content/Dataset of Diabetes .csv'  # Replace with your actual file path
diabetes_df = pd.read_csv(diabetes_file_path)

# 1. Handling Missing Values: 
# a. Identify numeric and categorical columns
numeric_cols = diabetes_df.select_dtypes(include=np.number).columns
categorical_cols = diabetes_df.select_dtypes(exclude=np.number).columns

# b. Impute missing values using the mean for numeric columns only
imputer_numeric = SimpleImputer(strategy='mean')  
diabetes_df[numeric_cols] = imputer_numeric.fit_transform(diabetes_df[numeric_cols])

# c. Impute missing values using the most frequent value for categorical columns
imputer_categorical = SimpleImputer(strategy='most_frequent')
diabetes_df[categorical_cols] = imputer_categorical.fit_transform(diabetes_df[categorical_cols])


# --- Adult Income Dataset ---

# Load the adult income dataset
adult_file_path = '/content/adult.csv'  # Replace with your actual file path
adult_df = pd.read_csv(adult_file_path)

# 1. Handling Missing Values: Replace '?' with NaN and then impute
adult_df.replace('?', np.nan, inplace=True)
imputer = SimpleImputer(strategy='most_frequent')  # Use most frequent for categorical features

# Fit and transform, but keep column names using columns=adult_df.columns
adult_df_imputed = pd.DataFrame(imputer.fit_transform(adult_df), columns=adult_df.columns)

# 2. Handling Categorical Data: One-hot encoding for adult income dataset
# Ensure categorical_cols are present in adult_df_imputed
categorical_cols = adult_df_imputed.select_dtypes(include=['object']).columns.tolist() 

adult_df_encoded = pd.get_dummies(adult_df_imputed, columns=categorical_cols, drop_first=True)

# --- Handling Outliers (for both datasets) ---
# (Example using Z-score - adjust as needed)
# from scipy import stats
# z = np.abs(stats.zscore(diabetes_df_imputed))  # For diabetes dataset
# diabetes_df_no_outliers = diabetes_df_imputed[(z < 3).all(axis=1)]
# z = np.abs(stats.zscore(adult_df_encoded.select_dtypes(include=np.number)))  # For adult dataset
# adult_df_no_outliers = adult_df_encoded[(z < 3).all(axis=1)]

# Print the preprocessed dataframes (example)
print("Diabetes data with imputed missing values:\n", diabetes_df.head())
print("\nAdult income data with imputed values and encoding:\n", adult_df_encoded.head())
