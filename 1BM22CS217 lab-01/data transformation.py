import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer

# --- Diabetes Dataset ---

# Load the diabetes dataset
diabetes_file_path = '/content/Dataset of Diabetes .csv' 
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


# 2. Data Transformations:
# a. Min-Max Scaling
scaler_minmax = MinMaxScaler()
diabetes_df[numeric_cols] = scaler_minmax.fit_transform(diabetes_df[numeric_cols])

# b. Standard Scaling (create a separate copy)
diabetes_df_std = diabetes_df.copy() 
scaler_std = StandardScaler()
diabetes_df_std[numeric_cols] = scaler_std.fit_transform(diabetes_df_std[numeric_cols])



# --- Adult Income Dataset ---

# Load the adult income dataset
adult_file_path = '/content/adult.csv'  
adult_df = pd.read_csv(adult_file_path)

# 1. Handling Missing Values: Replace '?' with NaN and then impute
adult_df.replace('?', np.nan, inplace=True)
imputer = SimpleImputer(strategy='most_frequent')  
adult_df_imputed = pd.DataFrame(imputer.fit_transform(adult_df), columns=adult_df.columns)

# 2. Handling Categorical Data: One-hot encoding 
categorical_cols = adult_df_imputed.select_dtypes(include=['object']).columns.tolist() 
adult_df_encoded = pd.get_dummies(adult_df_imputed, columns=categorical_cols, drop_first=True)

# ... (previous code)

# 3. Data Transformations:
# Get numeric columns after encoding.
# Check if any numeric columns exist to avoid error
numeric_cols_adult = adult_df_encoded.select_dtypes(include=np.number).columns
# Initialize adult_df_encoded_std before the if statement 
# to an empty DataFrame if no numeric columns are found
adult_df_encoded_std = pd.DataFrame()  
if len(numeric_cols_adult) > 0:
    # a. Min-Max Scaling
    # ... (code for Min-Max scaling remains the same) ...

    # b. Standard Scaling (create a separate copy)
    adult_df_encoded_std = adult_df_encoded.copy()
    scaler_std_adult = StandardScaler()
    adult_df_encoded_std[numeric_cols_adult] = scaler_std_adult.fit_transform(adult_df_encoded_std[numeric_cols_adult])
else:
    print("No numeric columns found for scaling in adult income dataset.")
# ... (rest of the code)



# Print the preprocessed dataframes (examples)
print("Diabetes data with Min-Max scaling:\n", diabetes_df.head())
print("\nDiabetes data with Standard scaling:\n", diabetes_df_std.head())
print("\nAdult income data with Min-Max scaling:\n", adult_df_encoded.head())
print("\nAdult income data with Standard scaling:\n", adult_df_encoded_std.head())
