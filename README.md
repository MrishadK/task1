## Titanic Dataset - Data Cleaning & Preprocessing

##Objective
Clean and preprocess the Titanic dataset to make it suitable for machine learning.

## Steps Performed
1. Imported the Titanic dataset (offline CSV)
2. Handled missing values using mean/mode
3. Dropped the 'Cabin' column due to many nulls
4. Converted categorical features using One-Hot Encoding
5. Standardized numerical columns ('Age' and 'Fare') using `StandardScaler`
6. Visualized and removed outliers from 'Fare' using IQR method
7. Saved the cleaned dataset as `cleaned_titanic.csv`

## Files Included
- `Task1.py`
- `Titanic-Dataset.csv`
- `cleaned_titanic.csv`
