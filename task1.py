import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('Titanic-Dataset.csv')

print("Head of the dataset:\n", df.head())
print("\n Info:\n")
print(df.info())
print("\n Description:\n", df.describe())
print("\n Missing values:\n", df.isnull().sum())

df.drop(columns=['Cabin'], inplace=True)

df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
print("\n After handling missing values:\n", df.isnull().sum())
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

plt.figure(figsize=(10, 5))
sns.boxplot(x=df['Fare'])
plt.title("Boxplot - Outliers in Fare")
plt.show()

Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['Fare'] >= Q1 - 1.5 * IQR) & (df['Fare'] <= Q3 + 1.5 * IQR)]

df.to_csv('cleaned_titanic.csv', index=False)
print("Cleaned dataset saved as cleaned_titanic.csv")
