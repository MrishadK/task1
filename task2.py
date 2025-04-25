import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("Titanic-Dataset.csv")

print("Data Info:")
print(df.info())
print("\n Descriptive Statistics:")
print(df.describe(include='all'))

print("\n Missing Values:\n", df.isnull().sum())

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[numeric_cols].hist(bins=20, figsize=(15, 10), color='skyblue')
plt.suptitle("Histograms of Numeric Features")
plt.show()

for col in numeric_cols:
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot for {col}")
    plt.show()

corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


print("\n🔹 Survival Rate by Gender:")
if 'Sex' in df.columns and 'Survived' in df.columns:
    print(df.groupby("Sex")["Survived"].mean())

print("\n🔹 Survival Rate by Pclass:")
if 'Pclass' in df.columns and 'Survived' in df.columns:
    print(df.groupby("Pclass")["Survived"].mean())