# Linear Regression Task
# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Load the dataset
# You can replace the URL/filepath with your local file
data = pd.read_csv('Housing.csv')  # Assume you downloaded and placed it in the same folder

# View the data
print(data.head())

# Preprocessing
# Convert categorical variables to numeric using One-Hot Encoding
data = pd.get_dummies(data, drop_first=True)

# Features and Target
X = data.drop('price', axis=1)
y = data['price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
r2 = metrics.r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Coefficients
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print("\nModel Coefficients:")
print(coefficients)

# Plotting the regression line for one feature (e.g., area)
plt.scatter(X_test['area'], y_test, color='blue', label='Actual Price')
plt.plot(X_test['area'], model.predict(X_test), color='red', linewidth=2, label='Predicted Price')
plt.xlabel('Area')
plt.ylabel('Price')
plt.title('Area vs Price')
plt.legend()
plt.show()
