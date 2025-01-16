# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load your custom housing dataset
data = pd.read_csv('Housing.csv')  # Replace with your file path

# Verify dataset structure
print(data.head())
print(data.info())

# Replace with actual feature and target column names from your dataset
X = data[['area', 'bedrooms','bathrooms']]  # Replace with your dataset's feature columns
y = data['price']  # Replace with your dataset's target column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Visualize Actual vs Predicted values
plt.scatter(y_test, y_pred, color='green')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Prices')
plt.show()

# Visualize Regression Line for a single feature (if applicable)
plt.scatter(X_test['area'], y_test, color='blue', label='Actual')  # Replace feature1 with a column
plt.scatter(X_test['area'], y_pred, color='red', label='Predicted')
plt.xlabel('AREA')
plt.ylabel('Target Column')
plt.title('Regression Line: Actual vs Predicted')
plt.legend()
plt.show()
