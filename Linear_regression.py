


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load data
data = pd.read_csv("sales_data.csv")

# Fill missing values
data['Price'] = data['Price'].fillna(0)
data['Demand'] = data['Demand'].fillna(0)

# Simple regression: Demand vs Price
X = data[['Price']]  # only Price
y = data['Demand']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict for test data
y_pred = model.predict(X_test)

# Print accuracy


# User input
price = float(input("Enter Price: "))
new_data = pd.DataFrame([[price]], columns=['Price'])
predicted_demand = model.predict(new_data)
print("Predicted Demand:", predicted_demand[0])

# Plot regression line
plt.figure(figsize=(8, 4))

x_line = np.linspace(X['Price'].min(), X['Price'].max(), 100).reshape(-1, 1)
y_line = model.predict(x_line)
plt.plot(x_line, y_line, color='red', linewidth=2, label="Regression Line")
plt.xlabel("Price")
plt.ylabel("Demand")
plt.title("Price vs Demand with Regression Line")
plt.legend()
plt.grid(True)
plt.show()
