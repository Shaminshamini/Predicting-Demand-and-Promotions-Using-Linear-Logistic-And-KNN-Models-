import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("sales_data.csv")

# Handle division by zero
data['Sold_Percentage'] = np.where(data['Units Ordered'] == 0, 0, data['Units Sold'] / data['Units Ordered'])

# Features
features = ['Sold_Percentage']
X = data[features]
y = data['Promotion'].fillna(0)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

#user input
units_sold = float(input("Enter Units Sold: "))
units_ordered = float(input("Enter Units Ordered: "))

if units_sold / units_ordered > 0.8:  # for example, sold >80% of ordered
    print("Prediction: Promotion is likely ACTIVE.")
else:
    print("Prediction: Promotion is likely NOT ACTIVE.")


