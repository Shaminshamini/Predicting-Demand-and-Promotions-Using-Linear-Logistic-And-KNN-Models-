import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load cleaned dataset
df = pd.read_csv("cleaned_sales_data.csv")

# Target: Promotion (already encoded)
y = df['Promotion']

# Features
features = ['Demand', 'Units Ordered', 'Units Sold', 'Price', 'Discount', 'Epidemic']
X = df[features]

# Fill missing values if any (extra safety)
X = X.fillna(0)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# KNN model with fixed k (for example, k=5)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predict and evaluate
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
