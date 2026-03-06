import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle

# Load data - using the larger dataset
df = pd.read_csv("placement_1000.csv")
df = df.iloc[:, 1:]  # Remove index column

# Extract features and target
X = df.iloc[:, 0:2]
y = df.iloc[:, -1]

# Scale the values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the model
clf = LogisticRegression()
clf.fit(X_scaled, y)

# Save both model and scaler
pickle.dump(clf, open('model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

print("model.pkl and scaler.pkl have been saved successfully!")
