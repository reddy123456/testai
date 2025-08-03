import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib

# Training data: [heart_rate, spo2, temperature]
X = np.array([
    [70, 98, 98.6],
    [110, 89, 102],
    [65, 97, 98.2],
    [120, 90, 101.5],
    [75, 99, 98.3]
])
y = np.array([0, 1, 0, 1, 0])  # 1 = risk

model = LogisticRegression()
model.fit(X, y)

joblib.dump(model, 'risk_model.pkl')
print("Model trained and saved as risk_model.pkl")