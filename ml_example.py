"""
Machine Learning Example: Iris Flower Classification
Demonstrates: data loading, EDA, preprocessing, training, evaluation
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ── 1. Load Data ──────────────────────────────────────────────────────────────
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
class_names = iris.target_names

print("=== Dataset Overview ===")
print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
print(f"Classes: {class_names}")
print(f"Features: {feature_names}\n")

# ── 2. Exploratory Data Analysis ──────────────────────────────────────────────
print("=== Feature Statistics ===")
for i, name in enumerate(feature_names):
    print(f"{name:30s}  mean={X[:, i].mean():.2f}  std={X[:, i].std():.2f}")
print()

# ── 3. Split & Scale ──────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── 4. Train Models ───────────────────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=200, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(max_depth=4, random_state=42),
}

for name, model in models.items():
    # Decision tree doesn't need scaled features, but it works fine with them
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc    = accuracy_score(y_test, y_pred)

    print(f"=== {name} ===")
    print(f"Accuracy: {acc:.2%}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print()

# ── 5. Predict a New Sample ───────────────────────────────────────────────────
new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])   # typical setosa measurements
new_scaled = scaler.transform(new_sample)

print("=== Predict New Sample ===")
print(f"Input: {new_sample[0]}")
for name, model in models.items():
    pred  = model.predict(new_scaled)[0]
    proba = model.predict_proba(new_scaled)[0]
    print(f"{name}: {class_names[pred]}  "
          f"(confidence: {proba.max():.0%})")
