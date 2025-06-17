
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, classification_report
import joblib


# Load dataset
df = pd.read_csv('dashboard/diabets_dataset_clean.csv')
X = df.drop(columns='diabetes', axis=1)
y = df['diabetes']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split tanpa RFE
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=2)

# Model training tanpa RFE
classifier_full = SVC(kernel='linear')
classifier_full.fit(X_train_full, y_train_full)

# Evaluasi akurasi
y_train_full_pred = classifier_full.predict(X_train_full)
y_test_full_pred = classifier_full.predict(X_test_full)

train_acc_full = accuracy_score(y_train_full, y_train_full_pred)
test_acc_full = accuracy_score(y_test_full, y_test_full_pred)

print("=== Evaluasi Sebelum RFE ===")
print("Train Accuracy (full):", train_acc_full)
print("Test Accuracy (full):", test_acc_full)

print("=== Classification Report Sebelum RFE ===")
print(classification_report(y_test_full, y_test_full_pred))

# Recursive Feature Elimination (select top 5 features)
svc_estimator = SVC(kernel='linear')
selector = RFE(estimator=svc_estimator, n_features_to_select=5)
X_selected = selector.fit_transform(X_scaled, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, stratify=y, random_state=2)

# Model training
classifier = SVC(kernel='linear')
classifier.fit(X_train, y_train)

# Evaluation
y_train_pred = classifier.predict(X_train)
y_test_pred = classifier.predict(X_test)
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("Train Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)

print("=== Classification Report Sesudah RFE ===")
print(classification_report(y_test, y_test_pred))

# Visualisasi Learning Curve untuk evaluasi overfitting dan underfitting
train_sizes, train_scores, test_scores = learning_curve(
    classifier, X_selected, y,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10),
    shuffle=True,
    random_state=42
)

# Hitung rata-rata skor
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)

# Plot learning curve
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label='Train Accuracy', marker='o')
plt.plot(train_sizes, test_mean, label='Validation Accuracy', marker='s')
plt.title('Learning Curve untuk Evaluasi Overfitting/Underfitting')
plt.xlabel('Jumlah Data Training')
plt.ylabel('Akurasi')
plt.grid(True)
plt.legend(loc='best')
plt.tight_layout()
plt.show()


# Save all components
joblib.dump(classifier, 'model/diabetes_model.sav')
joblib.dump(scaler, 'model/scaler.sav')
joblib.dump(selector, 'model/rfe_selector.sav')

# Optional: Print selected feature names
selected_features = X.columns[selector.support_]
print("Selected features:", selected_features.tolist())
