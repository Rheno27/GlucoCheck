
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv('dashboard/diabets_dataset_clean.csv')
X = df.drop(columns='diabetes', axis=1)
y = df['diabetes']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

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

# Save all components
joblib.dump(classifier, 'model/diabetes_model.sav')
joblib.dump(scaler, 'model/scaler.sav')
joblib.dump(selector, 'model/rfe_selector.sav')

# Optional: Print selected feature names
selected_features = X.columns[selector.support_]
print("Selected features:", selected_features.tolist())
