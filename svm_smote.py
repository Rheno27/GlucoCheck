import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score
from imblearn.over_sampling import SMOTE
import joblib

# Load the dataset
df = pd.read_csv('dataset/diabetes_prediction_dataset.csv')
df.head()

#Eksplorasi Awal & Visualisasi Sebelum Cleaning
plt.pie(df['diabetes'].value_counts(), labels=['Tidak Diabetes', 'Diabetes'], autopct='%1.2f%%')
plt.title("Distribusi Diabetes - Sebelum Cleaning")
plt.show()

#Cek missing values
df.isnull().sum()

# Cek duplikat
df.duplicated().sum()

# Cek tipe data
df.drop_duplicates(inplace=True)
df.duplicated().sum()
df.info()

#Visualisasi Setelah Pembersihan
plt.figure(figsize=(6, 6))
plt.pie(df['diabetes'].value_counts(), labels=['Tidak Diabetes', 'Diabetes'], autopct='%1.2f%%')
plt.title("Distribusi Diabetes - Setelah Cleaning")
plt.show()

#EXPLANATORY DATA ANALYSIS
print("Jumlah data yang memiliki diabetes:", df[df['diabetes'] == 1].shape[0])
plt.figure(figsize = [15,15])

plt.subplot(1,2,1)
plt.pie(df[df['hypertension'] == 1]['diabetes'].value_counts().values, labels=['Tidak Diabetes', 'Diabetes'], autopct='%1.2f%%', startangle=45)
plt.title('Memiliki Hipertensi')

plt.subplot(1,2,2)
plt.pie(df[df['hypertension'] == 0]['diabetes'].value_counts().values, labels=['Tidak Diabetes', 'Diabetes'], autopct='%1.2f%%', startangle=45)
plt.title("Tidak Memiliki Hipertensi")

plt.subplots_adjust(wspace=0.4)
plt.show()

print("Jumlah data yang memiliki hipertensi:", df[df['hypertension'] == 1].shape[0])
plt.figure(figsize = [15,15])

plt.subplot(1,2,1)
plt.pie(df[df['heart_disease'] == 1]['diabetes'].value_counts().values, labels=['Tidak Diabetes', 'Diabetes'], autopct='%1.2f%%', startangle=45)
plt.title('Memiliki Penyakit Jantung')


plt.subplot(1,2,2)
plt.pie(df[df['heart_disease'] == 0]['diabetes'].value_counts().values, labels=['Tidak Diabetes', 'Diabetes'], autopct='%1.2f%%', startangle=45)
plt.title("Tidak Memiliki Penyakit Jantung")

plt.subplots_adjust(wspace=0.4)
plt.show()

non_diabetes_counts = df[df['diabetes'] == 0]['smoking_history'].value_counts()
sorted_categories = non_diabetes_counts.index

plt.figure(figsize=(10, 6))
sns.countplot(
    x='smoking_history',
    hue='diabetes',
    data=df,
    order=sorted_categories)
plt.title('Perbandingan Riwayat Merokok Terhadap Hasil Diabetes')
plt.xlabel('Riwayat Merokok')
plt.ylabel('Jumlah')
plt.legend(title='Diabetes', loc='upper right', labels=['Tidak Diabetes', 'Diabetes'])
plt.grid(axis='y')
plt.show()

tipe_bmi = []

for tipe in df['bmi']:
    if tipe <= 18.5:
        tipe_bmi.append('underweight')
    elif(tipe > 18.5 and tipe <= 24.9):
        tipe_bmi.append('normal')
    elif(tipe > 24.9 and tipe <=29.9):
        tipe_bmi.append('overweight')
    else :
        tipe_bmi.append('obesity')

df['tipe_bmi'] = tipe_bmi

plt.figure(figsize=(10, 6))
sns.countplot(
    x='tipe_bmi',
    hue='diabetes',
    data=df)
plt.title('Perbandingan Jenis BMI terhadap Hasil Diabetes')
plt.xlabel('Jenis BMI')
plt.ylabel('Jumlah')
plt.legend(title='Diabetes', loc='upper right', labels=['Tidak Diabetes', 'Diabetes'])
plt.grid(axis='y')
plt.show()

blood_glucose = []

for level in df['blood_glucose_level']:
    if level <= 99:
        blood_glucose.append('normal')
    elif (level > 99) and (level <=125):
        blood_glucose.append('prediabetes')
    else :
        blood_glucose.append('diabetes')

df['blood_glucose_test'] = blood_glucose

plt.figure(figsize=(10, 6))
sns.countplot(
    x='blood_glucose_test',
    hue='diabetes',
    data=df
)
plt.title('Perbandingan Level Gula Darah terhadap Hasil Diabetes')
plt.xlabel('Kategori Gula Darah')
plt.ylabel('Jumlah')
plt.legend(title='Diabetes', loc='upper right', labels=['Tidak Diabetes', 'Diabetes'])
plt.grid(axis='y')
plt.show()

# Data Cleaning
df.drop(columns='tipe_bmi', inplace=True)
df.drop(columns='blood_glucose_test', inplace=True)

#Data Cleaning & Encoding
df = df.dropna() 
le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])
df['smoking_history'] = le.fit_transform(df['smoking_history'])

#Split Feature & Target
X = df.drop('diabetes', axis=1)
y = df['diabetes']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

#Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#SMOTE Oversampling
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

#Pie Chart Setelah SMOTE
plt.figure(figsize=(5, 5))
plt.pie(pd.Series(y_train_resampled).value_counts().values, labels=['Tidak Diabetes', 'Diabetes'], autopct='%1.2f%%', colors=['#66b3ff','#ff9999'])
plt.title('Perbandingan Kelas Setelah SMOTE')
plt.legend()
plt.show()

# Train SVM Classifier with SMOTE
classifier = SVC(kernel='linear', random_state=42)
classifier.fit(X_train_resampled, y_train_resampled)

# evaluasi model
# Evaluasi training pakai data hasil SMOTE
y_train_pred = classifier.predict(X_train_resampled)
print("Train Accuracy:", accuracy_score(y_train_resampled, y_train_pred))

# SMOTE untuk test set agar seimbang (hanya untuk evaluasi, tidak untuk prediksi nyata)
X_test_resampled, y_test_resampled = smote.fit_resample(X_test_scaled, y_test)
y_test_pred = classifier.predict(X_test_resampled)
print("Test Accuracy:", accuracy_score(y_test_resampled, y_test_pred))

print("F1 Score:", f1_score(y_test_resampled, y_test_pred))
print("Recall:", recall_score(y_test_resampled, y_test_pred))
print("Classification Report:\n", classification_report(y_test_resampled, y_test_pred))

sns.heatmap(confusion_matrix(y_test_resampled, y_test_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - SVM + SMOTE (Resampled Test Data)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# Learning Curve
train_sizes, train_scores, test_scores = learning_curve(
    classifier,
    X_train_resampled, y_train_resampled,   
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 5),
    shuffle=True,
    random_state=42
)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, np.mean(train_scores, axis=1), marker='o', label='Train Accuracy')
plt.plot(train_sizes, np.mean(test_scores, axis=1), marker='s', label='Validation Accuracy')
plt.title('Learning Curve - SVM + SMOTE')
plt.xlabel('Jumlah Data Training (Resampled)')
plt.ylabel('Akurasi')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Simpan model dan scaler
joblib.dump(classifier, 'model/diabetes_model_svm_smote.sav')
joblib.dump(scaler, 'model/scaler_svm_smote.sav')

# Simpan dataset bersih
df_clean = df.copy()
df_clean.to_csv('dashboard/diabetes_dataset_clean.csv', index=False)
print("✅ Dataset clean berhasil disimpan di: dashboard/diabetes_dataset_clean.csv")
