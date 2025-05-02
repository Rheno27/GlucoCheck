import joblib
import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Load model, scaler, dan RFE selector
model = joblib.load("../model/diabetes_model.sav")
scaler = joblib.load("../model/scaler.sav")
selector = joblib.load("../model/rfe_selector.sav")

st.set_page_config(
    page_title="DiabetaKu",
    layout="wide",
    page_icon="asset/diabetes_icon.png"
)

st.title('DiabetaKu : Aplikasi Prediksi Diabetes')
st.markdown("Masukkan informasi berikut untuk memprediksi apakah Anda berisiko terkena diabetes:")

# Input form
col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Jenis Kelamin", options=[0, 1], format_func=lambda x: "Perempuan" if x == 0 else "Laki-laki")
    age = st.number_input("Usia Anda", min_value=1, max_value=120, step=1)
    hypertension = st.selectbox("Hipertensi", options=[0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")
    heart_disease = st.selectbox("Penyakit Jantung", options=[0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")

with col2:
    smoking_history = st.selectbox("Riwayat Merokok", options=[-1, 0, 1, 2, 3, 4],
        format_func=lambda x: { -1: "Tidak Diketahui", 0: "Tidak Pernah", 1: "Mantan Perokok", 2: "Saat Ini", 3: "Bukan Saat Ini", 4: "Pernah" }.get(x, ""))
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, step=0.1)
    hba1c = st.number_input("Level HbA1c (Hemoglobin selama 3 bulan terakhir)", min_value=3.0, max_value=15.0, step=0.1)
    blood_glucose = st.number_input("Level Gula Darah ", min_value=50.0, max_value=500.0, step=1.0)

# Validasi input
input_ready = all(val is not None for val in [gender, age, hypertension, heart_disease, smoking_history, bmi, hba1c, blood_glucose])

# Proses prediksi
if st.button("Prediksi"):
    if not input_ready:
        st.error("Harap isi semua kolom input sebelum melakukan prediksi.")
    else:
        input_data = np.array([[gender, age, hypertension, heart_disease, smoking_history, bmi, hba1c, blood_glucose]])
        input_scaled = scaler.transform(input_data)
        input_selected = selector.transform(input_scaled)

        prediction = model.predict(input_selected)[0]
        diagnosis = "✅ Anda tidak terindikasi diabetes." if prediction == 0 else "⚠️ Anda terindikasi diabetes."
        st.success(f"Hasil: {diagnosis}")

        # Simpan hasil prediksi
        result_row = pd.DataFrame([[gender, age, hypertension, heart_disease, smoking_history, bmi, hba1c, blood_glucose, prediction]],
                                columns=["gender", "age", "hypertension", "heart_disease", "smoking_history", "bmi", "hba1c", "blood_glucose", "prediction"])

        if os.path.exists("riwayat_prediksi.csv"):
            result_row.to_csv("riwayat_prediksi.csv", mode='a', header=False, index=False)
        else:
            result_row.to_csv("riwayat_prediksi.csv", index=False)

        st.markdown("---")
        st.write("### Riwayat Prediksi")
        history = pd.read_csv("riwayat_prediksi.csv")
        st.dataframe(history.tail(5))

# Visualisasi data (jika dataset tersedia)
if os.path.exists("diabets_dataset_clean.csv"):
    df = pd.read_csv("diabets_dataset_clean.csv")
    st.markdown("---")
    st.write("### Visualisasi Dataset")

    col_a, col_b = st.columns(2)

    with col_a:
        st.write("Distribusi Diabetes")
        dist = df['diabetes'].value_counts()
        fig, ax = plt.subplots()
        ax.pie(dist, labels=["Tidak Diabetes", "Diabetes"], autopct="%1.1f%%", startangle=90, colors=["#6BAED6", "#FB6A4A"])
        ax.axis("equal")
        st.pyplot(fig)

    with col_b:
        st.write("Rata-rata BMI berdasarkan Status Diabetes")
        avg_bmi = df.groupby('diabetes')['bmi'].mean()
        fig, ax = plt.subplots()
        ax.bar(['Tidak Diabetes', 'Diabetes'], avg_bmi, color=["#74C476", "#EF3B2C"])
        ax.set_ylabel("Rata-rata BMI")
        st.pyplot(fig)
