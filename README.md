# GlucoCheck

GlucoCheck adalah aplikasi prediksi risiko diabetes berbasis machine learning yang dibangun menggunakan Python dan Streamlit.

## 🆕 Pembaruan Terbaru

- **Visualisasi Data**: Menambahkan grafik distribusi untuk setiap fitur menggunakan Seaborn.
- **Evaluasi Model**: Menambahkan metrik evaluasi seperti precision, recall, dan F1-score.
- **Antarmuka Pengguna**: Memperbaiki tampilan aplikasi Streamlit untuk pengalaman pengguna yang lebih baik.

## 📂 Struktur Proyek

- `PREDIKSI_PENYAKIT_DIABETES.ipynb`: Notebook utama untuk eksplorasi data dan pelatihan model.
- `train_diabetes_model_with_rfe.py`: Skrip untuk pelatihan model menggunakan Recursive Feature Elimination (RFE).
- `stream_diabetes.py`: Aplikasi Streamlit untuk antarmuka pengguna.
- `dataset/`: Folder berisi data yang digunakan untuk pelatihan dan pengujian model.
- `model/`: Folder untuk menyimpan model yang telah dilatih.
- `requirements.txt`: Daftar dependensi yang diperlukan.

## 🚀 Cara Menjalankan Aplikasi

1. Klon repositori ini:
   ```bash
   git clone https://github.com/Rheno27/GlucoCheck.git
   cd GlucoCheck

2. Instal dependensi:
    ```bash
    pip install -r requirements.txt

3. Jalankan aplikasi Streamlit:
    ```bash
    streamlit run stream_diabetes.py

