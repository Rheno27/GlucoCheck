import streamlit as st
import pandas as pd

# Judul Web
st.title('Sistem Prediksi Diabetes 🔥')

# Baca data CSV
data = pd.read_csv('diabetes_prediction_dataset.csv')

# Tampilkan data
st.write('### Data Diabetes')
st.dataframe(data)
