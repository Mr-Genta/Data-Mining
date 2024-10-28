import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import pickle
from sklearn.naive_bayes import GaussianNB

st.write("""
# Klasifikasi Obat (Web Apps)
Aplikasi berbasis Web untuk memprediksi jenis obat.
Data yang diambil dari Kaggle.
""")

# Menampilkan gambar
img = Image.open('cat.jpg')
img = img.resize((700, 478))
st.image(img, use_column_width=False)

img2 = Image.open('pct2.jpg')
img2 = img2.resize((700, 451))
st.image(img2, use_column_width=False)

st.sidebar.header('Parameter Inputan')

# Upload File CSV untuk parameter inputan
upload_file = st.sidebar.file_uploader('Upload file CSV Anda', type=["csv"])
if upload_file is not None:
    inputan = pd.read_csv(upload_file)
else:
    def input_user():
        Jenis_Kelamin = st.sidebar.selectbox('Jenis Kelamin', ('F', 'M'))
        tekanan_darah = st.sidebar.selectbox('Tekanan Darah', ('HIGH', 'NORMAL', 'LOW'))
        Cholesterol = st.sidebar.selectbox('Cholesterol', ('HIGH', 'NORMAL'))
        Umur = st.sidebar.slider('Umur (Tahun)', 15, 74, 20)
        Rasio_Natrium_terhadap_Kalium = st.sidebar.slider('Rasio Natrium terhadap Kalium (gram)', 6.2, 38.2, 25.3)
        data = {
            'Jenis_Kelamin': Jenis_Kelamin,
            'tekanan_darah': tekanan_darah,
            'Cholesterol': Cholesterol,
            'Umur': Umur,
            'Rasio_Natrium_terhadap_Kalium': Rasio_Natrium_terhadap_Kalium
        }
        fitur = pd.DataFrame(data, index=[0])
        return fitur

    inputan = input_user()

# Menghubungkan inputan dan dataset obat
obat_raw = pd.read_csv('drug200.csv', sep=';')
obat = obat_raw.drop(columns=['obat'], errors='ignore')
df = pd.concat([inputan, obat], axis=0)

# Encode untuk atribut kategori
encode = ['Jenis_Kelamin', 'tekanan_darah', 'Cholesterol']
df = pd.get_dummies(df, columns=encode)

# Pastikan kolom sesuai dengan kolom model
load_model = pickle.load(open('modelNBC_obat.pkl', 'rb'))
model_features = load_model.feature_names_in_

# Tambahkan kolom yang hilang
missing_cols = set(model_features) - set(df.columns)
for col in missing_cols:
    df[col] = 0

# Urutkan kolom agar sesuai dengan model
df = df[model_features]
df = df[:1]  # Ambil baris pertama (input data user)

# Menampilkan parameter inputan
st.subheader('Parameter Inputan')

if upload_file is not None:
    st.write(df)
else:
    st.write('Menunggu file CSV untuk diupload. Saat ini menggunakan sampel inputan (seperti tampilan di bawah)')
    st.write(df)

# Terapkan model Naive Bayes
prediksi = load_model.predict(df)
prediksi_proba = load_model.predict_proba(df)

st.subheader('Keterangan Label Obat')
jenis_obat = np.array(['drugA', 'drugB', 'drugC', 'drugX', 'drugY'])
st.write(jenis_obat)

st.subheader('Hasil Prediksi (Klasifikasi Obat)')
st.write(jenis_obat[prediksi])

st.subheader('Probabilitas Hasil Prediksi (Klasifikasi Obat)')
st.write(prediksi_proba)
