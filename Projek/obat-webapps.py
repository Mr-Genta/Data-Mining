import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import pickle
from sklearn.naive_bayes import GaussianNB


st.write("""
# Klasifikasi Obat (Web Apps)
Aplikasi berbasis Web untuk mempresiksi jenis obat \n
Data yang didapta dari Kagle.        
""")

img = Image.open('cat.jpg')
img = img.resize((700,478))
st.image(img, use_column_width=False)

img2 = Image.open('pct2.jpg')
img2 = img2.resize((700,451))
st.image(img2, use_column_width=False)

st.sidebar.header('parameter inputan')

# Uplode FIle CSV untuk parameter inputan
upload_file = st.sidebar.file_uploader('Upload dile CSV Anda', type=["csv"])
if upload_file is not None:
    inputan = pd.read_csv(upload_file)
else:
    def input_user():
        Jenis_Kelamin = st.sidebar.selectbox('Jenis Kelamin',('F','M'))
        tekanan_darah = st.sidebar.selectbox('tekanan darah',('HIGH','NORMAL','LOW'))
        Cholesterol = st.sidebar.selectbox('Cholesterol',('HIGH','NORMAL'))
        Umur = st.sidebar.slider('Umur (Tahun)',15,74,20)
        Rasio_Natrium_terhadap_Kalium = st.sidebar.slider('Rasio Natrium terhadap Kalium (gram)', 6.2, 38.2, 25.3)
        data = {'Jenis_Kelamin' : Jenis_Kelamin,
                'tekanan_darah' : tekanan_darah,
                'Cholesterol' : Cholesterol,
                'Umur' : Umur,
                'Rasio_Natrium_terhadap_Kalium' : Rasio_Natrium_terhadap_Kalium}
        fitur = pd.DataFrame(data, index=[0])

        return fitur
    
    inputan = input_user()

# Menghbungkan inputan dan dataset obat
obat_raw = pd.read_csv('drug200.csv', sep=';')
obat = obat_raw.drop(columns=['obat'], errors='ignore')
df = pd.concat([inputan, obat], axis=0)


# Encode untuk atribut numerik
encode = ['Jenis_Kelamin','tekanan_darah','Cholesterol']
for col in encode:
    dummy = pd.get_dummies(df[col],prefix=col)
    dummy = dummy.astype(int)
    data = pd.concat([df,dummy], axis=1)
    del df[col]
df =df[:1] # diambil baris pertama( input data user)


# Menampilkan parameter inputan
st.subheader('parameter inputan')

if upload_file is not None:
    st.write(df)
else:
    st.write('Menunggu file csv untuk diupload. Saat ini memakai dampel inputan (seperti tampilan dibawah)')
    st.write(df)

# Load model NBC
load_model = pickle.load(open('modelNBC_obat.pkl', 'rb'))

# terapkan NBC
prediksi = load_model.predict(df)
prediksi_proba = load_model.predict_proba(df)

st.subheader('keterangan label obat')
jenis_obat = np.array(['drugA','drugB','drugC','drugX','drugY'])
st.write(jenis_obat)

st.subheader('hasil prediksi (klasifikasi oabat)')
st.write(jenis_obat[prediksi])

st.subheader('probabilitas hasil prediksi (klasifikasi obat)')
st.write(prediksi_proba)