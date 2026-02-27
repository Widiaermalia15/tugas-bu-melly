import streamlit as st
import pandas as pd

# Judul aplikasi
st.title("Aplikasi Analisis Data Siswa")

st.write("Upload file Excel untuk analisis")

# Upload file
uploaded_file = st.file_uploader("Upload file Excel", type=["xlsx"])

if uploaded_file is not None:
    # Baca data
    df = pd.read_excel(uploaded_file)

    st.subheader("Data Siswa")
    st.dataframe(df)

    # Hitung skor total
    df["Total_Skor"] = df.sum(axis=1)

    st.subheader("Total Skor Siswa")
    st.dataframe(df[["Total_Skor"]])

    # Statistik
    st.subheader("Statistik")
    st.write("Rata-rata:", df["Total_Skor"].mean())
    st.write("Nilai tertinggi:", df["Total_Skor"].max())
    st.write("Nilai terendah:", df["Total_Skor"].min())

    # Grafik
    st.subheader("Distribusi Nilai")
    st.bar_chart(df["Total_Skor"])