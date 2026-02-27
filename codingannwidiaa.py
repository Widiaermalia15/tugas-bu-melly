import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==================================================
# KONFIGURASI HALAMAN
# ==================================================
st.set_page_config(page_title="Analisis Hasil Belajar", layout="wide")
st.title("📊 Dashboard Analisis Hasil Belajar Siswa")

# ==================================================
# INPUT KKM
# ==================================================
kkm = st.number_input("Masukkan KKM", value=75)

# ==================================================
# UPLOAD DATA
# ==================================================
file = st.file_uploader("Upload data nilai siswa (Excel)", type=["xlsx"])

if file is not None:
    df = pd.read_excel(file)

    st.subheader("📄 Data Siswa")
    st.dataframe(df)

    # Ambil kolom nilai
    nilai = df.iloc[:, 2:]  # sesuaikan jika perlu

    # ==================================================
    # 1️⃣ RATA-RATA HASIL BELAJAR
    # ==================================================
    st.header("1️⃣ Statistik Hasil Belajar")

    df["Rata-rata"] = nilai.mean(axis=1)
    rata_kelas = df["Rata-rata"].mean()

    c1, c2 = st.columns(2)
    c1.metric("Rata-rata Kelas", f"{rata_kelas:.2f}")
    c2.metric("Jumlah Siswa", len(df))

    # ==================================================
    # 2️⃣ KETUNTASAN BELAJAR
    # ==================================================
    st.header("2️⃣ Ketuntasan Belajar")

    df["Ketuntasan"] = np.where(df["Rata-rata"] >= kkm, "Tuntas", "Belum Tuntas")

    persen_tuntas = (df["Ketuntasan"] == "Tuntas").mean() * 100

    st.metric("Persentase Ketuntasan", f"{persen_tuntas:.2f}%")

    # Grafik ketuntasan
    fig1, ax1 = plt.subplots()
    df["Ketuntasan"].value_counts().plot(kind="bar", ax=ax1)
    ax1.set_title("Distribusi Ketuntasan")
    st.pyplot(fig1)

    # ==================================================
    # 3️⃣ RANKING HASIL BELAJAR
    # ==================================================
    st.header("3️⃣ Ranking Siswa")

    ranking = df.sort_values("Rata-rata", ascending=False)
    st.dataframe(ranking)

    # ==================================================
    # 4️⃣ ANALISIS PER MATA PELAJARAN
    # ==================================================
    st.header("4️⃣ Analisis Mata Pelajaran")

    mean_mapel = nilai.mean()

    fig2, ax2 = plt.subplots()
    ax2.bar(mean_mapel.index, mean_mapel.values)
    ax2.axhline(kkm, linestyle="--")
    ax2.set_title("Rata-rata Nilai per Mata Pelajaran")

    st.pyplot(fig2)

    st.info(f"Mata pelajaran perlu perbaikan: {mean_mapel.idxmin()}")

    # ==================================================
    # 5️⃣ SEGMENTASI HASIL BELAJAR
    # ==================================================
    st.header("5️⃣ Segmentasi Siswa")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(nilai)

    kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
    cluster = kmeans.fit_predict(X_scaled)

    df["Kategori"] = cluster

    st.dataframe(df)

    # ==================================================
    # 6️⃣ REKOMENDASI PEMBELAJARAN
    # ==================================================
    st.header("6️⃣ Rekomendasi")

    if persen_tuntas < 70:
        st.warning("Ketuntasan rendah → perlu remedial dan diferensiasi pembelajaran.")
    else:
        st.success("Ketuntasan baik → lanjutkan strategi pembelajaran.")

    st.write("Siswa belum tuntas perlu:")
    st.write("- Pembelajaran remedial")
    st.write("- Pendampingan individual")
    st.write("- Penguatan konsep dasar")

else:
    st.info("Upload file Excel terlebih dahulu.")