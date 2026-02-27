import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==============================
# KONFIGURASI HALAMAN
# ==============================
st.set_page_config(page_title="Dashboard Analisis Hasil Belajar", layout="wide")
st.title("🎓 Dashboard Analisis Hasil Belajar Siswa")
st.markdown("Upload data Excel (50 siswa × 20 soal) untuk analisis otomatis")

# ==============================
# UPLOAD FILE
# ==============================
uploaded_file = st.file_uploader("Upload File Excel", type=["xlsx"])

if uploaded_file is not None:

    try:
        df = pd.read_excel(uploaded_file)
    except:
        st.error("File tidak bisa dibaca. Pastikan format Excel benar.")
        st.stop()

    df_numeric = df.select_dtypes(include=np.number)

    if df_numeric.shape[1] < 20:
        st.error("Minimal harus ada 20 kolom numerik untuk soal.")
        st.stop()

    if df_numeric.iloc[:, :20].isnull().sum().sum() > 0:
        st.error("Terdapat nilai kosong pada data.")
        st.stop()

    data = df_numeric.iloc[:, :20].copy()
    data["Total_Skor"] = data.sum(axis=1)

    # ==============================
    # 1. STATISTIK NILAI
    # ==============================
    st.header("1️⃣ Statistik Nilai Siswa")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Jumlah Siswa", len(data))
    col2.metric("Rata-rata", round(data["Total_Skor"].mean(), 2))
    col3.metric("Nilai Tertinggi", int(data["Total_Skor"].max()))
    col4.metric("Nilai Terendah", int(data["Total_Skor"].min()))

    fig1, ax1 = plt.subplots()
    ax1.hist(data["Total_Skor"], bins=10)
    ax1.set_title("Distribusi Skor Total")
    st.pyplot(fig1)

    # ==============================
    # 2. TINGKAT KESUKARAN
    # ==============================
    st.header("2️⃣ Tingkat Kesukaran Soal")

    tingkat_kesukaran = data.iloc[:, :20].mean()

    fig2, ax2 = plt.subplots(figsize=(10,4))
    tingkat_kesukaran.plot(kind="bar", ax=ax2)
    ax2.axhline(tingkat_kesukaran.mean(), linestyle="--", color="red")
    ax2.set_title("Rata-rata Skor per Soal")
    st.pyplot(fig2)

    st.success(f"Soal Paling Mudah: {tingkat_kesukaran.idxmax()}")
    st.error(f"Soal Paling Sulit: {tingkat_kesukaran.idxmin()}")

    # ==============================
    # 3. DAYA PEMBEDA (Corrected)
    # ==============================
    st.header("3️⃣ Daya Pembeda Soal")

    daya_pembeda = {}
    for col in data.columns[:20]:
        total_minus_item = data["Total_Skor"] - data[col]
        daya_pembeda[col] = data[col].corr(total_minus_item)

    daya_pembeda = pd.Series(daya_pembeda)

    fig3, ax3 = plt.subplots(figsize=(10,4))
    daya_pembeda.plot(kind="bar", ax=ax3)
    ax3.axhline(0.3, linestyle="--", color="red")
    ax3.set_title("Corrected Item-Total Correlation")
    st.pyplot(fig3)

    # ==============================
    # 4. CRONBACH ALPHA
    # ==============================
    st.header("4️⃣ Reliabilitas Tes")

    item_var = data.iloc[:, :20].var(axis=0, ddof=1)
    total_var = data["Total_Skor"].var(ddof=1)
    k = 20

    cronbach_alpha = (k / (k - 1)) * (1 - item_var.sum() / total_var)

    st.metric("Cronbach Alpha", round(cronbach_alpha, 3))

    # ==============================
    # 5. HEATMAP KORELASI
    # ==============================
    st.header("5️⃣ Korelasi Antar Soal")

    corr = data.iloc[:, :20].corr()

    fig4, ax4 = plt.subplots(figsize=(8,6))
    sns.heatmap(corr, cmap="coolwarm", ax=ax4)
    ax4.set_title("Heatmap Korelasi")
    st.pyplot(fig4)

    # ==============================
    # 6. CLUSTERING SISWA
    # ==============================
    st.header("6️⃣ Segmentasi Kemampuan")

    jumlah_cluster = st.slider("Jumlah Cluster", 2, 5, 3)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(data.iloc[:, :20])

    kmeans = KMeans(n_clusters=jumlah_cluster, random_state=42, n_init=10)
    cluster = kmeans.fit_predict(scaled)

    data["Cluster"] = cluster

    cluster_mean = data.groupby("Cluster")["Total_Skor"].mean()

    fig5, ax5 = plt.subplots()
    cluster_mean.plot(kind="bar", ax=ax5)
    ax5.set_title("Rata-rata Skor per Cluster")
    st.pyplot(fig5)

    # ==============================
    # 7. RANKING SOAL
    # ==============================
    st.header("7️⃣ Ranking Soal")

    ranking = daya_pembeda.sort_values(ascending=False)
    st.dataframe(ranking)

    # ==============================
    # DOWNLOAD
    # ==============================
    st.download_button(
        label="Download Hasil (CSV)",
        data=data.to_csv(index=False).encode("utf-8"),
        file_name="hasil_analisis.csv",
        mime="text/csv"
    )

    st.success("Analisis selesai ✅")