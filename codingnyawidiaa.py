import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ======================================================
# KONFIGURASI HALAMAN
# ======================================================
st.set_page_config(page_title="DASHBOARD ANALISIS HASIL BELAJAR SISWA", layout="wide")
st.title("🎓 DASHBOARD ANALISIS HASIL BELAJAR SISWA")
st.markdown("Analisis Data 50 Siswa × 20 Soal Secara Interaktif dan Visual")

# ======================================================
# UPLOAD FILE
# ======================================================
uploaded_file = st.file_uploader("Upload File Excel (50 siswa × 20 soal)", type=["xlsx"])

if uploaded_file is not None:

    df = pd.read_excel(uploaded_file)
    df_numeric = df.select_dtypes(include=np.number)

    if df_numeric.shape[1] < 20:
        st.error("File harus memiliki minimal 20 kolom soal numerik.")
        st.stop()

    if df_numeric.iloc[:, :20].isnull().sum().sum() > 0:
        st.error("Terdapat nilai kosong. Mohon bersihkan data terlebih dahulu.")
        st.stop()

    data = df_numeric.iloc[:, :20].copy()

    # ======================================================
    # 1️⃣ SKOR TOTAL
    # ======================================================
    data["Total_Skor"] = data.sum(axis=1)

    st.header("1️⃣ Statistik Nilai Siswa")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Jumlah Siswa", len(data))
    col2.metric("Rata-rata", round(data["Total_Skor"].mean(), 2))
    col3.metric("Nilai Tertinggi", data["Total_Skor"].max())
    col4.metric("Nilai Terendah", data["Total_Skor"].min())

    fig1, ax1 = plt.subplots()
    ax1.hist(data["Total_Skor"], bins=10)
    ax1.set_title("Distribusi Skor Total")
    ax1.set_xlabel("Skor")
    ax1.set_ylabel("Frekuensi")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    ax2.boxplot(data["Total_Skor"], vert=False)
    ax2.set_title("Boxplot Skor Total")
    st.pyplot(fig2)

    # ======================================================
    # 2️⃣ TINGKAT KESUKARAN
    # ======================================================
    st.header("2️⃣ Analisis Tingkat Kesukaran Soal")

    tingkat_kesukaran = data.iloc[:, :20].mean()

    fig3, ax3 = plt.subplots(figsize=(10,4))
    tingkat_kesukaran.plot(kind="bar", ax=ax3)
    ax3.axhline(tingkat_kesukaran.mean(), linestyle="--", color="red")
    ax3.set_title("Rata-rata Skor per Soal")
    st.pyplot(fig3)

    soal_mudah = tingkat_kesukaran.idxmax()
    soal_sulit = tingkat_kesukaran.idxmin()

    col5, col6 = st.columns(2)
    col5.success(f"Soal Paling Mudah: {soal_mudah}")
    col6.error(f"Soal Paling Sulit: {soal_sulit}")

    # ======================================================
    # 3️⃣ DAYA PEMBEDA (Corrected Item-Total Correlation)
    # ======================================================
    st.header("3️⃣ Analisis Daya Pembeda")

    daya_pembeda = {}
    for col in data.columns[:20]:
        total_minus_item = data["Total_Skor"] - data[col]
        daya_pembeda[col] = data[col].corr(total_minus_item)

    daya_pembeda = pd.Series(daya_pembeda)

    fig4, ax4 = plt.subplots(figsize=(10,4))
    daya_pembeda.plot(kind="bar", ax=ax4)
    ax4.axhline(0.3, linestyle="--", color="red")
    ax4.set_title("Corrected Item-Total Correlation")
    st.pyplot(fig4)

    # ======================================================
    # 4️⃣ RELIABILITAS (CRONBACH ALPHA)
    # ======================================================
    st.header("4️⃣ Reliabilitas Tes (Cronbach Alpha)")

    item_var = data.iloc[:, :20].var(axis=0, ddof=1)
    total_var = data["Total_Skor"].var(ddof=1)
    k = 20

    cronbach_alpha = (k / (k - 1)) * (1 - item_var.sum() / total_var)

    st.metric("Nilai Cronbach Alpha", round(cronbach_alpha, 3))

    if cronbach_alpha >= 0.9:
        st.success("Reliabilitas Sangat Tinggi")
    elif cronbach_alpha >= 0.7:
        st.info("Reliabilitas Baik")
    else:
        st.warning("Reliabilitas Rendah")

    # ======================================================
    # 5️⃣ HEATMAP KORELASI
    # ======================================================
    st.header("5️⃣ Korelasi Antar Soal")

    corr = data.iloc[:, :20].corr()

    fig5, ax5 = plt.subplots(figsize=(8,6))
    sns.heatmap(corr, cmap="coolwarm", ax=ax5)
    ax5.set_title("Heatmap Korelasi Soal")
    st.pyplot(fig5)

    # ======================================================
    # 6️⃣ CLUSTERING SISWA
    # ======================================================
    st.header("6️⃣ Segmentasi Kemampuan Siswa")

    jumlah_cluster = st.slider("Pilih Jumlah Cluster", 2, 5, 3)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(data.iloc[:, :20])

    kmeans = KMeans(n_clusters=jumlah_cluster, random_state=42, n_init=10)
    cluster = kmeans.fit_predict(scaled)

    data["Cluster"] = cluster

    cluster_mean = data.groupby("Cluster")["Total_Skor"].mean()

    # Interpretasi otomatis
    sorted_cluster = cluster_mean.sort_values().index.tolist()
    kategori = {}

    if jumlah_cluster == 3:
        kategori = {
            sorted_cluster[0]: "Rendah",
            sorted_cluster[1]: "Sedang",
            sorted_cluster[2]: "Tinggi"
        }

    data["Kategori_Kemampuan"] = data["Cluster"].map(kategori)

    fig6, ax6 = plt.subplots()
    cluster_mean.plot(kind="bar", ax=ax6)
    ax6.set_title("Rata-rata Skor per Cluster")
    st.pyplot(fig6)

    fig7, ax7 = plt.subplots()
    data["Cluster"].value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax7)
    ax7.set_ylabel("")
    ax7.set_title("Komposisi Kelompok Siswa")
    st.pyplot(fig7)

    # ======================================================
    # 7️⃣ RANKING SOAL
    # ======================================================
    st.header("7️⃣ Ranking Soal Berdasarkan Daya Pembeda")

    ranking = daya_pembeda.sort_values(ascending=False)
    st.dataframe(ranking)

    # ======================================================
    # DOWNLOAD HASIL
    # ======================================================
    st.header("📥 Download Hasil Analisis")

    output = data.copy()
    output["Total_Skor"] = data["Total_Skor"]

    st.download_button(
        label="Download Hasil Analisis (Excel)",
        data=output.to_csv(index=False).encode("utf-8"),
        file_name="hasil_analisis_siswa.csv",
        mime="text/csv"
    )

    st.success("Analisis Selesai ✅ Dashboard Siap Digunakan")