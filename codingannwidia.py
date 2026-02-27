import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==================================================
# KONFIGURASI HALAMAN
# ==================================================
st.set_page_config(page_title="Dashboard Kepuasan", layout="wide")
st.title("📊 Dashboard Analisis Kepuasan Pegawai")

# ==================================================
# UPLOAD DATA (lebih fleksibel)
# ==================================================
file = st.file_uploader("Upload Data Excel", type=["xlsx"])

if file is not None:
    df = pd.read_excel(file)

    # Ambil indikator (ubah sesuai data)
    indikator = df.iloc[:, 1:6].apply(pd.to_numeric, errors="coerce")

    # ==================================================
    # KPI KEPUASAN
    # ==================================================
    st.header("1️⃣ Indeks Kepuasan")

    mean_scores = indikator.mean()
    ikm = (mean_scores.mean() / 5) * 100

    def kategori(x):
        if x >= 81: return "Sangat Baik"
        elif x >= 66: return "Baik"
        elif x >= 51: return "Cukup"
        else: return "Kurang"

    c1, c2, c3 = st.columns(3)
    c1.metric("IKM", f"{ikm:.2f}%")
    c2.metric("Kategori", kategori(ikm))
    c3.metric("Jumlah Responden", len(df))

    # ==================================================
    # ANALISIS GAP
    # ==================================================
    st.header("2️⃣ Analisis GAP")

    gap = 5 - mean_scores
    prioritas = gap.idxmax()

    fig, ax = plt.subplots()
    ax.bar(gap.index, gap.values)
    ax.set_title("Gap Kepuasan")

    st.pyplot(fig)
    st.info(f"Prioritas perbaikan: {prioritas}")

    # ==================================================
    # KORELASI
    # ==================================================
    st.header("3️⃣ Korelasi")

    corr = indikator.corr()

    fig2, ax2 = plt.subplots()
    im = ax2.imshow(corr, vmin=-1, vmax=1)
    plt.colorbar(im)

    ax2.set_xticks(range(len(corr.columns)))
    ax2.set_xticklabels(corr.columns)

    ax2.set_yticks(range(len(corr.columns)))
    ax2.set_yticklabels(corr.columns)

    st.pyplot(fig2)

    # ==================================================
    # REGRESI
    # ==================================================
    st.header("4️⃣ Regresi Linear")

    X = sm.add_constant(indikator.iloc[:, 0:4])
    y = indikator.iloc[:, 4]

    model = sm.OLS(y, X, missing="drop").fit()

    coef = model.params[1:]

    fig3, ax3 = plt.subplots()
    ax3.bar(coef.index, coef.values)
    ax3.axhline(0)

    st.pyplot(fig3)
    st.write("R² =", model.rsquared)

    # ==================================================
    # CLUSTERING
    # ==================================================
    st.header("5️⃣ Segmentasi Kepuasan")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(indikator.fillna(indikator.mean()))

    kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
    cluster = kmeans.fit_predict(X_scaled)

    indikator["Cluster"] = cluster

    st.dataframe(indikator.groupby("Cluster").mean())

    st.success("Segmentasi selesai ✅")

else:
    st.warning("Silakan upload data terlebih dahulu.")