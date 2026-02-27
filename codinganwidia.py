import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

st.set_page_config(page_title="Analisis Butir Soal Lengkap", layout="wide")

st.title("📊 Dashboard Analisis Butir Soal")

uploaded_file = st.sidebar.file_uploader("Upload File Excel", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

    nama_siswa = df.iloc[:, 0]
    jawaban = df.iloc[:, 1:].astype(str)

    jumlah_soal = jawaban.shape[1]

    st.sidebar.subheader("🔑 Input Kunci Jawaban")
    kunci_input = st.sidebar.text_input(
        f"Masukkan {jumlah_soal} kunci (contoh: ABCDABCD...)"
    )

    if kunci_input and len(kunci_input) == jumlah_soal:

        kunci = list(kunci_input.upper())

        # =====================
        # SKORING
        # =====================
        skor = pd.DataFrame()

        for i, col in enumerate(jawaban.columns):
            skor[col] = (jawaban[col].str.upper() == kunci[i]).astype(int)

        total_skor = skor.sum(axis=1)

        # =====================
        # TINGKAT KESUKARAN
        # =====================
        tingkat_kesukaran = skor.mean()

        # =====================
        # DAYA PEMBEDA (27%)
        # =====================
        df_temp = skor.copy()
        df_temp["Total"] = total_skor
        df_sorted = df_temp.sort_values("Total", ascending=False)

        batas = int(0.27 * len(df_sorted))

        kelompok_atas = df_sorted.head(batas).drop("Total", axis=1)
        kelompok_bawah = df_sorted.tail(batas).drop("Total", axis=1)

        daya_pembeda = kelompok_atas.mean() - kelompok_bawah.mean()

        # =====================
        # VALIDITAS
        # =====================
        validitas = []

        for col in skor.columns:
            if skor[col].nunique() > 1:
                r, _ = pearsonr(skor[col], total_skor)
                validitas.append(r)
            else:
                validitas.append(0)

        validitas = pd.Series(validitas, index=skor.columns)

        # =====================
        # RELIABILITAS
        # =====================
        k = skor.shape[1]
        var_total = total_skor.var()
        var_item = skor.var().sum()

        if var_total != 0:
            cronbach_alpha = (k / (k - 1)) * (1 - (var_item / var_total))
        else:
            cronbach_alpha = 0

        # =====================
        # TABS
        # =====================
        tab1, tab2, tab3 = st.tabs(["📊 Statistik", "📋 Tabel Item", "📈 Grafik"])

        # ---------------------
        # TAB 1
        # ---------------------
        with tab1:
            col1, col2, col3 = st.columns(3)
            col1.metric("Rata-rata", round(total_skor.mean(), 2))
            col2.metric("Skor Tertinggi", total_skor.max())
            col3.metric("Reliabilitas", round(cronbach_alpha, 3))

        # ---------------------
        # TAB 2
        # ---------------------
        with tab2:
            hasil_item = pd.DataFrame({
                "Kesukaran (P)": tingkat_kesukaran,
                "Daya Pembeda (D)": daya_pembeda,
                "Validitas (r)": validitas
            })
            st.dataframe(hasil_item)

        # ---------------------
        # TAB 3 (GRAFIK FIX)
        # ---------------------
        with tab3:

            st.subheader("Tingkat Kesukaran")
            fig1, ax1 = plt.subplots()
            ax1.bar(tingkat_kesukaran.index, tingkat_kesukaran.values)
            plt.xticks(rotation=90)
            st.pyplot(fig1)

            st.subheader("Daya Pembeda")
            fig2, ax2 = plt.subplots()
            ax2.bar(daya_pembeda.index, daya_pembeda.values)
            plt.xticks(rotation=90)
            st.pyplot(fig2)

            st.subheader("Validitas")
            fig3, ax3 = plt.subplots()
            ax3.bar(validitas.index, validitas.values)
            plt.xticks(rotation=90)
            st.pyplot(fig3)

else:
    st.info("Silakan upload file terlebih dahulu.")