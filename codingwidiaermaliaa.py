import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

st.set_page_config(page_title="Analisis Butir Soal Lengkap", layout="wide")

st.title("📊 Dashboard Analisis Butir Soal Lengkap")

uploaded_file = st.sidebar.file_uploader("Upload File Excel", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

    st.subheader("📋 Data Awal")
    st.dataframe(df)

    nama_siswa = df.iloc[:, 0]
    jawaban = df.iloc[:, 1:]

    jumlah_soal = jawaban.shape[1]

    st.sidebar.subheader("🔑 Input Kunci Jawaban")
    kunci_input = st.sidebar.text_input(
        f"Masukkan {jumlah_soal} kunci jawaban (contoh: ABCDABCD...)"
    )

    if kunci_input and len(kunci_input) == jumlah_soal:

        kunci = list(kunci_input.upper())

        # Skoring
        skor = jawaban.apply(lambda x: x.str.upper() == kunci)
        skor = skor.astype(int)

        total_skor = skor.sum(axis=1)

        df["Total Skor"] = total_skor

        # ==============================
        # 1️⃣ Tingkat Kesukaran
        # ==============================
        tingkat_kesukaran = skor.mean()

        # ==============================
        # 2️⃣ Daya Pembeda (27%)
        # ==============================
        df_sorted = df.sort_values("Total Skor", ascending=False)
        batas = int(0.27 * len(df))

        kelompok_atas = skor.iloc[df_sorted.index[:batas]]
        kelompok_bawah = skor.iloc[df_sorted.index[-batas:]]

        daya_pembeda = kelompok_atas.mean() - kelompok_bawah.mean()

        # ==============================
        # 3️⃣ Validitas (Item-Total)
        # ==============================
        validitas = []
        for col in skor.columns:
            r, _ = pearsonr(skor[col], total_skor)
            validitas.append(r)

        validitas = pd.Series(validitas, index=skor.columns)

        # ==============================
        # 4️⃣ Reliabilitas (Cronbach Alpha)
        # ==============================
        k = skor.shape[1]
        var_total = total_skor.var()
        var_item = skor.var().sum()
        cronbach_alpha = (k / (k - 1)) * (1 - (var_item / var_total))

        # ==============================
        # TAB
        # ==============================
        tab1, tab2, tab3, tab4 = st.tabs(
            ["📊 Statistik Umum", "📈 Analisis Item", "📉 Visualisasi", "🏆 Ranking"]
        )

        # ==============================
        # TAB 1
        # ==============================
        with tab1:
            col1, col2, col3 = st.columns(3)
            col1.metric("Rata-rata", round(total_skor.mean(), 2))
            col2.metric("Skor Tertinggi", total_skor.max())
            col3.metric("Reliabilitas (Alpha)", round(cronbach_alpha, 3))

        # ==============================
        # TAB 2
        # ==============================
        with tab2:
            hasil_item = pd.DataFrame({
                "Kesukaran (P)": tingkat_kesukaran,
                "Daya Pembeda (D)": daya_pembeda,
                "Validitas (r)": validitas
            })

            st.dataframe(hasil_item)

        # ==============================
        # TAB 3
        # ==============================
        with tab3:
            st.subheader("📊 Grafik Tingkat Kesukaran")
            st.bar_chart(tingkat_kesukaran)

            st.subheader("📊 Grafik Daya Pembeda")
            st.bar_chart(daya_pembeda)

            st.subheader("📊 Grafik Validitas")
            st.bar_chart(validitas)

        # ==============================
        # TAB 4
        # ==============================
        with tab4:
            ranking = df.sort_values("Total Skor", ascending=False)
            st.dataframe(ranking[[df.columns[0], "Total Skor"]])

        # ==============================
        # Download
        # ==============================
        st.sidebar.download_button(
            "⬇️ Download Hasil Analisis",
            hasil_item.to_csv(),
            "hasil_analisis_item.csv",
            "text/csv"
        )

    else:
        st.warning("Masukkan kunci jawaban sesuai jumlah soal.")

else:
    st.info("Silakan upload file Excel terlebih dahulu.")