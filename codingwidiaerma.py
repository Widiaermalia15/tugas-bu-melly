import streamlit as st
import pandas as pd

# Konfigurasi halaman
st.set_page_config(
    page_title="Analisis Hasil Belajar Siswa",
    page_icon="📊",
    layout="wide"
)

# Header
st.title("📊 Dashboard Analisis Hasil Belajar")
st.markdown("Aplikasi ini digunakan untuk menganalisis data jawaban siswa secara otomatis.")

# Sidebar
st.sidebar.header("⚙️ Pengaturan")
uploaded_file = st.sidebar.file_uploader("Upload file Excel", type=["xlsx"])

# Jika file diupload
if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

    st.success("File berhasil diupload!")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📋 Data", "📈 Statistik", "📊 Visualisasi"])

    # =====================
    # TAB 1 : DATA
    # =====================
    with tab1:
        st.subheader("Data Jawaban Siswa")
        st.dataframe(df, use_container_width=True)

    # =====================
    # Hitung skor total
    # =====================
    df["Total Skor"] = df.sum(axis=1)

    # =====================
    # TAB 2 : STATISTIK
    # =====================
    with tab2:
        col1, col2, col3 = st.columns(3)

        col1.metric("📊 Rata-rata", round(df["Total Skor"].mean(), 2))
        col2.metric("🏆 Nilai Tertinggi", df["Total Skor"].max())
        col3.metric("📉 Nilai Terendah", df["Total Skor"].min())

        st.write("### Distribusi Skor")
        st.write(df["Total Skor"].describe())

    # =====================
    # TAB 3 : VISUALISASI
    # =====================
    with tab3:
        st.subheader("Grafik Distribusi Nilai")
        st.bar_chart(df["Total Skor"])

        st.subheader("Histogram Nilai")
        st.histogram_chart = st.bar_chart(df["Total Skor"].value_counts().sort_index())

    # =====================
    # Download hasil
    # =====================
    st.sidebar.download_button(
        label="⬇️ Download hasil",
        data=df.to_csv(index=False),
        file_name="hasil_analisis.csv",
        mime="text/csv"
    )

else:
    st.info("Silakan upload file Excel terlebih dahulu.")