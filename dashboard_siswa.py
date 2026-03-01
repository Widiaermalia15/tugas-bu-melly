import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

# ======================================================
# KONFIGURASI HALAMAN
# ======================================================
st.set_page_config(
    page_title="EduInsight — Analisis Hasil Belajar",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================
# CUSTOM CSS — PREMIUM DARK THEME
# ======================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    min-height: 100vh;
}

/* Main container */
.block-container {
    padding-top: 2rem !important;
    max-width: 1400px;
}

/* Header */
.main-header {
    background: linear-gradient(90deg, rgba(139,92,246,0.15) 0%, rgba(236,72,153,0.15) 100%);
    border: 1px solid rgba(139,92,246,0.3);
    border-radius: 20px;
    padding: 32px 40px;
    margin-bottom: 32px;
    backdrop-filter: blur(10px);
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(236,72,153,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.main-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.6rem;
    font-weight: 900;
    background: linear-gradient(90deg, #a78bfa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 8px 0;
    letter-spacing: -0.02em;
}
.main-subtitle {
    color: rgba(196,181,253,0.7);
    font-size: 0.95rem;
    font-weight: 300;
    letter-spacing: 0.05em;
}

/* Metric cards */
.metric-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 20px 24px;
    text-align: center;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}
.metric-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #8b5cf6, #ec4899);
    border-radius: 0 0 16px 16px;
}
.metric-number {
    font-family: 'Playfair Display', serif;
    font-size: 2.4rem;
    font-weight: 700;
    color: #f9fafb;
    line-height: 1;
    margin-bottom: 4px;
}
.metric-label {
    font-size: 0.75rem;
    font-weight: 500;
    color: rgba(196,181,253,0.6);
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.metric-badge {
    font-size: 0.72rem;
    padding: 2px 10px;
    border-radius: 20px;
    display: inline-block;
    margin-top: 8px;
    font-weight: 500;
}

/* Section headers */
.section-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 40px 0 20px 0;
    padding-bottom: 12px;
    border-bottom: 1px solid rgba(139,92,246,0.2);
}
.section-icon {
    width: 38px; height: 38px;
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.2rem;
    flex-shrink: 0;
}
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: #f3f4f6;
    margin: 0;
}
.section-desc {
    font-size: 0.78rem;
    color: rgba(196,181,253,0.5);
    margin: 2px 0 0 0;
}

/* Info boxes */
.insight-box {
    border-radius: 12px;
    padding: 14px 18px;
    margin: 8px 0;
    font-size: 0.85rem;
    border-left: 3px solid;
}
.insight-success {
    background: rgba(16,185,129,0.1);
    border-color: #10b981;
    color: #6ee7b7;
}
.insight-warning {
    background: rgba(245,158,11,0.1);
    border-color: #f59e0b;
    color: #fcd34d;
}
.insight-info {
    background: rgba(59,130,246,0.1);
    border-color: #3b82f6;
    color: #93c5fd;
}
.insight-danger {
    background: rgba(239,68,68,0.1);
    border-color: #ef4444;
    color: #fca5a5;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(15,12,41,0.95) !important;
    border-right: 1px solid rgba(139,92,246,0.2);
}

/* Upload area */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(139,92,246,0.4) !important;
    border-radius: 16px !important;
    background: rgba(139,92,246,0.05) !important;
}

/* Tabs */
[data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif !important;
}

/* DataFrame */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}

/* Progress bars */
.prog-row {
    display: flex; align-items: center; gap: 10px;
    margin: 6px 0;
}
.prog-label { font-size: 0.78rem; color: #d1d5db; min-width: 60px; }
.prog-track {
    flex: 1; height: 8px;
    background: rgba(255,255,255,0.08);
    border-radius: 4px; overflow: hidden;
}
.prog-fill {
    height: 100%; border-radius: 4px;
}
.prog-val { font-size: 0.72rem; color: #9ca3af; min-width: 38px; text-align: right; }

/* Cluster badges */
.cluster-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# MATPLOTLIB STYLE
# ======================================================
plt.rcParams.update({
    'figure.facecolor': '#1a1635',
    'axes.facecolor': '#1a1635',
    'axes.edgecolor': '#3d3666',
    'axes.labelcolor': '#c4b5fd',
    'xtick.color': '#9ca3af',
    'ytick.color': '#9ca3af',
    'text.color': '#f3f4f6',
    'grid.color': '#2d2b55',
    'grid.alpha': 0.5,
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'axes.titlecolor': '#f3f4f6',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

GRAD_PURPLE = ['#4c1d95', '#7c3aed', '#a78bfa', '#c4b5fd', '#ede9fe']
GRAD_PINK   = ['#831843', '#be185d', '#ec4899', '#f9a8d4']
PURPLE_PINK = LinearSegmentedColormap.from_list('pp', ['#7c3aed', '#ec4899'])
CLUSTER_COLORS = ['#8b5cf6', '#10b981', '#f59e0b']

# ======================================================
# HEADER
# ======================================================
st.markdown("""
<div class="main-header">
  <p class="main-title">EduInsight</p>
  <p class="main-subtitle">✦ DASHBOARD ANALISIS HASIL BELAJAR SISWA &nbsp;·&nbsp; PSIKOMETRI & SEGMENTASI KELAS</p>
</div>
""", unsafe_allow_html=True)

# ======================================================
# SIDEBAR
# ======================================================
with st.sidebar:
    st.markdown("### ⚙️ Pengaturan")
    st.markdown("---")
    n_clusters = st.slider("Jumlah Kelompok Siswa", 2, 5, 3)
    kkm = st.number_input("KKM (Nilai Minimum Lulus)", min_value=0, max_value=100, value=60)
    show_raw = st.checkbox("Tampilkan Data Mentah", value=False)
    color_scheme = st.selectbox("Tema Warna Chart", ["Ungu-Pink", "Biru-Hijau", "Api"])
    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem; color:rgba(196,181,253,0.5); line-height:1.7;'>
    📌 <b>Panduan Format File</b><br>
    • Baris = siswa (min 10)<br>
    • Kolom = soal (min 20)<br>
    • Nilai = angka (0–100 atau 0–1)<br>
    • Baris pertama = nama kolom
    </div>
    """, unsafe_allow_html=True)

if color_scheme == "Biru-Hijau":
    CLUSTER_COLORS = ['#3b82f6', '#10b981', '#f59e0b']
    PURPLE_PINK = LinearSegmentedColormap.from_list('bg', ['#3b82f6', '#10b981'])
elif color_scheme == "Api":
    CLUSTER_COLORS = ['#ef4444', '#f97316', '#fbbf24']
    PURPLE_PINK = LinearSegmentedColormap.from_list('fire', ['#ef4444', '#fbbf24'])

# ======================================================
# UPLOAD
# ======================================================
uploaded_file = st.file_uploader(
    "📂  Unggah file Excel (50 siswa × 20 soal)",
    type=["xlsx"],
    help="Format: baris=siswa, kolom=soal, nilai numerik"
)

# ======================================================
# GENERATE DEMO DATA
# ======================================================
if uploaded_file is None:
    st.markdown("""
    <div class="insight-box insight-info">
    💡 Belum ada file? Klik <b>Gunakan Data Demo</b> di bawah untuk melihat dashboard dengan 50 siswa simulasi.
    </div>
    """, unsafe_allow_html=True)
    use_demo = st.button("🎲 Gunakan Data Demo (50 Siswa × 20 Soal)")
    if not use_demo:
        st.stop()

    rng = np.random.default_rng(42)
    n_s, n_q = 50, 20
    # 3 kelompok kemampuan berbeda
    groups = [
        rng.normal(0.75, 0.12, (17, n_q)),
        rng.normal(0.50, 0.15, (18, n_q)),
        rng.normal(0.28, 0.13, (15, n_q)),
    ]
    raw = np.clip(np.vstack(groups), 0, 1) * 100
    raw = np.round(raw).astype(int)
    names = [f"Siswa_{i+1:02d}" for i in range(n_s)]
    qnames = [f"Soal_{j+1:02d}" for j in range(n_q)]
    df = pd.DataFrame(raw, columns=qnames)
    df.insert(0, "Nama", names)
    st.success("✅ Data demo berhasil dimuat — 50 siswa, 20 soal")
else:
    df = pd.read_excel(uploaded_file)

# ======================================================
# DATA PREP
# ======================================================
df_numeric = df.select_dtypes(include=np.number)
if df_numeric.shape[1] < 20:
    st.error("❌ File harus memiliki minimal 20 kolom soal numerik.")
    st.stop()

data = df_numeric.iloc[:, :20].copy()
soal_cols = data.columns.tolist()
data["Total_Skor"] = data[soal_cols].sum(axis=1)
max_possible = data[soal_cols].max().sum()
data["Persentase"] = (data["Total_Skor"] / max_possible * 100).round(1)
data["Status"] = data["Persentase"].apply(lambda x: "Lulus" if x >= kkm else "Tidak Lulus")

n_siswa = len(data)
rata2 = data["Total_Skor"].mean()
tertinggi = data["Total_Skor"].max()
terendah = data["Total_Skor"].min()
n_lulus = (data["Status"] == "Lulus").sum()
pct_lulus = n_lulus / n_siswa * 100

if show_raw:
    with st.expander("📋 Tabel Data Mentah", expanded=False):
        st.dataframe(df, use_container_width=True)

# ======================================================
# NAVIGASI TAB
# ======================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Ringkasan",
    "🔍 Analisis Soal",
    "👥 Segmentasi Siswa",
    "🎯 Rekomendasi",
    "📈 Laporan Lengkap"
])

# ==============================================================
# TAB 1 — RINGKASAN
# ==============================================================
with tab1:
    # KPI Cards
    cols = st.columns(5)
    cards = [
        (str(n_siswa), "Total Siswa", "#8b5cf6"),
        (f"{rata2:.1f}", "Rata-rata", "#ec4899"),
        (str(int(tertinggi)), "Tertinggi", "#10b981"),
        (str(int(terendah)), "Terendah", "#ef4444"),
        (f"{pct_lulus:.1f}%", f"Lulus (≥{kkm})", "#f59e0b"),
    ]
    for col, (num, label, color) in zip(cols, cards):
        col.markdown(f"""
        <div class="metric-card">
          <div class="metric-number" style="color:{color}">{num}</div>
          <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([3, 2])

    with c1:
        # Distribusi skor — HISTOGRAM dengan KDE
        fig, ax = plt.subplots(figsize=(7, 3.5))
        bins = np.linspace(data["Total_Skor"].min()-5, data["Total_Skor"].max()+5, 15)
        n, bins_out, patches = ax.hist(data["Total_Skor"], bins=bins, edgecolor='#1a1635', linewidth=0.8)
        for i, patch in enumerate(patches):
            ratio = i / len(patches)
            patch.set_facecolor(plt.cm.plasma(0.2 + ratio * 0.6))

        # KDE manual
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(data["Total_Skor"])
        x_kde = np.linspace(data["Total_Skor"].min()-5, data["Total_Skor"].max()+5, 200)
        y_kde = kde(x_kde)
        ax2_twin = ax.twinx()
        ax2_twin.plot(x_kde, y_kde, color='#f472b6', linewidth=2, alpha=0.8)
        ax2_twin.set_yticks([])
        ax2_twin.spines['right'].set_visible(False)
        ax2_twin.spines['top'].set_visible(False)

        # KKM line
        kkm_score = kkm / 100 * max_possible
        ax.axvline(kkm_score, color='#fbbf24', linestyle='--', linewidth=1.5, alpha=0.8, label=f'KKM ({kkm}%)')
        ax.axvline(rata2, color='#10b981', linestyle=':', linewidth=1.5, alpha=0.9, label=f'Rata-rata ({rata2:.1f})')
        ax.set_xlabel("Skor Total", fontsize=10)
        ax.set_ylabel("Jumlah Siswa", fontsize=10)
        ax.set_title("Distribusi Skor Siswa", fontsize=13, pad=12)
        ax.legend(fontsize=8, framealpha=0.2, facecolor='#1a1635', edgecolor='#3d3666')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)

    with c2:
        # Status lulus donut chart
        fig, ax = plt.subplots(figsize=(3.8, 3.5))
        sizes = [n_lulus, n_siswa - n_lulus]
        labels = [f'Lulus\n{n_lulus}', f'Belum Lulus\n{n_siswa - n_lulus}']
        colors = ['#8b5cf6', '#374151']
        wedges, texts = ax.pie(sizes, labels=labels, colors=colors, startangle=90,
                                wedgeprops=dict(width=0.55, edgecolor='#1a1635', linewidth=2))
        for t in texts: t.set_fontsize(9); t.set_color('#d1d5db')
        ax.text(0, 0, f'{pct_lulus:.0f}%', ha='center', va='center',
                fontsize=20, fontweight='bold', color='#a78bfa')
        ax.set_title("Status Kelulusan", fontsize=12, pad=10)
        plt.tight_layout()
        st.pyplot(fig)

    # Boxplot per kuartil
    st.markdown("---")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3))

    # Box + swarm-style
    ax = axes[0]
    bp = ax.boxplot(data["Total_Skor"], vert=False, patch_artist=True,
                    boxprops=dict(facecolor='#4c1d95', color='#7c3aed', linewidth=1.5),
                    medianprops=dict(color='#f472b6', linewidth=2.5),
                    whiskerprops=dict(color='#7c3aed', linewidth=1.5),
                    capprops=dict(color='#7c3aed', linewidth=2),
                    flierprops=dict(marker='o', color='#ec4899', markersize=5, alpha=0.6))
    jitter = np.random.uniform(-0.3, 0.3, n_siswa)
    ax.scatter(data["Total_Skor"], 1 + jitter, alpha=0.35, color='#c4b5fd', s=20, zorder=5)
    ax.set_title("Distribusi Boxplot + Jitter", fontsize=11)
    ax.set_xlabel("Skor Total")
    ax.set_yticks([])

    # Nilai per desil
    ax2 = axes[1]
    desil_vals = [np.percentile(data["Total_Skor"], p) for p in range(10, 101, 10)]
    desil_labels = [f'D{i}' for i in range(1, 11)]
    bars = ax2.bar(desil_labels, desil_vals, color=[plt.cm.plasma(0.15 + i*0.08) for i in range(10)],
                   edgecolor='#1a1635', linewidth=0.5)
    ax2.axhline(kkm_score, color='#fbbf24', linestyle='--', linewidth=1.2, alpha=0.8, label='KKM')
    ax2.set_title("Nilai Batas Per Desil", fontsize=11)
    ax2.set_xlabel("Desil"); ax2.set_ylabel("Skor")
    ax2.legend(fontsize=8, framealpha=0.2, facecolor='#1a1635', edgecolor='#3d3666')
    ax2.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

# ==============================================================
# TAB 2 — ANALISIS SOAL
# ==============================================================
with tab2:
    tingkat_kesukaran = data[soal_cols].mean()
    max_val = data[soal_cols].max().mean()

    # Tingkat kesukaran
    tk_norm = tingkat_kesukaran / tingkat_kesukaran.max()
    kategori_tk = []
    for v in tk_norm:
        if v >= 0.7:   kategori_tk.append(("Mudah", "#10b981"))
        elif v >= 0.4: kategori_tk.append(("Sedang", "#f59e0b"))
        else:          kategori_tk.append(("Sulit", "#ef4444"))

    # Daya pembeda
    daya_pembeda = pd.Series({c: data[c].corr(data["Total_Skor"]) for c in soal_cols})

    c1, c2 = st.columns(2)

    with c1:
        # Tingkat kesukaran — Horizontal bar dengan kategori warna
        fig, ax = plt.subplots(figsize=(6, 5.5))
        y_pos = np.arange(len(soal_cols))
        bar_colors = [kat[1] for kat in kategori_tk]
        bars = ax.barh(y_pos, tk_norm, color=bar_colors, alpha=0.85, edgecolor='#1a1635', linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(soal_cols, fontsize=8)
        ax.axvline(0.7, color='#10b981', linestyle='--', alpha=0.5, linewidth=1, label='Mudah (0.7)')
        ax.axvline(0.4, color='#f59e0b', linestyle='--', alpha=0.5, linewidth=1, label='Sedang (0.4)')
        ax.set_xlabel("Indeks Kesukaran (0–1)")
        ax.set_title("Tingkat Kesukaran Per Soal", fontsize=12)
        legend_patches = [
            mpatches.Patch(color='#10b981', label='Mudah (≥0.7)'),
            mpatches.Patch(color='#f59e0b', label='Sedang (0.4–0.7)'),
            mpatches.Patch(color='#ef4444', label='Sulit (<0.4)'),
        ]
        ax.legend(handles=legend_patches, fontsize=7.5, framealpha=0.2,
                  facecolor='#1a1635', edgecolor='#3d3666', loc='lower right')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)

    with c2:
        # Daya pembeda — warna berdasarkan kualitas
        dp_colors = []
        for v in daya_pembeda:
            if v >= 0.4:   dp_colors.append('#8b5cf6')
            elif v >= 0.3: dp_colors.append('#3b82f6')
            elif v >= 0.2: dp_colors.append('#f59e0b')
            else:          dp_colors.append('#ef4444')

        fig, ax = plt.subplots(figsize=(6, 5.5))
        ax.barh(range(len(soal_cols)), daya_pembeda.values, color=dp_colors,
                alpha=0.85, edgecolor='#1a1635', linewidth=0.5)
        ax.set_yticks(range(len(soal_cols)))
        ax.set_yticklabels(soal_cols, fontsize=8)
        ax.axvline(0.4, color='#8b5cf6', linestyle='--', alpha=0.6, linewidth=1)
        ax.axvline(0.3, color='#3b82f6', linestyle='--', alpha=0.6, linewidth=1)
        ax.axvline(0.2, color='#f59e0b', linestyle='--', alpha=0.6, linewidth=1)
        ax.set_xlabel("Korelasi Item-Total")
        ax.set_title("Daya Pembeda Per Soal", fontsize=12)
        legend_patches = [
            mpatches.Patch(color='#8b5cf6', label='Sangat Baik (≥0.4)'),
            mpatches.Patch(color='#3b82f6', label='Baik (0.3–0.4)'),
            mpatches.Patch(color='#f59e0b', label='Cukup (0.2–0.3)'),
            mpatches.Patch(color='#ef4444', label='Jelek (<0.2)'),
        ]
        ax.legend(handles=legend_patches, fontsize=7.5, framealpha=0.2,
                  facecolor='#1a1635', edgecolor='#3d3666', loc='lower right')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)

    # Scatter: kesukaran vs daya pembeda (kualitas soal 2D)
    st.markdown("---")
    c3, c4 = st.columns([3, 2])

    with c3:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        scatter_colors = [kat[1] for kat in kategori_tk]
        sc = ax.scatter(tk_norm, daya_pembeda.values, c=daya_pembeda.values,
                        cmap='plasma', s=100, alpha=0.85, edgecolors='white', linewidth=0.8, zorder=5)
        plt.colorbar(sc, ax=ax, label='Daya Pembeda', shrink=0.8)
        for i, (x, y) in enumerate(zip(tk_norm, daya_pembeda.values)):
            ax.annotate(f'S{i+1}', (x, y), fontsize=7.5, color='#e2e8f0',
                        xytext=(4, 4), textcoords='offset points')
        ax.axhline(0.3, color='#f59e0b', linestyle='--', alpha=0.4, linewidth=1)
        ax.axvline(0.4, color='#f59e0b', linestyle='--', alpha=0.4, linewidth=1)
        # Quadrant labels
        ax.text(0.05, 0.92, 'Sulit &\nBaik', transform=ax.transAxes, fontsize=8,
                color='#a78bfa', alpha=0.6)
        ax.text(0.72, 0.92, 'Mudah &\nBaik', transform=ax.transAxes, fontsize=8,
                color='#10b981', alpha=0.6)
        ax.text(0.05, 0.06, 'Sulit &\nJelek', transform=ax.transAxes, fontsize=8,
                color='#ef4444', alpha=0.6)
        ax.text(0.72, 0.06, 'Mudah &\nJelek', transform=ax.transAxes, fontsize=8,
                color='#f59e0b', alpha=0.6)
        ax.set_xlabel("Indeks Kesukaran")
        ax.set_ylabel("Daya Pembeda (r)")
        ax.set_title("Peta Kualitas Soal", fontsize=12)
        ax.grid(alpha=0.25)
        plt.tight_layout()
        st.pyplot(fig)

    with c4:
        # Heatmap korelasi mini
        fig, ax = plt.subplots(figsize=(4, 4.5))
        corr = data[soal_cols].corr()
        im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax.set_xticks(range(20)); ax.set_yticks(range(20))
        ax.set_xticklabels([f'S{i+1}' for i in range(20)], fontsize=6, rotation=90)
        ax.set_yticklabels([f'S{i+1}' for i in range(20)], fontsize=6)
        plt.colorbar(im, ax=ax, shrink=0.8, label='r')
        ax.set_title("Korelasi Antar Soal", fontsize=11)
        plt.tight_layout()
        st.pyplot(fig)

    # Ringkasan kualitas soal
    st.markdown("---")
    col_a, col_b, col_c, col_d = st.columns(4)
    n_mudah = sum(1 for k in kategori_tk if k[0] == 'Mudah')
    n_sedang = sum(1 for k in kategori_tk if k[0] == 'Sedang')
    n_sulit = sum(1 for k in kategori_tk if k[0] == 'Sulit')
    n_baik = sum(1 for v in daya_pembeda if v >= 0.3)
    col_a.metric("Soal Mudah", n_mudah)
    col_b.metric("Soal Sedang", n_sedang)
    col_c.metric("Soal Sulit", n_sulit)
    col_d.metric("Soal Berkualitas (r≥0.3)", n_baik)

# ==============================================================
# TAB 3 — SEGMENTASI SISWA
# ==============================================================
with tab3:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data[soal_cols])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled)
    data["Cluster"] = cluster_labels

    # Beri nama kelompok berdasarkan rata-rata skor
    cluster_means = data.groupby("Cluster")["Total_Skor"].mean().sort_values(ascending=False)
    rank_map = {orig: rank for rank, orig in enumerate(cluster_means.index)}
    cluster_names = {0: "🏆 Tinggi", 1: "📘 Sedang", 2: "💪 Perlu Bantuan"}
    if n_clusters == 4:
        cluster_names[3] = "🌱 Pemula"
    elif n_clusters == 5:
        cluster_names[3] = "🌱 Pemula"; cluster_names[4] = "⚠️ Kritis"
    data["Kelompok"] = data["Cluster"].map(lambda c: cluster_names.get(rank_map.get(c, c), f"Grup {c+1}"))

    # PCA untuk visualisasi 2D
    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(scaled)
    var_exp = pca.explained_variance_ratio_

    c1, c2 = st.columns([3, 2])

    with c1:
        # Scatter PCA
        fig, ax = plt.subplots(figsize=(6.5, 4.8))
        for k in range(n_clusters):
            mask = cluster_labels == k
            rank = rank_map.get(k, k)
            color = CLUSTER_COLORS[rank % len(CLUSTER_COLORS)]
            ax.scatter(pca_result[mask, 0], pca_result[mask, 1],
                       color=color, s=70, alpha=0.75, edgecolors='white',
                       linewidth=0.5, label=cluster_names.get(rank, f'Grup {rank+1}'), zorder=5)
            # Centroid
            cx, cy = pca_result[mask, 0].mean(), pca_result[mask, 1].mean()
            ax.scatter(cx, cy, color=color, s=200, marker='*', edgecolors='white',
                       linewidth=1.5, zorder=6)

        ax.set_xlabel(f"PCA-1 ({var_exp[0]*100:.1f}% variansi)", fontsize=9)
        ax.set_ylabel(f"PCA-2 ({var_exp[1]*100:.1f}% variansi)", fontsize=9)
        ax.set_title("Peta Sebaran Kemampuan Siswa (PCA)", fontsize=12)
        ax.legend(fontsize=9, framealpha=0.2, facecolor='#1a1635', edgecolor='#3d3666')
        ax.grid(alpha=0.2)
        plt.tight_layout()
        st.pyplot(fig)

    with c2:
        # Pie komposisi
        fig, ax = plt.subplots(figsize=(4, 4.8))
        vc = data["Kelompok"].value_counts()
        pie_colors = [CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in range(len(vc))]
        wedges, texts, autotexts = ax.pie(
            vc.values, labels=vc.index, colors=pie_colors,
            autopct='%1.1f%%', startangle=90,
            wedgeprops=dict(edgecolor='#1a1635', linewidth=2),
            pctdistance=0.75
        )
        for t in texts: t.set_fontsize(8.5); t.set_color('#d1d5db')
        for at in autotexts: at.set_fontsize(8); at.set_color('white'); at.set_fontweight('bold')
        ax.set_title("Komposisi Kelompok", fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)

    # Profil radar per kelompok
    st.markdown("---")

    # Radar chart per kelompok (ambil 8 soal representatif)
    sample_soals = soal_cols[:8]
    group_profile = data.groupby("Cluster")[sample_soals].mean()

    fig, axes = plt.subplots(1, min(n_clusters, 3), figsize=(11, 3.8),
                              subplot_kw=dict(polar=True))
    if n_clusters == 1: axes = [axes]

    angles = np.linspace(0, 2*np.pi, len(sample_soals), endpoint=False).tolist()
    angles += angles[:1]

    for i, (cid, row) in enumerate(group_profile.iterrows()):
        if i >= 3: break
        ax = axes[i] if n_clusters > 1 else axes[0]
        rank = rank_map.get(cid, cid)
        color = CLUSTER_COLORS[rank % len(CLUSTER_COLORS)]
        vals = row.values.tolist(); vals += vals[:1]
        ax.plot(angles, vals, color=color, linewidth=2)
        ax.fill(angles, vals, color=color, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([f'S{j+1}' for j in range(len(sample_soals))], fontsize=8, color='#9ca3af')
        ax.set_yticklabels([]); ax.grid(color='#3d3666', alpha=0.5)
        ax.set_facecolor('#1a1635')
        ax.spines['polar'].set_color('#3d3666')
        ax.set_title(cluster_names.get(rank, f'Grup {rank+1}'), fontsize=10,
                     color=color, pad=12, fontweight='bold')
    plt.suptitle("Profil Kemampuan Per Soal (8 Soal Pertama)", y=1.02,
                 fontsize=12, color='#f3f4f6')
    plt.tight_layout()
    st.pyplot(fig)

    # Tabel summary kelompok
    st.markdown("---")
    summary = data.groupby("Kelompok")["Total_Skor"].agg(['count','mean','min','max','std']).round(2)
    summary.columns = ['Jumlah','Rata-rata','Min','Maks','Std Dev']
    st.dataframe(summary.style.background_gradient(cmap='plasma', subset=['Rata-rata']),
                 use_container_width=True)

# ==============================================================
# TAB 4 — REKOMENDASI
# ==============================================================
with tab4:
    # Soal bermasalah
    soal_jelek = daya_pembeda[daya_pembeda < 0.2].index.tolist()
    soal_terlalu_mudah = tk_norm[tk_norm > 0.85].index.tolist()
    soal_terlalu_sulit = tk_norm[tk_norm < 0.25].index.tolist()

    st.markdown("### 🔬 Evaluasi Kualitas Soal")
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        if soal_jelek:
            st.markdown(f"""<div class="insight-box insight-danger">
            ⚠️ <b>{len(soal_jelek)} soal berdaya pembeda rendah (r &lt; 0.2)</b> — perlu direvisi atau diganti:<br>
            <code>{', '.join(soal_jelek)}</code>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown('<div class="insight-box insight-success">✅ Semua soal memiliki daya pembeda yang memadai.</div>', unsafe_allow_html=True)

        if soal_terlalu_mudah:
            st.markdown(f"""<div class="insight-box insight-warning">
            📝 <b>{len(soal_terlalu_mudah)} soal terlalu mudah</b> (indeks &gt; 0.85): <code>{', '.join(soal_terlalu_mudah)}</code>
            </div>""", unsafe_allow_html=True)

        if soal_terlalu_sulit:
            st.markdown(f"""<div class="insight-box insight-danger">
            🔴 <b>{len(soal_terlalu_sulit)} soal terlalu sulit</b> (indeks &lt; 0.25): <code>{', '.join(soal_terlalu_sulit)}</code>
            </div>""", unsafe_allow_html=True)

    with col_r2:
        # Gauge visualisasi reliabilitas (Cronbach's Alpha)
        n = len(soal_cols)
        item_var = data[soal_cols].var()
        total_var = data["Total_Skor"].var()
        alpha = (n / (n - 1)) * (1 - item_var.sum() / total_var)
        alpha = round(alpha, 3)

        fig, ax = plt.subplots(figsize=(4.5, 2.8))
        ax.set_xlim(0, 1); ax.set_ylim(-0.1, 1.1)
        ax.set_facecolor('#1a1635')

        # Gauge bar
        zones = [(0.0, 0.5, '#ef4444', 'Tidak Reliabel'),
                 (0.5, 0.6, '#f97316', 'Cukup'),
                 (0.6, 0.7, '#f59e0b', 'Sedang'),
                 (0.7, 0.8, '#3b82f6', 'Baik'),
                 (0.8, 1.0, '#10b981', 'Sangat Baik')]
        for xmin, xmax, col, lbl in zones:
            ax.barh(0.5, xmax - xmin, left=xmin, height=0.35, color=col, alpha=0.8, edgecolor='none')
            ax.text((xmin+xmax)/2, 0.1, lbl, ha='center', va='center', fontsize=7, color='#9ca3af')

        # Needle
        ax.annotate('', xy=(alpha, 0.5), xytext=(alpha, 0.95),
                    arrowprops=dict(arrowstyle='->', color='white', lw=2.5))
        ax.text(alpha, 1.05, f'α = {alpha}', ha='center', va='bottom',
                fontsize=13, fontweight='bold', color='white')
        ax.set_xticks([]); ax.set_yticks([])
        ax.spines[:].set_visible(False)
        ax.set_title("Reliabilitas Tes (Cronbach's Alpha)", fontsize=11, color='#f3f4f6')
        plt.tight_layout()
        st.pyplot(fig)

    # Rekomendasi tindak lanjut per kelompok
    st.markdown("### 👨‍🏫 Strategi Tindak Lanjut Per Kelompok")
    if "Kelompok" in data.columns:
        for k_name in data["Kelompok"].unique():
            k_data = data[data["Kelompok"] == k_name]
            avg = k_data["Total_Skor"].mean()
            pct = avg / max_possible * 100

            if "Tinggi" in k_name:
                color = "success"; icon = "🏆"
                rec = "Berikan pengayaan dan soal-soal HOTS. Dapat dijadikan tutor sebaya."
            elif "Sedang" in k_name:
                color = "info"; icon = "📘"
                rec = "Berikan latihan tambahan pada soal dengan daya pembeda tinggi. Review materi inti."
            else:
                color = "warning"; icon = "💪"
                rec = "Prioritaskan remedial. Identifikasi soal yang paling banyak salah. Pendampingan intensif."

            st.markdown(f"""<div class="insight-box insight-{color}">
            {icon} <b>{k_name}</b> ({len(k_data)} siswa) — Rata-rata: <b>{avg:.1f}</b> ({pct:.1f}%)<br>
            📌 {rec}
            </div>""", unsafe_allow_html=True)

# ==============================================================
# TAB 5 — LAPORAN LENGKAP
# ==============================================================
with tab5:
    st.markdown("### 📋 Tabel Skor Per Siswa")

    display_df = data[soal_cols + ["Total_Skor", "Persentase", "Status"]].copy()
    if "Kelompok" in data.columns:
        display_df["Kelompok"] = data["Kelompok"]

    def color_status(val):
        if val == "Lulus": return 'color: #10b981; font-weight: bold'
        return 'color: #ef4444; font-weight: bold'

    styled = display_df.style.applymap(color_status, subset=["Status"]) \
        .background_gradient(subset=["Total_Skor"], cmap='plasma') \
        .format({"Persentase": "{:.1f}%"})

    st.dataframe(styled, use_container_width=True, height=400)

    # Download CSV
    csv = display_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download Hasil Analisis (CSV)",
        data=csv,
        file_name="hasil_analisis_belajar.csv",
        mime="text/csv"
    )

    # Summary statistik lengkap
    st.markdown("---")
    st.markdown("### 📊 Statistik Deskriptif Lengkap")
    st.dataframe(data[soal_cols + ["Total_Skor"]].describe().round(3).T,
                 use_container_width=True)

# ======================================================
# FOOTER
# ======================================================
st.markdown("---")
st.markdown("""
<div style='text-align:center; font-size:0.75rem; color:rgba(196,181,253,0.4); padding:8px 0;'>
EduInsight Dashboard &nbsp;·&nbsp; Analisis Psikometri Hasil Belajar Siswa
</div>
""", unsafe_allow_html=True)
