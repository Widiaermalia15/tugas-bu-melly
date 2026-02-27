import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ======================================================
# KONFIGURASI HALAMAN
# ======================================================
st.set_page_config(
    page_title="Dashboard Analisis Hasil Belajar Siswa",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================
# CUSTOM CSS — Dark Academic / Refined Minimal
# ======================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0f0f13;
    color: #e8e4dc;
}

.stApp {
    background: linear-gradient(135deg, #0f0f13 0%, #14141c 50%, #0f0f13 100%);
}

/* HEADER */
.main-title {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    font-weight: 900;
    letter-spacing: -0.02em;
    background: linear-gradient(90deg, #f5c842 0%, #e8964e 60%, #d4567a 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.25rem;
}

.sub-title {
    font-family: 'DM Sans', sans-serif;
    font-size: 1rem;
    font-weight: 300;
    color: #8a8580;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 2.5rem;
}

/* METRIC CARDS */
.metric-card {
    background: linear-gradient(145deg, #1a1a24, #16161e);
    border: 1px solid #2a2a38;
    border-radius: 16px;
    padding: 1.5rem 1.25rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s, border-color 0.2s;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #f5c842, #e8964e);
}
.metric-card:hover {
    transform: translateY(-3px);
    border-color: #f5c84255;
}
.metric-label {
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #8a8580;
    margin-bottom: 0.5rem;
}
.metric-value {
    font-family: 'Playfair Display', serif;
    font-size: 2.25rem;
    font-weight: 700;
    color: #f5c842;
    line-height: 1;
}

/* SECTION HEADERS */
.section-header {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #e8e4dc;
    margin-top: 2.5rem;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #2a2a38;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* BADGE */
.badge-mudah {
    background: linear-gradient(90deg, #22c55e22, #22c55e11);
    border: 1px solid #22c55e55;
    border-radius: 8px;
    padding: 0.75rem 1.25rem;
    color: #4ade80;
    font-size: 0.9rem;
    font-weight: 500;
}
.badge-sulit {
    background: linear-gradient(90deg, #ef444422, #ef444411);
    border: 1px solid #ef444455;
    border-radius: 8px;
    padding: 0.75rem 1.25rem;
    color: #f87171;
    font-size: 0.9rem;
    font-weight: 500;
}

/* DATAFRAME */
.stDataFrame {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid #2a2a38;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: #12121a;
    border-right: 1px solid #2a2a38;
}
section[data-testid="stSidebar"] * {
    color: #c8c4bc !important;
}

/* UPLOAD AREA */
.stFileUploader {
    background: #1a1a24;
    border-radius: 12px;
    border: 2px dashed #2a2a38;
    padding: 1rem;
}

/* TABS */
.stTabs [data-baseweb="tab-list"] {
    background: #14141c;
    border-radius: 10px;
    gap: 4px;
    padding: 4px;
    border: 1px solid #2a2a38;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 7px;
    color: #8a8580;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.875rem;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1e1e2e, #252535) !important;
    color: #f5c842 !important;
    border: 1px solid #f5c84233;
}

/* EXPANDER */
.stExpander {
    background: #1a1a24;
    border: 1px solid #2a2a38;
    border-radius: 12px;
}

/* SUCCESS / ERROR */
.element-container .stAlert {
    border-radius: 12px;
    border: none;
}

/* SCROLLBAR */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0f0f13; }
::-webkit-scrollbar-thumb { background: #2a2a38; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ======================================================
# MATPLOTLIB DARK THEME
# ======================================================
plt.rcParams.update({
    'figure.facecolor': '#1a1a24',
    'axes.facecolor': '#1a1a24',
    'axes.edgecolor': '#2a2a38',
    'axes.labelcolor': '#c8c4bc',
    'xtick.color': '#8a8580',
    'ytick.color': '#8a8580',
    'text.color': '#e8e4dc',
    'grid.color': '#2a2a38',
    'grid.linestyle': '--',
    'grid.alpha': 0.5,
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

GOLD = '#f5c842'
ORANGE = '#e8964e'
ROSE = '#d4567a'
TEAL = '#4ecdc4'
PURPLE = '#9b59b6'
GREEN = '#2ecc71'

# ======================================================
# HELPER — generate sample data
# ======================================================
def generate_sample_data():
    np.random.seed(42)
    n = 50
    cols = {f"Soal_{i+1}": np.random.randint(0, 2, n) for i in range(20)}
    return pd.DataFrame(cols)

# ======================================================
# HEADER
# ======================================================
st.markdown('<div class="main-title">📚 Dashboard Analisis<br>Hasil Belajar Siswa</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Analisis Interaktif · 50 Siswa · 20 Soal</div>', unsafe_allow_html=True)

# ======================================================
# SIDEBAR
# ======================================================
with st.sidebar:
    st.markdown("### ⚙️ Pengaturan")
    st.markdown("---")
    n_clusters = st.slider("Jumlah Cluster Siswa", min_value=2, max_value=6, value=3)
    kkm = st.number_input("KKM (Nilai Minimal Lulus)", min_value=0, max_value=100, value=60)
    st.markdown("---")
    show_raw = st.checkbox("Tampilkan Data Mentah", value=False)
    show_desc = st.checkbox("Tampilkan Statistik Deskriptif", value=True)
    st.markdown("---")
    st.markdown("**Cara Penggunaan:**")
    st.markdown("1. Upload file Excel berisi data siswa (50 baris × 20 kolom soal)\n2. Setiap sel berisi skor per soal (misal 0 atau 1)\n3. Dashboard akan otomatis menganalisis data Anda")
    st.markdown("---")
    use_sample = st.button("🧪 Gunakan Data Sampel", use_container_width=True)

# ======================================================
# UPLOAD / SAMPLE
# ======================================================
if "df_loaded" not in st.session_state:
    st.session_state.df_loaded = None

uploaded_file = st.file_uploader("📂 Upload File Excel (50 siswa × 20 soal)", type=["xlsx"])

if uploaded_file is not None:
    st.session_state.df_loaded = pd.read_excel(uploaded_file)
elif use_sample:
    st.session_state.df_loaded = generate_sample_data()
    st.success("✅ Data sampel berhasil dimuat! (50 siswa × 20 soal, nilai acak 0–1)")

# ======================================================
# ANALYSIS
# ======================================================
if st.session_state.df_loaded is not None:
    df = st.session_state.df_loaded
    df_numeric = df.select_dtypes(include=np.number)

    if df_numeric.shape[1] < 20:
        st.error("❌ File harus memiliki minimal 20 kolom soal numerik.")
        st.stop()

    data = df_numeric.iloc[:, :20].copy()
    data.columns = [f"Soal_{i+1}" for i in range(20)]
    data["Total_Skor"] = data.sum(axis=1)

    if show_raw:
        with st.expander("📋 Lihat Data Mentah"):
            st.dataframe(df, use_container_width=True)

    if show_desc:
        with st.expander("📊 Statistik Deskriptif"):
            st.dataframe(data.describe().round(3), use_container_width=True)

    # ======================================================
    # SECTION 1 — STATISTIK NILAI
    # ======================================================
    st.markdown('<div class="section-header">1️⃣ &nbsp;Statistik Nilai Siswa</div>', unsafe_allow_html=True)

    lulus = (data["Total_Skor"] >= kkm).sum()
    tidak_lulus = len(data) - lulus
    std = round(data["Total_Skor"].std(), 2)
    median = data["Total_Skor"].median()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    metrics = [
        ("Jumlah Siswa", len(data)),
        ("Rata-rata", round(data["Total_Skor"].mean(), 2)),
        ("Nilai Tertinggi", data["Total_Skor"].max()),
        ("Nilai Terendah", data["Total_Skor"].min()),
        (f"Lulus (≥{kkm})", lulus),
        ("Standar Deviasi", std),
    ]
    for col, (label, value) in zip([c1,c2,c3,c4,c5,c6], metrics):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        fig, ax = plt.subplots(figsize=(7, 4))
        n_bins = 10
        counts, bins, patches = ax.hist(data["Total_Skor"], bins=n_bins, edgecolor='none', alpha=0.9)
        cmap_hist = LinearSegmentedColormap.from_list("gh", [ROSE, GOLD])
        for i, patch in enumerate(patches):
            patch.set_facecolor(cmap_hist(i / n_bins))
        ax.axvline(data["Total_Skor"].mean(), color=GOLD, linewidth=2, linestyle='--', label=f'Mean: {round(data["Total_Skor"].mean(),1)}')
        ax.axvline(kkm, color=ROSE, linewidth=2, linestyle=':', label=f'KKM: {kkm}')
        ax.set_title("Distribusi Skor Total", fontweight='bold', fontsize=13, color='#e8e4dc', pad=12)
        ax.set_xlabel("Skor", labelpad=8)
        ax.set_ylabel("Frekuensi", labelpad=8)
        ax.legend(framealpha=0, labelcolor='#c8c4bc', fontsize=9)
        ax.grid(True, axis='y')
        fig.tight_layout()
        st.pyplot(fig)

    with col_b:
        fig, ax = plt.subplots(figsize=(7, 4))
        bp = ax.boxplot(data["Total_Skor"], vert=False, patch_artist=True, widths=0.5,
                        medianprops={'color': GOLD, 'linewidth': 2.5},
                        boxprops={'facecolor': '#f5c84222', 'edgecolor': GOLD, 'linewidth': 1.5},
                        whiskerprops={'color': '#c8c4bc', 'linewidth': 1.5},
                        capprops={'color': '#c8c4bc', 'linewidth': 2},
                        flierprops={'marker': 'o', 'color': ROSE, 'markersize': 7, 'alpha': 0.7})
        ax.set_title("Boxplot Skor Total", fontweight='bold', fontsize=13, color='#e8e4dc', pad=12)
        ax.set_xlabel("Skor", labelpad=8)
        ax.set_yticks([])
        ax.grid(True, axis='x')
        fig.tight_layout()
        st.pyplot(fig)

    # Lulus vs Tidak Lulus bar
    col_c, col_d = st.columns(2)
    with col_c:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        bars = ax.bar(["Lulus", "Tidak Lulus"], [lulus, tidak_lulus], color=[GREEN, ROSE], edgecolor='none',
                       width=0.5, alpha=0.9)
        for bar, val in zip(bars, [lulus, tidak_lulus]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, str(val),
                    ha='center', va='bottom', fontsize=13, fontweight='bold', color='#e8e4dc')
        ax.set_title(f"Kelulusan Siswa (KKM = {kkm})", fontweight='bold', fontsize=12, color='#e8e4dc', pad=10)
        ax.set_ylabel("Jumlah Siswa", labelpad=8)
        ax.grid(True, axis='y')
        fig.tight_layout()
        st.pyplot(fig)

    with col_d:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        bins_all = range(int(data["Total_Skor"].min()), int(data["Total_Skor"].max())+2)
        ax.hist(data["Total_Skor"], bins=bins_all, color=TEAL, alpha=0.7, edgecolor='none')
        ax.axvline(median, color=GOLD, linewidth=2, linestyle='--', label=f'Median: {median}')
        ax.set_title("Frekuensi per Nilai", fontweight='bold', fontsize=12, color='#e8e4dc', pad=10)
        ax.set_xlabel("Skor", labelpad=8)
        ax.set_ylabel("Frekuensi", labelpad=8)
        ax.legend(framealpha=0, labelcolor='#c8c4bc', fontsize=9)
        ax.grid(True, axis='y')
        fig.tight_layout()
        st.pyplot(fig)

    # ======================================================
    # SECTION 2 — TINGKAT KESUKARAN
    # ======================================================
    st.markdown('<div class="section-header">2️⃣ &nbsp;Analisis Tingkat Kesukaran Soal</div>', unsafe_allow_html=True)

    tingkat_kesukaran = data.iloc[:, :20].mean()

    def kategori_kesukaran(p):
        if p < 0.3: return "Sulit", ROSE
        elif p < 0.7: return "Sedang", GOLD
        else: return "Mudah", GREEN

    kat = [kategori_kesukaran(v) for v in tingkat_kesukaran]
    colors_bar = [c for _, c in kat]

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(tingkat_kesukaran.index, tingkat_kesukaran.values, color=colors_bar, alpha=0.85, edgecolor='none', width=0.65)
    ax.axhline(0.3, color=ROSE, linestyle='--', linewidth=1.5, alpha=0.7, label='Batas Sulit (0.3)')
    ax.axhline(0.7, color=GREEN, linestyle='--', linewidth=1.5, alpha=0.7, label='Batas Mudah (0.7)')
    ax.axhline(tingkat_kesukaran.mean(), color='#c8c4bc', linestyle=':', linewidth=1, alpha=0.6, label=f'Rata-rata ({round(tingkat_kesukaran.mean(),2)})')
    ax.set_title("Tingkat Kesukaran Tiap Soal", fontweight='bold', fontsize=14, color='#e8e4dc', pad=12)
    ax.set_ylabel("P-Value (Proporsi Jawaban Benar)", labelpad=8)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(range(len(tingkat_kesukaran)))
    ax.set_xticklabels(tingkat_kesukaran.index, rotation=45, ha='right', fontsize=9)
    legend_patches = [
        mpatches.Patch(color=ROSE, label='Sulit (< 0.30)'),
        mpatches.Patch(color=GOLD, label='Sedang (0.30–0.70)'),
        mpatches.Patch(color=GREEN, label='Mudah (> 0.70)'),
    ]
    ax.legend(handles=legend_patches, framealpha=0, labelcolor='#c8c4bc', fontsize=9, loc='upper right')
    ax.grid(True, axis='y')
    fig.tight_layout()
    st.pyplot(fig)

    soal_mudah = tingkat_kesukaran.idxmax()
    soal_sulit = tingkat_kesukaran.idxmin()
    n_mudah = sum(1 for k,_ in kat if k=='Mudah')
    n_sedang = sum(1 for k,_ in kat if k=='Sedang')
    n_sulit = sum(1 for k,_ in kat if k=='Sulit')

    col_e, col_f, col_g, col_h = st.columns(4)
    col_e.markdown(f'<div class="badge-mudah">✅ <b>Soal Termudah:</b> {soal_mudah}<br>P = {round(tingkat_kesukaran[soal_mudah],3)}</div>', unsafe_allow_html=True)
    col_f.markdown(f'<div class="badge-sulit">🔴 <b>Soal Tersulit:</b> {soal_sulit}<br>P = {round(tingkat_kesukaran[soal_sulit],3)}</div>', unsafe_allow_html=True)
    col_g.markdown(f'<div class="metric-card"><div class="metric-label">Kategori Mudah</div><div class="metric-value" style="color:{GREEN}">{n_mudah}</div></div>', unsafe_allow_html=True)
    col_h.markdown(f'<div class="metric-card"><div class="metric-label">Kategori Sulit</div><div class="metric-value" style="color:{ROSE}">{n_sulit}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ======================================================
    # SECTION 3 — DAYA PEMBEDA
    # ======================================================
    st.markdown('<div class="section-header">3️⃣ &nbsp;Analisis Daya Pembeda (Korelasi Item-Total)</div>', unsafe_allow_html=True)

    daya_pembeda = pd.Series({col: data[col].corr(data["Total_Skor"]) for col in data.columns[:20]})

    def dp_color(v):
        if v >= 0.4: return GREEN
        elif v >= 0.3: return TEAL
        elif v >= 0.2: return GOLD
        else: return ROSE

    dp_colors = [dp_color(v) for v in daya_pembeda.values]

    col_i, col_j = st.columns([2, 1])
    with col_i:
        fig, ax = plt.subplots(figsize=(12, 4.5))
        bars = ax.bar(daya_pembeda.index, daya_pembeda.values, color=dp_colors, alpha=0.85, edgecolor='none', width=0.65)
        ax.axhline(0.3, color=GOLD, linestyle='--', linewidth=2, alpha=0.8, label='Batas Baik (r = 0.30)')
        ax.axhline(0, color='#c8c4bc', linestyle='-', linewidth=0.8, alpha=0.4)
        ax.set_title("Daya Pembeda Tiap Soal (Korelasi Biserial)", fontweight='bold', fontsize=13, color='#e8e4dc', pad=12)
        ax.set_ylabel("Koefisien Korelasi r", labelpad=8)
        ax.set_xticks(range(len(daya_pembeda)))
        ax.set_xticklabels(daya_pembeda.index, rotation=45, ha='right', fontsize=9)
        ax.legend(framealpha=0, labelcolor='#c8c4bc', fontsize=9)
        ax.grid(True, axis='y')
        fig.tight_layout()
        st.pyplot(fig)

    with col_j:
        soal_terbaik = daya_pembeda.idxmax()
        fig, ax = plt.subplots(figsize=(5, 4))
        sc = ax.scatter(data[soal_terbaik], data["Total_Skor"],
                        c=data["Total_Skor"], cmap='YlOrRd', alpha=0.8, s=60, edgecolors='none')
        plt.colorbar(sc, ax=ax, label='Total Skor')
        ax.set_xlabel(f"{soal_terbaik} (jawaban)", labelpad=8)
        ax.set_ylabel("Total Skor", labelpad=8)
        ax.set_title(f"Scatter: {soal_terbaik} (Terbaik)", fontweight='bold', fontsize=11, color='#e8e4dc', pad=10)
        ax.grid(True)
        fig.tight_layout()
        st.pyplot(fig)

    n_baik = sum(1 for v in daya_pembeda if v >= 0.3)
    n_cukup = sum(1 for v in daya_pembeda if 0.2 <= v < 0.3)
    n_kurang = sum(1 for v in daya_pembeda if v < 0.2)

    st.markdown(f"""
    <div style="display:flex; gap:1rem; margin-top:1rem; flex-wrap:wrap;">
        <div class="metric-card" style="flex:1; min-width:140px;">
            <div class="metric-label">Soal Terbaik</div>
            <div class="metric-value" style="font-size:1.5rem; color:{GREEN}">{soal_terbaik}</div>
            <div style="color:#8a8580; font-size:0.8rem">r = {round(daya_pembeda[soal_terbaik],3)}</div>
        </div>
        <div class="metric-card" style="flex:1; min-width:140px;">
            <div class="metric-label">Daya Beda Baik (≥0.3)</div>
            <div class="metric-value" style="color:{GREEN}">{n_baik}</div>
        </div>
        <div class="metric-card" style="flex:1; min-width:140px;">
            <div class="metric-label">Daya Beda Cukup</div>
            <div class="metric-value" style="color:{GOLD}">{n_cukup}</div>
        </div>
        <div class="metric-card" style="flex:1; min-width:140px;">
            <div class="metric-label">Daya Beda Kurang</div>
            <div class="metric-value" style="color:{ROSE}">{n_kurang}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ======================================================
    # SECTION 4 — HEATMAP KORELASI
    # ======================================================
    st.markdown('<div class="section-header">4️⃣ &nbsp;Korelasi Antar Soal</div>', unsafe_allow_html=True)

    corr = data.iloc[:, :20].corr()

    col_k, col_l = st.columns([3, 2])
    with col_k:
        cmap_custom = LinearSegmentedColormap.from_list("custom", ['#d4567a', '#1a1a24', '#4ecdc4'])
        fig, ax = plt.subplots(figsize=(9, 7))
        im = ax.imshow(corr.values, cmap=cmap_custom, vmin=-1, vmax=1, aspect='auto')
        cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
        cbar.set_label('Korelasi', color='#c8c4bc', labelpad=8)
        cbar.ax.yaxis.set_tick_params(color='#8a8580')
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#8a8580')
        ax.set_xticks(range(20))
        ax.set_xticklabels([f"S{i+1}" for i in range(20)], rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(20))
        ax.set_yticklabels([f"S{i+1}" for i in range(20)], fontsize=8)
        ax.set_title("Heatmap Korelasi Antar Soal", fontweight='bold', fontsize=13, color='#e8e4dc', pad=12)
        fig.tight_layout()
        st.pyplot(fig)

    with col_l:
        # Top correlations
        corr_flat = corr.where(np.tril(np.ones(corr.shape), k=-1).astype(bool)).stack()
        top_corr = corr_flat.abs().sort_values(ascending=False).head(10)
        st.markdown("**🔗 Pasangan Soal Berkorelasi Tinggi**")
        for (r, c), v in zip([i for i in top_corr.index], top_corr.values):
            bar_w = int(abs(v) * 100)
            col_c_ = GREEN if v > 0 else ROSE
            st.markdown(f"""
            <div style="margin:4px 0; font-size:0.82rem;">
                <span style="color:#c8c4bc">{r} × {c}</span>
                <span style="float:right; color:{col_c_}; font-weight:600">{round(v,3)}</span>
                <div style="background:#2a2a38; border-radius:4px; height:5px; margin-top:3px;">
                    <div style="background:{col_c_}; width:{bar_w}%; height:100%; border-radius:4px; opacity:0.8;"></div>
                </div>
            </div>""", unsafe_allow_html=True)

    # ======================================================
    # SECTION 5 — CLUSTERING
    # ======================================================
    st.markdown(f'<div class="section-header">5️⃣ &nbsp;Segmentasi Kemampuan Siswa ({n_clusters} Kelompok)</div>', unsafe_allow_html=True)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(data.iloc[:, :20])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster = kmeans.fit_predict(scaled)
    data["Cluster"] = cluster

    # Sort clusters by mean score for consistent labeling
    cluster_mean = data.groupby("Cluster")["Total_Skor"].mean().sort_values(ascending=False)
    rank_map = {old: new for new, old in enumerate(cluster_mean.index)}
    data["Cluster_Label"] = data["Cluster"].map(rank_map)
    label_names = {0: "Tinggi", 1: "Sedang", 2: "Rendah", 3: "Kelompok 4", 4: "Kelompok 5", 5: "Kelompok 6"}
    cluster_colors = [GOLD, TEAL, ROSE, PURPLE, GREEN, ORANGE]

    cluster_stats = data.groupby("Cluster_Label")["Total_Skor"].agg(["mean","min","max","count"]).round(2)

    col_m, col_n, col_o = st.columns(3)

    with col_m:
        fig, ax = plt.subplots(figsize=(6, 4))
        labels_sorted = sorted(data["Cluster_Label"].unique())
        means = [data[data["Cluster_Label"]==l]["Total_Skor"].mean() for l in labels_sorted]
        bar_labels = [label_names.get(l, f"Klp {l+1}") for l in labels_sorted]
        bars = ax.bar(bar_labels, means, color=[cluster_colors[l] for l in labels_sorted], alpha=0.85, width=0.55, edgecolor='none')
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, f'{round(val,1)}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold', color='#e8e4dc')
        ax.axhline(kkm, color=ROSE, linestyle='--', linewidth=1.5, alpha=0.7, label=f'KKM={kkm}')
        ax.set_title("Rata-rata Skor per Kelompok", fontweight='bold', fontsize=12, color='#e8e4dc', pad=10)
        ax.set_ylabel("Rata-rata Skor", labelpad=8)
        ax.legend(framealpha=0, labelcolor='#c8c4bc', fontsize=9)
        ax.grid(True, axis='y')
        fig.tight_layout()
        st.pyplot(fig)

    with col_n:
        sizes = data["Cluster_Label"].value_counts().sort_index()
        pie_labels = [label_names.get(l, f"Klp {l+1}") for l in sizes.index]
        pie_colors = [cluster_colors[l] for l in sizes.index]
        fig, ax = plt.subplots(figsize=(5, 4))
        wedges, texts, autotexts = ax.pie(sizes.values, labels=pie_labels, autopct='%1.1f%%',
                                           colors=pie_colors, startangle=90, pctdistance=0.75,
                                           wedgeprops={'edgecolor': '#0f0f13', 'linewidth': 2})
        for at in autotexts:
            at.set_color('#0f0f13')
            at.set_fontweight('bold')
            at.set_fontsize(9)
        for t in texts:
            t.set_color('#c8c4bc')
            t.set_fontsize(10)
        ax.set_title("Komposisi Kelompok", fontweight='bold', fontsize=12, color='#e8e4dc', pad=10)
        fig.tight_layout()
        st.pyplot(fig)

    with col_o:
        fig, ax = plt.subplots(figsize=(5, 4))
        for l in labels_sorted:
            subset = data[data["Cluster_Label"]==l]["Total_Skor"]
            ax.scatter([l]*len(subset) + np.random.normal(0,0.08,len(subset)), subset,
                       color=cluster_colors[l], alpha=0.6, s=40, edgecolors='none',
                       label=label_names.get(l, f"Klp {l+1}"))
        ax.set_xticks(labels_sorted)
        ax.set_xticklabels([label_names.get(l, f"Klp {l+1}") for l in labels_sorted])
        ax.set_ylabel("Total Skor", labelpad=8)
        ax.set_title("Sebaran Skor per Kelompok", fontweight='bold', fontsize=12, color='#e8e4dc', pad=10)
        ax.grid(True, axis='y')
        fig.tight_layout()
        st.pyplot(fig)

    st.markdown("**📊 Ringkasan Statistik Kelompok**")
    cluster_stats.index = [label_names.get(i, f"Klp {i+1}") for i in cluster_stats.index]
    cluster_stats.columns = ["Rata-rata", "Min", "Maks", "Jumlah Siswa"]
    st.dataframe(cluster_stats, use_container_width=True)

    # ======================================================
    # SECTION 6 — RANKING SOAL
    # ======================================================
    st.markdown('<div class="section-header">6️⃣ &nbsp;Ranking Kualitas Soal</div>', unsafe_allow_html=True)

    ranking = daya_pembeda.sort_values(ascending=False).reset_index()
    ranking.columns = ["Soal", "Daya Pembeda (r)"]
    ranking["P-Value"] = [round(tingkat_kesukaran[s], 3) for s in ranking["Soal"]]
    ranking["Kategori Kesukaran"] = [kategori_kesukaran(tingkat_kesukaran[s])[0] for s in ranking["Soal"]]
    ranking["Kualitas Daya Beda"] = ranking["Daya Pembeda (r)"].apply(
        lambda v: "Baik Sekali" if v >= 0.4 else ("Baik" if v >= 0.3 else ("Cukup" if v >= 0.2 else "Buruk")))
    ranking["Rank"] = range(1, len(ranking)+1)
    ranking = ranking.set_index("Rank")
    ranking["Daya Pembeda (r)"] = ranking["Daya Pembeda (r)"].round(3)

    col_p, col_q = st.columns([2, 1])
    with col_p:
        st.dataframe(ranking, use_container_width=True)

    with col_q:
        fig, ax = plt.subplots(figsize=(5, 5))
        scatter_colors = [dp_color(v) for v in daya_pembeda.values]
        ax.scatter(tingkat_kesukaran.values, daya_pembeda.values,
                   c=scatter_colors, s=80, alpha=0.85, edgecolors='none')
        ax.axhline(0.3, color=GOLD, linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axvline(0.3, color=ROSE, linestyle=':', linewidth=1.5, alpha=0.5)
        ax.axvline(0.7, color=GREEN, linestyle=':', linewidth=1.5, alpha=0.5)
        for i, soal in enumerate(daya_pembeda.index):
            ax.annotate(f"S{i+1}", (tingkat_kesukaran.iloc[i], daya_pembeda.iloc[i]),
                        fontsize=7, color='#c8c4bc', ha='center', va='bottom',
                        xytext=(0, 5), textcoords='offset points')
        ax.set_xlabel("Tingkat Kesukaran (P)", labelpad=8)
        ax.set_ylabel("Daya Pembeda (r)", labelpad=8)
        ax.set_title("Peta Kualitas Soal", fontweight='bold', fontsize=12, color='#e8e4dc', pad=10)
        ax.grid(True)
        fig.tight_layout()
        st.pyplot(fig)

    # ======================================================
    # FOOTER
    # ======================================================
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center; padding: 1.5rem; border-top: 1px solid #2a2a38; margin-top: 2rem;">
        <span style="font-family:'Playfair Display',serif; font-size:1.1rem; color:#f5c842;">✅ Analisis Selesai</span>
        <span style="color:#8a8580; font-size:0.85rem; display:block; margin-top:4px; letter-spacing:0.1em; text-transform:uppercase;">
            Dashboard Analisis Hasil Belajar Siswa · Powered by Streamlit
        </span>
    </div>
    """, unsafe_allow_html=True)

else:
    # Landing state
    st.markdown("""
    <div style="text-align:center; padding:4rem 2rem; background: linear-gradient(145deg, #1a1a24, #14141c);
         border-radius:20px; border: 1px dashed #2a2a38; margin-top:2rem;">
        <div style="font-size:3.5rem; margin-bottom:1rem;">📂</div>
        <div style="font-family:'Playfair Display',serif; font-size:1.8rem; color:#e8e4dc; margin-bottom:0.5rem;">
            Upload Data untuk Memulai
        </div>
        <div style="color:#8a8580; font-size:0.95rem; max-width:480px; margin:0 auto; line-height:1.6;">
            Siapkan file Excel dengan format: baris = siswa, kolom = soal (minimal 20 kolom numerik).
            Atau klik <b style="color:#f5c842">Gunakan Data Sampel</b> di sidebar kiri untuk mencoba.
        </div>
        <div style="margin-top:2rem; display:flex; justify-content:center; gap:1rem; flex-wrap:wrap;">
            <div style="background:#1e1e2e; border:1px solid #2a2a38; border-radius:12px; padding:0.75rem 1.5rem;
                 color:#c8c4bc; font-size:0.85rem;">📊 50 Siswa</div>
            <div style="background:#1e1e2e; border:1px solid #2a2a38; border-radius:12px; padding:0.75rem 1.5rem;
                 color:#c8c4bc; font-size:0.85rem;">📝 20 Soal</div>
            <div style="background:#1e1e2e; border:1px solid #2a2a38; border-radius:12px; padding:0.75rem 1.5rem;
                 color:#c8c4bc; font-size:0.85rem;">🤖 AI Clustering</div>
            <div style="background:#1e1e2e; border:1px solid #2a2a38; border-radius:12px; padding:0.75rem 1.5rem;
                 color:#c8c4bc; font-size:0.85rem;">📈 6 Analisis</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
