import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import base64
from pathlib import Path

from utils import prepare_price_dataframe, kebijakan_saran
from models_lstm import load_artifacts, forecast_lstm

ARTIFACT_WINDOW_SIZE = 30
FORECAST_DAYS_DEFAULT = 30

@st.cache_resource
def get_artifacts(pasar: str, komoditas: str):
    return load_artifacts(pasar, komoditas, ARTIFACT_WINDOW_SIZE)


# -------------------------
# Konfigurasi Halaman
# -------------------------
st.set_page_config(
    page_title="Dashboard Harga Barang & Prediksi",
    layout="wide"
)
#CSS KARTU
st.markdown(
    """
    <style>
    .komod-card {
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .komod-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 4px 10px rgba(0,0,0,0.18);
    }
    .komod-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 999px;
        font-size: 10px;
        font-weight: 600;
        color: white;
        margin-left: 6px;
    }
    /* === BADGE TREND/VOLATIL === */
    .badge{
      display:inline-block;
      padding:2px 10px;
      border-radius:999px;
      font-size:12px;
      font-weight:700;
      color:white;
      margin-right:6px;
    }
    .badge-up{ background:#E53935; }
    .badge-down{ background:#1E88E5; }
    .badge-flat{ background:#43A047; }
    .badge-vol{ background:#6D4C41; }
    .badge-soft{ opacity:0.92; }
    </style>
    """,
    unsafe_allow_html=True
)
# CSS BACKGROUND UNGU
st.markdown(
    """
    <style>
    .stApp {
        background-color: #F3E5F5 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# FUNGSI + HEADER DENGAN FOTO
def get_base64_of_image(image_path: str) -> str:
    """
    Mengubah file gambar lokal menjadi string base64
    agar bisa dipakai di CSS background-image.
    """
    img_path = Path(image_path)
    if not img_path.exists():
        return ""
    with open(img_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode("utf-8")
# === Header dengan background foto ===
img_b64 = get_base64_of_image("assets/background_header.jpeg")

if img_b64:
    st.markdown(
        f"""
        <style>
        .header-banner {{
            width: 100%;
            height: 280px;
            background-image: url("data:image/jpeg;base64,{img_b64}");
            background-size: cover;
            background-position: center -300px;   /* ⇦ GESER FOTO KE ATAS */
            background-repeat: no-repeat;
            border-radius: 12px;
            margin-bottom: 20px;
        }}
        </style>

        <div class="header-banner"></div>
        """,
        unsafe_allow_html=True
    )

# Hilangkan menu & footer Streamlit
hide_st_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

# Header utama
st.markdown(
    """
    <div style='text-align:center; margin-bottom: 15px;'>
        <h1 style='margin-bottom: 0;'>Dashboard Harga Barang & Prediksi</h1>
        <p style='font-size:14px; margin-top:4px; color:#555;'>
            Dinas Perindustrian & Perdagangan Kabupaten Tangerang – Analisis Harga Pasar
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# -------------------------
# Load Data (tanpa upload, langsung dari file lokal)
# -------------------------

@st.cache_data
def load_data():
    df_raw = pd.read_csv("harga_pasar_2024_2025.csv")
    return prepare_price_dataframe(df_raw)

try:
    df = load_data()
except Exception as e:
    st.error(f"Gagal membaca dataset 'harga_pasar_2024_2025.csv': {e}")
    st.stop()

def get_komoditas_style(nama: str):
    """
    Mengembalikan (kategori, bg_color, badge_color) berdasarkan nama komoditas.
    """
    n = str(nama).lower()

    # Beras
    if "beras" in n:
        return "BERAS", "#FFF8E1", "#F9A825"

    # Minyak goreng
    if "minyak" in n:
        return "MINYAK", "#FFF3E0", "#FB8C00"

    # Cabe / cabai / rawit
    if "cabe" in n or "cabai" in n or "rawit" in n:
        return "CABAI", "#FFEBEE", "#E53935"

    # Bawang
    if "bawang" in n:
        return "BAWANG", "#EDE7F6", "#8E24AA"

    # Tepung / terigu
    if "tepung" in n or "segitiga biru" in n:
        return "TEPUNG", "#E8F5E9", "#43A047"

    # Gula
    if "gula" in n:
        return "GULA", "#F3E5F5", "#7B1FA2"

    # Protein hewani
    if "ayam" in n or "daging" in n or "telur" in n:
        return "PROTEIN", "#E3F2FD", "#1E88E5"

    # Default
    return "LAINNYA", "#F5F5F5", "#757575"


# -------------------------
st.markdown("### 📊 Harga Komoditas Pasar + Prediksi (Model Tersimpan)")

pasar_list = sorted(df["pasar"].unique().tolist())
pasar = st.selectbox("Pilih Pasar", pasar_list, key="pilih_pasar")

df_pasar = df[df["pasar"] == pasar].copy()
if df_pasar.empty:
    st.warning(f"Tidak ada data untuk pasar **{pasar}**.")
    st.stop()

df_pasar["tanggal"] = pd.to_datetime(df_pasar["tanggal"])
min_date = df_pasar["tanggal"].min().date()
max_date = df_pasar["tanggal"].max().date()

# =========================
# CARD HARGA (ATAS) - WARNA TETAP
# =========================
st.markdown("#### 📅 Pilih Tanggal")

selected_date = st.date_input(
    "Tanggal harga yang ingin dilihat",
    value=max_date,
    min_value=min_date,
    max_value=max_date,
    key="tgl_pasar"
)

df_hari_ini = df_pasar[df_pasar["tanggal"].dt.date == selected_date].copy()

if df_hari_ini.empty:
    st.warning(f"Tidak ada data pada tanggal **{selected_date}**.")
else:
    df_hari_ini = df_hari_ini.sort_values("komoditas")
    st.markdown(f"#### 💰 Daftar Harga Komoditas – Pasar **{pasar}** ({selected_date})")

    num_cols = 3
    cols = st.columns(num_cols)

    for i, row in df_hari_ini.iterrows():
        c = cols[i % num_cols]
        nama = str(row["komoditas"])
        harga = row["harga"]

        # pakai style warna kamu
        kategori, bg_color, badge_color = get_komoditas_style(nama)

        with c:
            st.markdown(
                f"""
                <div class="komod-card" style="
                    background-color: {bg_color};
                    padding: 14px 16px;
                    border-radius: 14px;
                    margin-bottom: 12px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.10);
                    border: 1px solid rgba(0,0,0,0.08);
                ">
                    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:6px;">
                        <div style="font-weight:700; font-size:14px;">
                            {nama.upper()}
                        </div>
                        <span class="komod-badge" style="background-color:{badge_color};">
                            {kategori}
                        </span>
                    </div>
                    <div style="font-size:12px; color:#555;">Harga</div>
                    <div style="font-size:20px; font-weight:800; color:#1A237E;">
                        Rp {harga:,.0f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

st.markdown("---")

# DETAIL + PREDIKSI (BAWAH CARD, FULL WIDTH)
# =========================
st.markdown("### 🔍 Detail Per Komoditas + Prediksi")

komoditas_list = sorted(df_pasar["komoditas"].unique().tolist())
komoditas = st.selectbox(
    "Pilih komoditas",
    ["— Pilih komoditas —"] + komoditas_list,
    index=0,
    key="komoditas_detail"
)

if komoditas == "— Pilih komoditas —":
    st.info("Pilih komoditas untuk melihat riwayat dan prediksi harganya.")
    st.stop()

forecast_days = st.slider(
    "Jumlah hari prediksi",
    min_value=7,
    max_value=60,
    value=FORECAST_DAYS_DEFAULT,
    step=1
)

df_sub = df_pasar[df_pasar["komoditas"] == komoditas].copy().sort_values("tanggal")
if df_sub.empty:
    st.warning("Data historis kosong.")
    st.stop()

st.caption(f"Periode: {df_sub['tanggal'].min().date()} s.d. {df_sub['tanggal'].max().date()}")

loaded = get_artifacts(pasar, komoditas)
if loaded is None:
    st.warning(
        f"Model untuk **{komoditas} – {pasar}** belum ada di folder `artifacts/` "
        f"(WS={ARTIFACT_WINDOW_SIZE})."
    )
    st.stop()

model = loaded["model"]
scaler = loaded["scaler"]

mae = loaded.get("mae")
rmse = loaded.get("rmse")
if mae is not None and rmse is not None:
    st.caption(f"📌 Evaluasi model: MAE={mae:.0f} | RMSE={rmse:.0f}")

df_pred = forecast_lstm(
    model=model,
    scaler=scaler,
    df_sub=df_sub,
    n_days=forecast_days,
    window_size=ARTIFACT_WINDOW_SIZE
)

if df_pred is None or df_pred.empty:
    st.warning("Prediksi tidak tersedia (cek artifacts / window size / data historis).")
    st.stop()

# ✅ lanjut grafik / tabel di bawah ini
# ======= KPI RINGKAS =======
h = min(7, len(df_pred))
last_actual = float(df_sub["harga"].iloc[-1])
mean_pred_7 = float(df_pred["prediksi"].head(h).mean())
last_pred_7 = float(df_pred["prediksi"].iloc[h-1])

change_pct_mean = ((mean_pred_7 - last_actual) / last_actual * 100) if last_actual > 0 else 0.0
change_pct_last = ((last_pred_7 - last_actual) / last_actual * 100) if last_actual > 0 else 0.0

# skor tren: ambil yang lebih "tegas"
trend_score = change_pct_last if abs(change_pct_last) > abs(change_pct_mean) else change_pct_mean

# volatilitas
if len(df_pred) > 2:
    pct_changes = df_pred["prediksi"].pct_change().dropna() * 100
    volatility = float(pct_changes.std()) if not pct_changes.empty else 0.0
else:
    volatility = 0.0

# BADGE TREND
if trend_score > 10:
    tren_text, tren_class = "TREND: naik tajam", "badge-up"
elif trend_score > 3:
    tren_text, tren_class = "TREND: naik ringan", "badge-up"
elif trend_score < -10:
    tren_text, tren_class = "TREND: turun tajam", "badge-down"
elif trend_score < -3:
    tren_text, tren_class = "TREND: turun ringan", "badge-down"
else:
    tren_text, tren_class = "TREND: stabil", "badge-flat"

if volatility > 8:
    vol_text = "VOL: tinggi"
elif volatility > 4:
    vol_text = "VOL: sedang"
else:
    vol_text = "VOL: rendah"

st.markdown(
    f'<span class="badge {tren_class}">{tren_text}</span>'
    f'<span class="badge badge-vol">{vol_text}</span>',
    unsafe_allow_html=True
)

# ✅ baru KPI metric di bawah badge
c1, c2, c3, c4 = st.columns(4)
c1.metric("Harga terakhir", f"Rp {last_actual:,.0f}")
c2.metric(f"Rata-rata prediksi {h} hari", f"Rp {mean_pred_7:,.0f}", f"{change_pct_mean:+.1f}%")
c3.metric(f"Prediksi hari ke-{h}", f"Rp {last_pred_7:,.0f}", f"{change_pct_last:+.1f}%")
c4.metric("Volatilitas prediksi", f"{volatility:.1f}%", "")


           # ============ GRAFIK (RIWAYAT + PREDIKSI) ============
st.markdown("#### 📉 Riwayat + Prediksi Harga (Overlay)")

df_sub_plot = df_sub.copy()
df_sub_plot["tanggal"] = pd.to_datetime(df_sub_plot["tanggal"])

df_pred_plot = df_pred.copy()
df_pred_plot["tanggal"] = pd.to_datetime(df_pred_plot["tanggal"])

last_actual_date = df_sub_plot["tanggal"].max()

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df_sub_plot["tanggal"],
    y=df_sub_plot["harga"],
    mode="lines+markers",
    name="Aktual",
    hovertemplate="<b>%{x|%d-%m-%Y}</b><br>Aktual: <b>Rp %{y:,.0f}</b><extra></extra>",
))

fig.add_trace(go.Scatter(
    x=df_pred_plot["tanggal"],
    y=df_pred_plot["prediksi"],
    mode="lines+markers",
    name="Prediksi",
    line=dict(dash="dash"),
    hovertemplate="<b>%{x|%d-%m-%Y}</b><br>Prediksi: <b>Rp %{y:,.0f}</b><extra></extra>",
))

fig.add_shape(
    type="line",
    x0=last_actual_date,
    x1=last_actual_date,
    y0=0,
    y1=1,
    xref="x",
    yref="paper",
    line=dict(color="gray", width=2, dash="dot"),
)

fig.update_layout(
    title={"text": f"{komoditas} – Pasar {pasar} (Prediksi {forecast_days} hari)", "x": 0.5},
    xaxis_title="Tanggal",
    yaxis_title="Harga (Rp)",
    template="plotly_white",
    hovermode="x unified",
    margin=dict(l=30, r=10, t=60, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
)

st.plotly_chart(fig, use_container_width=True)

# ============ PREDIKSI (RINGKAS + EXPANDER) ============
st.markdown("#### 📋 Prediksi (ringkas)")

df_pred_tampil = df_pred.copy()
df_pred_tampil["tanggal"] = pd.to_datetime(df_pred_tampil["tanggal"]).dt.strftime("%d-%m-%Y")
df_pred_tampil["prediksi"] = df_pred_tampil["prediksi"].round(0).astype(int)
df_pred_tampil = df_pred_tampil.rename(columns={"tanggal": "Tanggal", "prediksi": "Prediksi (Rp)"})

st.dataframe(df_pred_tampil.head(7), use_container_width=True, hide_index=True)

with st.expander("Lihat semua prediksi"):
    st.dataframe(df_pred_tampil, use_container_width=True, hide_index=True)

# ============ SARAN KEBIJAKAN ============
st.markdown("#### 📑 Saran Kebijakan")
st.markdown(kebijakan_saran(df_sub, df_pred, horizon_analisis=7))

