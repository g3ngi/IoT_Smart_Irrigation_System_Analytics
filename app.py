"""
Smart Irrigation System - Big Data Analytics Dashboard
Streamlit + Plotly + scikit-learn + NetworkX/PyVis

Jalankan:  streamlit run app.py
"""
from __future__ import annotations

import io
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

try:
    from pyvis.network import Network

    HAS_PYVIS = True
except ImportError:  # pyvis opsional
    HAS_PYVIS = False

BASE_DIR = Path(__file__).parent
DEFAULT_SENSOR_CSV = BASE_DIR / "system_data.csv"
DEFAULT_EDGES_CSV = BASE_DIR / "irrigation_graph_edges.csv"
GITHUB_RAW_URL = (
    "https://raw.githubusercontent.com/Soutehkeshan/Smart-Watering-System/"
    "main/data/system_data.csv"
)

# Nama kolom asli dataset GitHub -> nama kolom internal
RENAME_MAP = {
    "Soil Moisture (%)": "soil_moisture",
    "Temperature (°C)": "air_temp",
    "Humidity (%)": "air_humidity",
    "Light Level (lx)": "light_lx",
    "Valve Duration (s)": "valve_s",
}
FEATURES = ["soil_moisture", "air_temp", "air_humidity", "light_lx"]
LABELS = {
    "soil_moisture": "Soil Moisture (%)",
    "air_temp": "Air Temp (°C)",
    "air_humidity": "Air Humidity (%)",
    "light_lx": "Light Level (lx)",
    "valve_s": "Valve Duration (s)",
    "water_liters": "Water (L)",
}
ZONES = ["Zone_A", "Zone_B", "Zone_C", "Zone_D"]
ZONE_TO_SENSOR = {
    "Zone_A": "Sensor_Zone1",
    "Zone_B": "Sensor_Zone2",
    "Zone_C": "Sensor_Zone3",
    "Zone_D": "Sensor_Zone4",
}

st.set_page_config(page_title="Smart Irrigation Analytics", page_icon="💧", layout="wide")


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_csv(source: str | bytes, is_bytes: bool = False) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(source) if is_bytes else source)


def prepare_sensor_df(raw: pd.DataFrame, flow_lps: float, start_ts: str) -> pd.DataFrame:
    """Normalisasi kolom + lengkapi timestamp/zona/device bila dataset tidak punya."""
    df = raw.rename(columns=RENAME_MAP).copy()

    missing = [c for c in FEATURES + ["valve_s"] if c not in df.columns]
    if missing:
        st.error(f"Kolom wajib tidak ditemukan: {missing}. Kolom tersedia: {list(raw.columns)}")
        st.stop()

    # timestamp: pakai kolom asli bila ada, kalau tidak -> simulasi per jam
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    else:
        df["timestamp"] = pd.date_range(start_ts, periods=len(df), freq="h")

    # zona / device: pakai kolom asli bila ada, kalau tidak -> round-robin 4 zona
    if "zone_id" not in df.columns:
        df["zone_id"] = [ZONES[i % len(ZONES)] for i in range(len(df))]
    if "device_id" not in df.columns:
        df["device_id"] = df["zone_id"].map(lambda z: f"ESP32_{ZONES.index(z) + 1:02d}" if z in ZONES else z)

    # target: durasi valve (detik) x debit (L/detik) = volume air (liter)
    if "target_water_liters" in df.columns:
        df["water_liters"] = df["target_water_liters"]
    else:
        df["water_liters"] = df["valve_s"] * flow_lps
    return df.sort_values("timestamp").reset_index(drop=True)


@st.cache_resource(show_spinner="Melatih Random Forest...")
def train_model(X_train, y_train, n_estimators: int, max_depth: int | None, seed: int):
    model = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth, random_state=seed, n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def default_edges() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source": ["Pump_Main", "Pump_Main", "Junction_A", "Junction_A", "Junction_B", "Junction_B"],
            "target": ["Junction_A", "Junction_B", "Sensor_Zone1", "Sensor_Zone2", "Sensor_Zone3", "Sensor_Zone4"],
            "capacity_lpm": [500, 500, 200, 200, 200, 200],
            "length_m": [50, 80, 30, 45, 60, 40],
            "leak_risk": [0.01, 0.05, 0.02, 0.03, 0.08, 0.01],
        }
    )


def node_type(name: str) -> str:
    n = name.lower()
    if n.startswith("pump"):
        return "Pump Station"
    if n.startswith("junction"):
        return "Junction"
    if n.startswith("sensor"):
        return "Sensor Node"
    return "Other"


NODE_STYLE = {
    "Pump Station": {"color": "#1f77b4", "size": 34, "symbol": "square"},
    "Junction": {"color": "#ff7f0e", "size": 26, "symbol": "diamond"},
    "Sensor Node": {"color": "#2ca02c", "size": 22, "symbol": "circle"},
    "Other": {"color": "#7f7f7f", "size": 20, "symbol": "circle"},
}


# ----------------------------------------------------------------------------
# Sidebar: sumber data & parameter
# ----------------------------------------------------------------------------
st.sidebar.title("⚙️ Pengaturan")
st.sidebar.subheader("1. Sumber data sensor")
src = st.sidebar.radio(
    "Cara memuat CSV",
    ["File lokal (path)", "Upload CSV", "URL GitHub (raw)"],
    help="Dataset: Soutehkeshan/Smart-Watering-System → data/system_data.csv",
)

raw_df: pd.DataFrame | None = None
try:
    if src == "File lokal (path)":
        path = st.sidebar.text_input("Path CSV", value=str(DEFAULT_SENSOR_CSV))
        raw_df = load_csv(path)
    elif src == "Upload CSV":
        up = st.sidebar.file_uploader("Upload system_data.csv", type="csv")
        if up is not None:
            raw_df = load_csv(up.getvalue(), is_bytes=True)
    else:
        url = st.sidebar.text_input("URL raw CSV", value=GITHUB_RAW_URL)
        raw_df = load_csv(url)
except Exception as exc:  # noqa: BLE001
    st.sidebar.error(f"Gagal memuat data: {exc}")

if raw_df is None:
    st.info("Pilih / upload dataset sensor di sidebar untuk memulai.")
    st.stop()

st.sidebar.subheader("2. Asumsi data")
flow_lps = st.sidebar.number_input(
    "Debit valve (liter/detik)", min_value=0.1, max_value=50.0, value=2.0, step=0.1,
    help="Dataset hanya punya durasi valve (detik). Volume air = durasi × debit.",
)
start_ts = st.sidebar.text_input(
    "Timestamp awal (simulasi, interval 1 jam)", value="2026-11-01 00:00:00",
    help="Dipakai hanya jika CSV tidak punya kolom 'timestamp'.",
)

st.sidebar.subheader("3. Model Random Forest")
test_size = st.sidebar.slider("Porsi data uji", 0.1, 0.4, 0.2, 0.05)
n_estimators = st.sidebar.slider("n_estimators", 50, 500, 200, 50)
max_depth_opt = st.sidebar.slider("max_depth (0 = tanpa batas)", 0, 30, 0)
seed = st.sidebar.number_input("Random seed", value=42, step=1)
target_choice = st.sidebar.radio(
    "Target prediksi", ["valve_s", "water_liters"],
    format_func=lambda c: LABELS[c], horizontal=False,
)

try:
    df = prepare_sensor_df(raw_df, flow_lps, start_ts)
except Exception as exc:  # noqa: BLE001
    st.error(f"Gagal memproses data: {exc}")
    st.stop()

st.title("💧 Smart Irrigation System — Big Data Analytics Dashboard")
st.caption(
    f"{len(df):,} baris telemetri • {df['zone_id'].nunique()} zona • "
    f"{df['timestamp'].min():%Y-%m-%d} s/d {df['timestamp'].max():%Y-%m-%d}"
)
if "timestamp" not in raw_df.columns:
    st.warning(
        "Dataset sumber **tidak memiliki** kolom timestamp, zona, dan device. "
        "Ketiganya **disimulasikan** (jam berurutan, zona round-robin A–D) agar tab time-series "
        "dan graph bisa berjalan. Untuk laporan akhir, jelaskan ini sebagai asumsi."
    )

graph_info: dict = {}  # diisi tab 3, dibaca tab 4

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Executive KPI & Telemetry",
        "🤖 ML Predictive Analytics",
        "🕸️ Graph Analytics & Network",
        "💡 Key Insights",
    ]
)

# ----------------------------------------------------------------------------
# TAB 1: KPI & telemetry
# ----------------------------------------------------------------------------
with tab1:
    zones_sel = st.multiselect("Filter zona", sorted(df["zone_id"].unique()), default=sorted(df["zone_id"].unique()))
    d = df[df["zone_id"].isin(zones_sel)]
    if d.empty:
        st.warning("Pilih minimal satu zona.")
    else:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total konsumsi air", f"{d['water_liters'].sum():,.0f} L")
        k2.metric("Rata-rata kelembapan tanah", f"{d['soil_moisture'].mean():.1f} %")
        k3.metric("Rata-rata suhu udara", f"{d['air_temp'].mean():.1f} °C")
        dry_share = (d["soil_moisture"] < 40).mean() * 100
        k4.metric("Pembacaan tanah kering (<40%)", f"{dry_share:.1f} %")

        metric_col = st.selectbox(
            "Metrik time-series", ["soil_moisture", "air_temp", "air_humidity", "light_lx", "water_liters"],
            format_func=lambda c: LABELS[c],
        )
        roll = st.slider("Rolling mean (jam)", 1, 48, 6)
        ts = d.copy()
        ts["smooth"] = ts.groupby("zone_id")[metric_col].transform(lambda s: s.rolling(roll, min_periods=1).mean())
        fig = px.line(ts, x="timestamp", y="smooth", color="zone_id", labels={"smooth": LABELS[metric_col], "timestamp": "Waktu"})
        fig.update_layout(height=380, legend_title_text="Zona", margin=dict(t=30))
        st.plotly_chart(fig, width="stretch")

        c1, c2 = st.columns(2)
        with c1:
            zsum = d.groupby("zone_id", as_index=False).agg(water=("water_liters", "sum"), moisture=("soil_moisture", "mean"))
            figb = px.bar(zsum, x="zone_id", y="water", color="moisture", color_continuous_scale="RdYlBu",
                          labels={"water": "Total air (L)", "zone_id": "Zona", "moisture": "Rata2 moisture (%)"},
                          title="Konsumsi air per zona")
            st.plotly_chart(figb, width="stretch")
        with c2:
            corr = d[FEATURES + ["valve_s"]].rename(columns=LABELS).corr()
            figc = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Matriks korelasi")
            st.plotly_chart(figc, width="stretch")

        with st.expander("Lihat data mentah"):
            st.dataframe(d, width="stretch", height=300)

# ----------------------------------------------------------------------------
# TAB 2: ML
# ----------------------------------------------------------------------------
with tab2:
    X, y = df[FEATURES], df[target_choice]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=int(seed))
    model = train_model(X_tr, y_tr, n_estimators, max_depth_opt or None, int(seed))
    pred = model.predict(X_te)

    rmse = float(np.sqrt(mean_squared_error(y_te, pred)))
    mae = float(mean_absolute_error(y_te, pred))
    r2 = float(r2_score(y_te, pred))

    unit = "detik" if target_choice == "valve_s" else "liter"
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("RMSE", f"{rmse:.3f} {unit}")
    m2.metric("MAE", f"{mae:.3f} {unit}")
    m3.metric("R² Score", f"{r2:.3f}")
    m4.metric("Data latih / uji", f"{len(X_tr)} / {len(X_te)}")

    lim = [float(min(y_te.min(), pred.min())), float(max(y_te.max(), pred.max()))]
    figs = go.Figure()
    figs.add_trace(go.Scatter(x=y_te, y=pred, mode="markers", name="Prediksi",
                              marker=dict(size=7, opacity=0.65, color=X_te["soil_moisture"], colorscale="RdYlBu",
                                          showscale=True, colorbar=dict(title="Soil Moisture (%)"))))
    figs.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="Ideal (y = x)", line=dict(dash="dash", color="grey")))
    figs.update_layout(title=f"Actual vs Predicted — {LABELS[target_choice]}", height=450,
                       xaxis_title=f"Actual ({unit})", yaxis_title=f"Predicted ({unit})")
    st.plotly_chart(figs, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        imp = pd.DataFrame({"feature": [LABELS[f] for f in FEATURES], "importance": model.feature_importances_}).sort_values("importance")
        st.plotly_chart(px.bar(imp, x="importance", y="feature", orientation="h", title="Feature importance"), width="stretch")
    with c2:
        resid = y_te.values - pred
        st.plotly_chart(px.histogram(x=resid, nbins=30, labels={"x": f"Residual ({unit})"}, title="Distribusi residual"), width="stretch")

    st.subheader("Simulasi kebutuhan air")
    s1, s2, s3, s4 = st.columns(4)
    sm = s1.slider("Soil Moisture (%)", 0.0, 100.0, float(df["soil_moisture"].median()))
    tp = s2.slider("Suhu (°C)", 0.0, 45.0, float(df["air_temp"].median()))
    hm = s3.slider("Humidity (%)", 0.0, 100.0, float(df["air_humidity"].median()))
    lx = s4.slider("Light (lx)", 0.0, 2000.0, float(df["light_lx"].median()))
    sim = model.predict(pd.DataFrame([[sm, tp, hm, lx]], columns=FEATURES))[0]
    st.success(f"Prediksi {LABELS[target_choice]}: **{sim:.2f} {unit}**")

    st.caption(
        "Catatan: pada dataset ini soil moisture berkorelasi sangat kuat negatif dengan durasi valve "
        "(label dihasilkan oleh aturan kontrol), sehingga R² tinggi bukan berarti model menangkap "
        "fenomena fisik yang kompleks. Bahas hal ini di bagian limitasi laporan."
    )

# ----------------------------------------------------------------------------
# TAB 3: Graph
# ----------------------------------------------------------------------------
with tab3:
    st.markdown("**Topologi jaringan pipa irigasi** — sumber edge list:")
    e_src = st.radio("Edge list", ["Contoh bawaan (irrigation_graph_edges.csv)", "Upload CSV"], horizontal=True)
    if e_src == "Upload CSV":
        eup = st.file_uploader("CSV dengan kolom source,target,capacity_lpm,length_m,leak_risk", type="csv", key="edges")
        edges = load_csv(eup.getvalue(), is_bytes=True) if eup else None
    else:
        edges = load_csv(str(DEFAULT_EDGES_CSV)) if DEFAULT_EDGES_CSV.exists() else default_edges()

    if edges is None:
        st.info("Upload edge list untuk melanjutkan.")
    elif not {"source", "target"}.issubset(edges.columns):
        st.error("Edge list harus punya kolom 'source' dan 'target'.")
    else:
        for col, dflt in [("capacity_lpm", 0.0), ("length_m", 1.0), ("leak_risk", 0.0)]:
            if col not in edges.columns:
                edges[col] = dflt
        edges = edges.copy()
        edges["inv_capacity"] = 1000.0 / edges["capacity_lpm"].replace(0, np.nan)
        edges["inv_capacity"] = edges["inv_capacity"].fillna(1000.0)

        G = nx.DiGraph()
        for r in edges.itertuples(index=False):
            G.add_edge(r.source, r.target, capacity_lpm=r.capacity_lpm, length_m=r.length_m,
                       leak_risk=r.leak_risk, inv_capacity=r.inv_capacity)

        # --- zona paling kering dari data sensor ---
        zone_moist = df.groupby("zone_id")["soil_moisture"].mean().sort_values()
        zone_tbl = zone_moist.rename("avg_soil_moisture").reset_index()
        zone_tbl["sensor_node"] = zone_tbl["zone_id"].map(ZONE_TO_SENSOR)
        zone_tbl = zone_tbl.dropna(subset=["sensor_node"])
        zone_tbl = zone_tbl[zone_tbl["sensor_node"].isin(G.nodes)]

        pumps = [n for n in G.nodes if node_type(n) == "Pump Station"]
        sensors = [n for n in G.nodes if node_type(n) == "Sensor Node"]

        cA, cB, cC = st.columns(3)
        origin = cA.selectbox("Titik asal (pump)", pumps or list(G.nodes))
        weight_key = cB.selectbox(
            "Bobot jalur terpendek",
            ["length_m", "leak_risk", "inv_capacity"],
            format_func={"length_m": "Panjang pipa (m)", "leak_risk": "Risiko kebocoran", "inv_capacity": "Kebalikan kapasitas (1000/lpm)"}.get,
        )
        if not zone_tbl.empty:
            default_target = zone_tbl.iloc[0]["sensor_node"]
            cC.metric("Zona paling kering", f"{zone_tbl.iloc[0]['zone_id']}", f"{zone_tbl.iloc[0]['avg_soil_moisture']:.1f}% moisture", delta_color="off")
        else:
            default_target = sensors[0] if sensors else list(G.nodes)[-1]
        target = st.selectbox("Node tujuan", list(G.nodes), index=list(G.nodes).index(default_target),
                              help="Default: sensor node dari zona dengan kelembapan tanah rata-rata terendah.")

        path, path_edges, stats = [], set(), None
        try:
            path = nx.shortest_path(G, origin, target, weight=weight_key)
            path_edges = set(zip(path[:-1], path[1:]))
            seg = [G[u][v] for u, v in zip(path[:-1], path[1:])]
            surv = float(np.prod([1 - s["leak_risk"] for s in seg])) if seg else 1.0
            stats = {
                "Total panjang (m)": sum(s["length_m"] for s in seg),
                "Kapasitas bottleneck (lpm)": min(s["capacity_lpm"] for s in seg) if seg else 0,
                "Probabilitas air sampai (tanpa bocor)": f"{surv * 100:.1f}%",
                "Jumlah hop": len(seg),
            }
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            st.error(f"Tidak ada jalur dari {origin} ke {target}.")

        if path:
            worst = edges.sort_values("leak_risk", ascending=False).iloc[0]
            graph_info = {
                "path": path, "stats": stats, "weight": weight_key,
                "worst_edge": f"{worst['source']} → {worst['target']}", "worst_leak": float(worst["leak_risk"]),
                "dry_zone": zone_tbl.iloc[0]["zone_id"] if not zone_tbl.empty else None,
            }
            st.success("Jalur: " + " → ".join(path))
            sc = st.columns(len(stats))
            for col, (k, v) in zip(sc, stats.items()):
                col.metric(k, f"{v:,.0f}" if isinstance(v, (int, float)) else v)

        # --- Plotly graph ---
        pos = nx.spring_layout(G.to_undirected(), seed=7, k=1.2)
        figg = go.Figure()
        for u, v, a in G.edges(data=True):
            on_path = (u, v) in path_edges
            figg.add_trace(go.Scatter(
                x=[pos[u][0], pos[v][0], None], y=[pos[u][1], pos[v][1], None], mode="lines",
                line=dict(width=5 if on_path else 1.5, color="#d62728" if on_path else "#aaaaaa"),
                hoverinfo="text", text=f"{u} → {v}<br>{a['length_m']} m • {a['capacity_lpm']} lpm • leak {a['leak_risk']}",
                showlegend=False,
            ))
        for t, style in NODE_STYLE.items():
            nodes = [n for n in G.nodes if node_type(n) == t]
            if not nodes:
                continue
            figg.add_trace(go.Scatter(
                x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes], mode="markers+text", name=t,
                text=nodes, textposition="top center",
                marker=dict(size=style["size"], color=style["color"], symbol=style["symbol"], line=dict(width=1, color="white")),
            ))
        figg.update_layout(height=480, xaxis=dict(visible=False), yaxis=dict(visible=False), margin=dict(t=20, b=10),
                           legend=dict(orientation="h", y=-0.05))
        st.plotly_chart(figg, width="stretch")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Kelembapan tanah per zona (terkering di atas)**")
            st.dataframe(zone_tbl.round(2), width="stretch", hide_index=True)
        with c2:
            st.markdown("**Centrality node**")
            cen = pd.DataFrame({
                "node": list(G.nodes),
                "tipe": [node_type(n) for n in G.nodes],
                "degree_centrality": [nx.degree_centrality(G)[n] for n in G.nodes],
                "betweenness": [nx.betweenness_centrality(G)[n] for n in G.nodes],
            }).sort_values("betweenness", ascending=False)
            st.dataframe(cen.round(3), width="stretch", hide_index=True)

        # --- PyVis interaktif ---
        if HAS_PYVIS:
            with st.expander("Tampilan interaktif (PyVis)"):
                net = Network(height="460px", width="100%", directed=True, bgcolor="#ffffff", font_color="#222")
                for n in G.nodes:
                    stl = NODE_STYLE[node_type(n)]
                    net.add_node(n, label=n, color=stl["color"], size=stl["size"], title=node_type(n))
                for u, v, a in G.edges(data=True):
                    on_path = (u, v) in path_edges
                    net.add_edge(u, v, color="#d62728" if on_path else "#bbbbbb", width=5 if on_path else 1.5,
                                 title=f"{a['length_m']} m • {a['capacity_lpm']} lpm • leak {a['leak_risk']}")
                components.html(net.generate_html(notebook=False), height=480, scrolling=False)
        else:
            st.caption("Install `pyvis` untuk tampilan jaringan interaktif tambahan.")

# ----------------------------------------------------------------------------
# TAB 4: Key insights (dihitung otomatis dari data, model, dan graf)
# ----------------------------------------------------------------------------
with tab4:
    st.subheader("Key Insights & Rekomendasi")
    st.caption("Semua angka dihitung ulang dari dataset yang sedang dimuat — berubah bila Anda ganti data atau parameter.")

    total_valve = float(df["valve_s"].sum())
    wet = df[df["valve_s"] > 0]
    dry = df[df["soil_moisture"] < 40]
    idle_share = float((df["valve_s"] == 0).mean() * 100)
    cutoff = float(wet["soil_moisture"].quantile(0.99)) if len(wet) else float("nan")
    dry_water_share = float(dry["valve_s"].sum() / total_valve * 100) if total_valve else 0.0

    # kontribusi fitur (permutation importance, data uji) + baseline linear 1 fitur
    pi = permutation_importance(model, X_te, y_te, n_repeats=5, random_state=int(seed), n_jobs=-1)
    pi_pos = np.clip(pi.importances_mean, 0, None)
    pi_share = pi_pos / pi_pos.sum() * 100 if pi_pos.sum() > 0 else pi_pos
    moist_share = float(pi_share[FEATURES.index("soil_moisture")])
    # baseline aturan: valve = 0 di atas batas moisture, garis linear di bawahnya
    below_tr = X_tr["soil_moisture"] < cutoff
    if below_tr.sum() >= 5:
        lin = LinearRegression().fit(X_tr.loc[below_tr, ["soil_moisture"]], y_tr[below_tr])
        rule_pred = np.where(X_te["soil_moisture"] < cutoff, lin.predict(X_te[["soil_moisture"]]), 0.0)
        r2_lin = float(r2_score(y_te, np.clip(rule_pred, 0, None)))
    else:
        r2_lin = float("nan")
    gap = r2 - r2_lin

    i1, i2, i3, i4 = st.columns(4)
    i1.metric("Valve tidak menyala", f"{idle_share:.0f}% pembacaan")
    i2.metric("Batas moisture penyiraman", f"≈ {cutoff:.0f}%", help="Persentil ke-99 soil moisture di antara baris yang valvenya menyala.")
    i3.metric("Air dipakai saat tanah <40%", f"{dry_water_share:.0f}%")
    i4.metric("Peran Soil Moisture di model", f"{moist_share:.0f}%", help="Porsi permutation importance pada data uji.")

    # --- chart: respons valve terhadap kelembapan tanah ---
    lo = int(np.floor(df["soil_moisture"].min() / 10) * 10)
    hi = int(np.ceil(df["soil_moisture"].max() / 10) * 10)
    edges_sm = np.arange(lo, hi + 10, 10)
    lbl = [f"{a}–{b}%" for a, b in zip(edges_sm[:-1], edges_sm[1:])]
    grp = df.assign(bin=pd.cut(df["soil_moisture"], edges_sm, labels=lbl, include_lowest=True)).groupby("bin", observed=True)
    resp = grp.agg(n=("valve_s", "size"), valve_mean=("valve_s", "mean"), idle=("valve_s", lambda s: (s == 0).mean() * 100)).reset_index()

    figr = go.Figure()
    figr.add_trace(go.Bar(x=resp["bin"], y=resp["valve_mean"], name="Rata-rata durasi valve (s)", marker_color="#1f77b4"))
    figr.add_trace(go.Scatter(x=resp["bin"], y=resp["idle"], name="% valve mati", yaxis="y2", mode="lines+markers", line=dict(color="#d62728")))
    figr.update_layout(
        title="Respons valve terhadap kelembapan tanah", height=380,
        xaxis_title="Soil moisture", yaxis=dict(title="Durasi valve (s)"),
        yaxis2=dict(title="% valve mati", overlaying="y", side="right", range=[0, 105]),
        legend=dict(orientation="h", y=-0.25), margin=dict(t=50),
    )

    # --- chart: kontribusi fitur ---
    figi = px.bar(
        pd.DataFrame({"fitur": [LABELS[f] for f in FEATURES], "kontribusi (%)": pi_share}).sort_values("kontribusi (%)"),
        x="kontribusi (%)", y="fitur", orientation="h", title="Kontribusi fitur (permutation importance)",
    )
    figi.update_layout(height=380, margin=dict(t=50))

    cc1, cc2 = st.columns(2)
    cc1.plotly_chart(figr, width="stretch")
    cc2.plotly_chart(figi, width="stretch")

    # --- efek suhu & cahaya pada kondisi tanah kering ---
    env_lines = []
    if len(dry) >= 30:
        for col, name in [("air_temp", "suhu udara"), ("light_lx", "intensitas cahaya")]:
            try:
                q = pd.qcut(dry[col], 3, labels=["rendah", "sedang", "tinggi"])
                m = dry.groupby(q, observed=True)["valve_s"].mean()
                env_lines.append(
                    f"Saat tanah kering (<40%), valve rata-rata menyala **{m['tinggi']:.2f} s** ketika {name} tinggi "
                    f"vs **{m['rendah']:.2f} s** ketika rendah (selisih {m['tinggi'] - m['rendah']:+.2f} s)."
                )
            except (ValueError, KeyError):
                pass

    # --- konsumsi per zona ---
    zw = df.groupby("zone_id")["water_liters"].sum().sort_values(ascending=False)
    zone_line = (
        f"{zw.index[0]} memakai air terbanyak ({zw.iloc[0]:,.0f} L, {zw.iloc[0] / zw.sum() * 100:.0f}% dari total) "
        f"dan {zw.index[-1]} paling sedikit ({zw.iloc[-1]:,.0f} L)."
    )

    st.markdown("### Temuan utama")
    st.markdown(
        f"1. **Penyiraman adalah keputusan 'ya/tidak' yang ditentukan kelembapan tanah.** Valve mati pada {idle_share:.0f}% "
        f"pembacaan dan praktis tidak pernah menyala di atas ≈{cutoff:.0f}% soil moisture; di bawah itu durasinya naik tajam "
        f"seiring tanah makin kering (lihat grafik kiri)."
    )
    st.markdown(
        f"2. **Soil moisture mendominasi model.** Ia menyumbang ≈{moist_share:.0f}% kontribusi prediksi. Aturan sederhana "
        f"(valve mati di atas ≈{cutoff:.0f}%, garis linear di bawahnya, hanya memakai soil moisture) mencapai R² **{r2_lin:.3f}** "
        f"vs Random Forest **{r2:.3f}** (selisih {gap:+.3f}). "
        + ("Selisih kecil: fitur lain menambah sedikit." if gap < 0.05 else
           "Sisa selisih berasal dari fitur lain (suhu, cahaya, humidity) dan bentuk kurva yang tidak persis linear.")
    )
    for k, line in enumerate(env_lines, start=3):
        st.markdown(f"{k}. {line} *(korelasi, bukan bukti sebab-akibat.)*")
    nxt = 3 + len(env_lines)
    st.markdown(f"{nxt}. **Air terkonsentrasi di kondisi tanah kering:** {dry_water_share:.0f}% total durasi valve terjadi saat soil moisture <40%.")
    zone_sim = "zone_id" not in raw_df.columns
    st.markdown(f"{nxt + 1}. **Per zona:** {zone_line}" + (" *(zona disimulasikan, jangan ditafsirkan sebagai kondisi lahan nyata.)*" if zone_sim else ""))
    if graph_info:
        gs = graph_info["stats"]
        st.markdown(
            f"{nxt + 2}. **Jaringan pipa:** jalur ke zona terkering ({graph_info['dry_zone']}) = {' → '.join(graph_info['path'])}, "
            f"{gs['Total panjang (m)']:,.0f} m, bottleneck {gs['Kapasitas bottleneck (lpm)']:,.0f} lpm, peluang air sampai tanpa bocor "
            f"{gs['Probabilitas air sampai (tanpa bocor)']}. Ruas dengan risiko bocor tertinggi: **{graph_info['worst_edge']}** "
            f"({graph_info['worst_leak'] * 100:.0f}%)."
        )

    st.markdown("### Rekomendasi")
    st.markdown(
        f"- **Mulai dari aturan ambang sederhana di edge (ESP32):** valve aktif bila soil moisture < ≈{cutoff:.0f}%, durasi mengikuti "
        f"garis linear. Aturan ini sudah mencapai R² {r2_lin:.2f} (model penuh {r2:.2f}) dengan biaya komputasi nyaris nol — "
        f"jadikan baseline yang harus dikalahkan ML.\n"
        f"- **Pakai suhu/cahaya sebagai koreksi, bukan penentu utama,** terutama untuk zona kering — efeknya terukur tapi kecil.\n"
        f"- **Prioritaskan inspeksi ruas pipa berisiko bocor tinggi** ({graph_info.get('worst_edge', '—')}) sebelum menambah kapasitas pompa.\n"
        f"- **Nilai tambah ML baru terlihat bila ada data yang lebih kaya:** prakiraan hujan/evapotranspirasi, jenis tanaman, fase tumbuh, "
        f"dan label volume air nyata dari flow meter."
    )

    st.markdown("### Limitasi yang harus disebut di laporan")
    st.info(
        "• Dataset publik ini kemungkinan besar dibangkitkan dari aturan kontrol, sehingga R² tinggi mencerminkan aturan tersebut, "
        "bukan fenomena lapangan.\n\n"
        "• Timestamp, zona, dan device tidak ada di CSV sumber dan disimulasikan; volume air = durasi × debit asumsi.\n\n"
        "• Topologi graf memakai contoh, bukan jaringan pipa nyata."
    )
