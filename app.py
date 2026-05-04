"""
IIoT Botnet Live Simulation — Streamlit Dashboard
===================================================
Fixes applied vs previous version:
  PERF    — IF result cached in session_state (no recompute every rerun)
  PERF    — spring_layout cached per-timestep (no jitter + no lag)
  PERF    — sim resolved ONCE outside all loops via _get_sim()
  CORE    — CONFIG["duration"] used everywhere (no magic 200)
  CORE    — empty-df guard before .iloc[0]
  UI      — Threat Level metric card with colour emoji
  UI      — Progress bar showing simulation % complete
  UI      — Delta arrows on Traffic metric
  UI      — Network graph legend (node colours + edge types)
  UI      — Coloured edges (C2=orange, DDoS=red, Scan=blue)
  UI      — All st.pyplot(..., use_container_width=True)
  UI      — Reduced figsize throughout
  BONUS   — Live anomaly score subplot alongside traffic
  BONUS   — Live botnet competition stackplot
  BONUS   — Botnet fingerprint scatter (dst_ips vs outbound) in expander
  BONUS   — Auto-run with delay selector
"""

import time

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import streamlit as st

from iiot_badbox_botnet_sim_v2 import (
    CONFIG,
    Simulation,
    build_network_graph,
    run_isolation_forest,
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="IIoT Botnet Sim")
st.title("🔴 IIoT Botnet Live Simulation")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
BOTNET_MODES = ["All (Realistic)", "Mirai Only", "Satori Only", "Persirai Only", "BadBox Only"]

MODE_META = {
    "All (Realistic)": {
        "emoji": "🌐",
        "desc":  "Mixed fleet — Mirai (DDoS), Satori (worm spread), Persirai (exfil), BadBox (proxy/fraud)",
    },
    "Mirai Only": {
        "emoji": "🔴",
        "desc":  "Pure volumetric DDoS botnet — strongest traffic spikes, moderate spread",
    },
    "Satori Only": {
        "emoji": "🟠",
        "desc":  "Fast worm — rapid infection curve, high scan count, many dst IPs",
    },
    "Persirai Only": {
        "emoji": "🟣",
        "desc":  "Stealth exfil — slow spread, very high outbound ratio, dst_ips = 1",
    },
    "BadBox Only": {
        "emoji": "🩵",
        "desc":  "Supply-chain pre-infected — proxy/fraud, no scanning, flat high-outbound traffic",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────────────────────────
if "selected_botnet" not in st.session_state:
    st.session_state.selected_botnet = "All (Realistic)"

if "sim" not in st.session_state:
    st.session_state.sim               = Simulation(selected_botnet=st.session_state.selected_botnet)
    st.session_state.data              = []
    st.session_state.t                 = 0
    st.session_state.df_enriched       = None   # IF cache
    st.session_state.df_enriched_len   = 0
    st.session_state.graph_pos         = None   # layout cache
    st.session_state.graph_pos_t       = -1


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _get_sim() -> Simulation:
    """Single accessor — resolved ONCE per interaction, never inside a loop."""
    return st.session_state.sim


def _run_steps(n: int) -> None:
    """Run up to n steps, capped by CONFIG duration. Invalidates IF cache."""
    sim = _get_sim()
    for _ in range(n):
        if st.session_state.t >= CONFIG["duration"]:
            break
        row = sim.step(st.session_state.t)
        st.session_state.data.append(row)
        st.session_state.t += 1
    # New data → force IF recompute on next render
    st.session_state.df_enriched = None


def _get_enriched(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return Isolation Forest enriched DataFrame.
    CACHED in session_state — only recomputed when new rows arrive.
    Requires >= 5 rows AND at least one normal-phase row for training.
    """
    n = len(df)
    if (
        st.session_state.df_enriched is not None
        and st.session_state.df_enriched_len == n
    ):
        return st.session_state.df_enriched        # cache hit

    if n >= 5 and (df["phase"] == "normal").any():
        try:
            enriched = run_isolation_forest(df)
        except Exception:
            enriched = df
    else:
        enriched = df

    st.session_state.df_enriched     = enriched
    st.session_state.df_enriched_len = n
    return enriched


def _get_graph_pos(G: nx.DiGraph, t: int) -> dict:
    """
    Return spring_layout positions, cached per timestep.
    Prevents node jitter and skips expensive layout recomputation on rerun.
    """
    if st.session_state.graph_pos is None or st.session_state.graph_pos_t != t:
        st.session_state.graph_pos   = nx.spring_layout(G, seed=42)
        st.session_state.graph_pos_t = t
    return st.session_state.graph_pos


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION MODE SELECTOR
# ─────────────────────────────────────────────────────────────────────────────
ms_col, info_col = st.columns([1, 3])

with ms_col:
    new_mode = st.selectbox(
        "Simulation Mode",
        options=BOTNET_MODES,
        index=BOTNET_MODES.index(st.session_state.selected_botnet),
        key="mode_selector",
    )

# If the user changes mode mid-session, auto-reset so it takes effect cleanly
if new_mode != st.session_state.selected_botnet:
    st.session_state.selected_botnet   = new_mode
    st.session_state.sim               = Simulation(selected_botnet=new_mode)
    st.session_state.data              = []
    st.session_state.t                 = 0
    st.session_state.df_enriched       = None
    st.session_state.df_enriched_len   = 0
    st.session_state.graph_pos         = None
    st.session_state.graph_pos_t       = -1
    st.rerun()

with info_col:
    meta = MODE_META[st.session_state.selected_botnet]
    st.info(f"{meta['emoji']} **{st.session_state.selected_botnet}** — {meta['desc']}")

# ─────────────────────────────────────────────────────────────────────────────
# CONTROL BAR
# ─────────────────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])

with c1:
    if st.button("▶️ Next Step"):
        _run_steps(1)

with c2:
    batch_n = st.selectbox(
        "Batch size", options=[5, 10, 15, 20], index=1, label_visibility="collapsed"
    )
    if st.button("⏩ Run Batch"):
        _run_steps(batch_n)
        st.rerun()

with c3:
    auto_n = st.selectbox(
        "Auto steps", options=[10, 20, 50], index=0, label_visibility="collapsed"
    )
    if st.button("🔁 Auto Run"):
        _run_steps(auto_n)
        st.rerun()

with c4:
    # Delay control — adds a visual pacing feel when combined with Auto Run
    auto_delay = st.selectbox(
        "Delay (s)", options=[0.0, 0.2, 0.5, 1.0], index=0, label_visibility="collapsed"
    )
    if auto_delay > 0 and st.session_state.t > 0:
        time.sleep(auto_delay)

with c5:
    if st.button("🔄 Reset"):
        st.session_state.sim             = Simulation(selected_botnet=st.session_state.selected_botnet)
        st.session_state.data            = []
        st.session_state.t               = 0
        st.session_state.df_enriched     = None
        st.session_state.df_enriched_len = 0
        st.session_state.graph_pos       = None
        st.session_state.graph_pos_t     = -1
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# DATA  (build + enrich)
# ─────────────────────────────────────────────────────────────────────────────
df = pd.DataFrame(st.session_state.data)

if df.empty:
    st.info("Click **▶️ Next Step** or **⏩ Run Batch** to start the simulation.")
    st.stop()

df  = _get_enriched(df)     # cached — safe to call on every rerun
row = df.iloc[-1]           # guarded: we already checked df.empty above
t   = int(row["time"])

# ─────────────────────────────────────────────────────────────────────────────
# TOP METRICS
# ─────────────────────────────────────────────────────────────────────────────
PHASE_EMOJI = {
    "normal": "🟢", "compromise": "🟡", "beaconing": "🟠",
    "scanning": "🔵", "attack": "🔴", "recovery": "⚪",
}
st.subheader(
    f"t={t}  {PHASE_EMOJI.get(row['phase'], '')} {row['phase'].capitalize()}"
    f"  |  C2: `{row['c2_command']}`"
    f"  |  {MODE_META[st.session_state.selected_botnet]['emoji']} {st.session_state.selected_botnet}"
)

prev = df.iloc[-2] if len(df) > 1 else None

m1, m2, m3, m4 = st.columns(4)
m1.metric(
    "Traffic (MB)", row["traffic_mb"],
    delta=round(float(row["traffic_mb"]) - float(prev["traffic_mb"]), 1) if prev is not None else None,
)
m2.metric("C2 Beacons",   int(row["c2_beacons"]))
m3.metric("Connections",  int(row["connections"]))
m4.metric("Compromised %", f"{row['pct_compromised']}%")

b1, b2, b3, b4, b5 = st.columns(5)
b1.metric("🔴 Mirai",    int(row.get("b_mirai",    0)))
b2.metric("🟠 Satori",   int(row.get("b_satori",   0)))
b3.metric("🟣 Persirai", int(row.get("b_persirai", 0)))
b4.metric("🩵 BadBox",   int(row.get("b_badbox",   0)))

threat        = row.get("threat_level", None)
threat_emoji  = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(threat, "⚪")
threat_label  = f"{threat_emoji} {threat.upper()}" if isinstance(threat, str) else "—"
b5.metric("Threat Level", threat_label)

# Threat alert banner
if threat == "high":
    st.error("🚨 **HIGH THREAT DETECTED** — Botnet attack activity confirmed. Immediate response recommended.")
elif threat == "medium":
    st.warning("⚠️ **Medium Threat** — Suspicious botnet activity detected. Monitor closely.")

st.progress(
    min(st.session_state.t / CONFIG["duration"], 1.0),
    text=f"Simulation progress: {st.session_state.t} / {CONFIG['duration']} steps",
)

# ─────────────────────────────────────────────────────────────────────────────
# LIVE CHARTS (2-column)
# ─────────────────────────────────────────────────────────────────────────────
lc1, lc2 = st.columns(2)

# ── Network Graph ─────────────────────────────────────────────────────────────
with lc1:
    st.subheader("🌐 Network Graph")

    G, meta = build_network_graph(df, st.session_state.sim, t)
    pos     = _get_graph_pos(G, t)   # cached layout

    fig, ax = plt.subplots(figsize=(4, 3.5))

    c2_edges   = [(u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == "c2_comm"]
    ddos_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == "ddos"]
    scan_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == "scan"]

    nx.draw_networkx_nodes(G, pos, node_color=meta["node_colors"], node_size=200, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=5, ax=ax)
    for edgelist, color in [
        (c2_edges,   "#e67e22"),
        (ddos_edges, "#e74c3c"),
        (scan_edges, "#3498db"),
    ]:
        if edgelist:
            nx.draw_networkx_edges(G, pos, edgelist=edgelist, edge_color=color,
                                   width=0.8, arrows=True, ax=ax, arrowsize=7)

    # Legend — nodes
    node_legend = [
        mpatches.Patch(color="#2ecc71", label="Clean"),
        mpatches.Patch(color="#f39c12", label="Compromised"),
        mpatches.Patch(color="#f1c40f", label="Beaconing"),
        mpatches.Patch(color="#95a5a6", label="Dormant"),
        mpatches.Patch(color="#e74c3c", label="Active"),
        mpatches.Patch(color="#8e44ad", label="Exfiltrating"),
        mpatches.Patch(color="#1abc9c", label="BadBox (proxy/fraud)"),
        mpatches.Patch(color="#2c3e50", label="C2 Server"),
        mpatches.Patch(color="#c0392b", label="Victim"),
    ]
    # Legend — edges
    edge_legend = [
        mpatches.Patch(color="#e67e22", label="→ C2 beacon"),
        mpatches.Patch(color="#e74c3c", label="→ DDoS"),
        mpatches.Patch(color="#3498db", label="→ Scan probe"),
    ]
    ax.legend(handles=node_legend + edge_legend, loc="lower left",
              fontsize=4.5, ncol=2, framealpha=0.75, handlelength=1.0)
    ax.axis("off")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ── Traffic + Anomaly ─────────────────────────────────────────────────────────
with lc2:
    st.subheader("📈 Traffic & Anomaly Score")

    has_anomaly = "anomaly_score" in df.columns
    rows_n      = 2 if has_anomaly else 1
    fig, axes   = plt.subplots(rows_n, 1, figsize=(4, 3.5), sharex=True)
    if rows_n == 1:
        axes = [axes]

    axes[0].plot(df["time"], df["traffic_mb"], color="steelblue", linewidth=1)
    axes[0].set_ylabel("Traffic (MB)", fontsize=7)
    axes[0].set_title("Network Traffic", fontsize=8)
    axes[0].tick_params(labelsize=6)

    if has_anomaly:
        axes[1].plot(df["time"], df["anomaly_score"], color="darkred", linewidth=1)
        axes[1].axhline(CONFIG["threat_medium"], color="orange", linestyle="--",
                        linewidth=0.7, label="Medium")
        axes[1].axhline(CONFIG["threat_high"],   color="red",    linestyle="--",
                        linewidth=0.7, label="High")
        axes[1].set_ylabel("Anomaly Score", fontsize=7)
        axes[1].set_xlabel("Time", fontsize=7)
        axes[1].set_title("Isolation Forest Score", fontsize=8)
        axes[1].tick_params(labelsize=6)
        axes[1].legend(fontsize=5, loc="upper left")

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# LIVE BOTNET COMPETITION
# ─────────────────────────────────────────────────────────────────────────────
if all(c in df.columns for c in ["b_mirai", "b_satori", "b_persirai", "b_badbox"]) and len(df) > 1:
    st.subheader("🦠 Live Botnet Competition")
    fig, ax = plt.subplots(figsize=(7, 2.2))
    ax.stackplot(
        df["time"],
        df["b_mirai"], df["b_satori"], df["b_persirai"], df["b_badbox"],
        labels=["Mirai 🔴", "Satori 🟠", "Persirai 🟣", "BadBox 🩵"],
        colors=["#e74c3c", "#e67e22", "#8e44ad", "#1abc9c"],
        alpha=0.82,
    )
    ax.set_xlabel("Time", fontsize=7)
    ax.set_ylabel("Bot Count", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=7, loc="upper left", ncol=3)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# BOTNET FINGERPRINT SCATTER  (in expander — optional high-impact)
# ─────────────────────────────────────────────────────────────────────────────
if "unique_dst_ips" in df.columns and "outbound_traffic_ratio" in df.columns and len(df) > 5:
    with st.expander("🔬 Botnet Fingerprint — dst_ips vs Outbound Ratio", expanded=False):
        fig, ax = plt.subplots(figsize=(5, 3))
        _phase_colors = {
            "normal":     "#2ecc71",
            "compromise": "#f39c12",
            "beaconing":  "#f1c40f",
            "scanning":   "#3498db",
            "attack":     "#e74c3c",
            "recovery":   "#95a5a6",
        }
        for ph in df["phase"].unique():
            sub = df[df["phase"] == ph]
            ax.scatter(sub["unique_dst_ips"], sub["outbound_traffic_ratio"],
                       label=ph, color=_phase_colors.get(ph, "#bdc3c7"),
                       s=16, alpha=0.75)
        ax.set_xlabel("Unique Dst IPs", fontsize=8)
        ax.set_ylabel("Outbound Ratio", fontsize=8)
        ax.set_title("Botnet Fingerprint by Phase", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, ncol=2)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption(
            "**Mirai** → high dst_ips + high outbound during attack.  "
            "**Persirai** → dst_ips=1, very high outbound (stealthy exfil).  "
            "**Satori** → many dst_ips (worm scanning), moderate outbound.  "
            "**BadBox** → moderate dst_ips (5–20), very high outbound (proxy routing), flat traffic."
        )

# ─────────────────────────────────────────────────────────────────────────────
# ALL GRAPHS — shown at all times once data exists (no gate, no button)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.header("📊 Live Simulation Graphs")

adf = df   # already enriched

def _panel(ax, x, y, color, title, ylabel, xlabel="Time"):
    ax.plot(x, y, color=color, linewidth=1)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=7)
    ax.set_ylabel(ylabel, fontsize=7)
    ax.tick_params(labelsize=6)

# Row 1 — Traffic | C2 Beacons
r1a, r1b = st.columns(2)
with r1a:
    fig, ax = plt.subplots(figsize=(4, 2.5))
    _panel(ax, adf["time"], adf["traffic_mb"], "steelblue", "Traffic vs Time", "MB")
    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)
with r1b:
    fig, ax = plt.subplots(figsize=(4, 2.5))
    _panel(ax, adf["time"], adf["c2_beacons"], "darkorange", "C2 Beacons vs Time", "Beacons")
    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

# Row 2 — Connections | ① Infection Curve
r2a, r2b = st.columns(2)
with r2a:
    fig, ax = plt.subplots(figsize=(4, 2.5))
    _panel(ax, adf["time"], adf["connections"], "green", "Connections vs Time", "Count")
    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)
with r2b:
    fig, ax = plt.subplots(figsize=(4, 2.5))
    ax.plot(adf["time"], adf["pct_compromised"], color="crimson", linewidth=1.2)
    ax.fill_between(adf["time"], adf["pct_compromised"], alpha=0.18, color="crimson")
    ax.set_title("① Infection Curve — % Devices Compromised", fontsize=9)
    ax.set_xlabel("Time", fontsize=7); ax.set_ylabel("Compromised (%)", fontsize=7)
    ax.tick_params(labelsize=6)
    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

# Row 3 — Exfiltration | ② Scan Activity
r3a, r3b = st.columns(2)
with r3a:
    fig, ax = plt.subplots(figsize=(4, 2.5))
    ax.fill_between(adf["time"], adf["exfiltration_mb"], color="purple", alpha=0.6)
    ax.set_title("Exfiltration vs Time", fontsize=9)
    ax.set_xlabel("Time", fontsize=7); ax.set_ylabel("MB", fontsize=7)
    ax.tick_params(labelsize=6)
    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)
with r3b:
    fig, ax = plt.subplots(figsize=(4, 2.5))
    ax.bar(adf["time"], adf["scan_attempts"], color="teal", alpha=0.75,
           label="Scan Attempts", width=1.0)
    ax.set_title("② Scan Activity Over Time", fontsize=9)
    ax.set_xlabel("Time", fontsize=7); ax.set_ylabel("Scan Attempts", fontsize=7)
    ax.tick_params(labelsize=6); ax.legend(fontsize=6)
    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

# Row 4 — ③ Outbound Ratio | ④ Target DDoS Load
r4a, r4b = st.columns(2)
with r4a:
    fig, ax = plt.subplots(figsize=(4, 2.5))
    ax.plot(adf["time"], adf["outbound_traffic_ratio"], color="#8e44ad", linewidth=1.2)
    ax.fill_between(adf["time"], adf["outbound_traffic_ratio"], alpha=0.15, color="#8e44ad")
    ax.axhline(0.85, color="#1abc9c", linestyle="--", linewidth=0.8, label="BadBox/Persirai ≥0.85")
    ax.set_title("③ Outbound Ratio Trend", fontsize=9)
    ax.set_xlabel("Time", fontsize=7); ax.set_ylabel("Outbound Ratio (0–1)", fontsize=7)
    ax.tick_params(labelsize=6); ax.legend(fontsize=6)
    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)
with r4b:
    fig, ax = plt.subplots(figsize=(4, 2.5))
    ax.plot(adf["time"], adf["target_ddos_load"], color="firebrick", linewidth=1.2)
    ax.fill_between(adf["time"], adf["target_ddos_load"], alpha=0.18, color="firebrick")
    ax.axhline(500, color="black", linestyle=":", linewidth=0.8, label="Degraded (500 MB)")
    ax.set_title("④ Target DDoS Load (Cumulative)", fontsize=9)
    ax.set_xlabel("Time", fontsize=7); ax.set_ylabel("Cumulative MB", fontsize=7)
    ax.tick_params(labelsize=6); ax.legend(fontsize=6)
    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

# Row 5 — Scanning detail | C2 Comms breakdown
r5a, r5b = st.columns(2)
with r5a:
    fig, ax = plt.subplots(figsize=(4, 2.5))
    ax.plot(adf["time"], adf["scan_attempts"], color="teal",   linewidth=1, label="Scan Attempts")
    ax.plot(adf["time"], adf["failed_logins"],  color="salmon", linewidth=1,
            linestyle="--", label="Failed Logins")
    ax.set_title("Scanning Activity Detail", fontsize=9)
    ax.set_xlabel("Time", fontsize=7); ax.set_ylabel("Count", fontsize=7)
    ax.tick_params(labelsize=6); ax.legend(fontsize=6)
    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)
with r5b:
    fig, ax = plt.subplots(figsize=(4, 2.5))
    ax.bar(adf["time"], adf["c2_beacons"],   label="Beacons", color="darkorange", alpha=0.7, width=1.0)
    ax.bar(adf["time"], adf["scan_attempts"], bottom=adf["c2_beacons"],
           label="Scans", color="slategray", alpha=0.7, width=1.0)
    ax.set_title("C2 Comms (Beacons + Scans)", fontsize=9)
    ax.set_xlabel("Time", fontsize=7); ax.set_ylabel("Count", fontsize=7)
    ax.tick_params(labelsize=6); ax.legend(fontsize=6)
    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

# Row 6 — Botnet Distribution (all 4 botnets)
if all(c in adf.columns for c in ["b_mirai", "b_satori", "b_persirai", "b_badbox"]):
    st.subheader("🦠 Botnet Type Distribution Over Time")
    fig, ax = plt.subplots(figsize=(7, 2.5))
    ax.stackplot(
        adf["time"],
        adf["b_mirai"], adf["b_satori"], adf["b_persirai"], adf["b_badbox"],
        labels=["Mirai 🔴", "Satori 🟠", "Persirai 🟣", "BadBox 🩵"],
        colors=["#e74c3c", "#e67e22", "#8e44ad", "#1abc9c"],
        alpha=0.82,
    )
    ax.set_xlabel("Time", fontsize=7); ax.set_ylabel("Bot Count", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=7, loc="upper left", ncol=4)
    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

# Row 7 — ⑤ Detection Timeline
if "anomaly_score" in adf.columns:
    st.subheader("⑤ Detection Timeline — Isolation Forest Anomaly Score")
    fig, ax = plt.subplots(figsize=(7, 2.8))
    _phase_bg = {
        "normal": "#d4edda", "compromise": "#fff3cd", "beaconing": "#fde8d8",
        "scanning": "#fddde6", "attack": "#f5c6cb", "recovery": "#d1ecf1",
    }
    for ph, color in _phase_bg.items():
        rows = adf[adf["phase"] == ph]
        if not rows.empty:
            ax.axvspan(rows["time"].min(), rows["time"].max(),
                       alpha=0.25, color=color, linewidth=0)
    ax.plot(adf["time"], adf["anomaly_score"], color="darkred", linewidth=1.2,
            label="Anomaly Score", zorder=3)
    ax.axhline(CONFIG["threat_medium"], color="orange", linestyle="--",
               linewidth=0.9, label=f"Medium ≥ {CONFIG['threat_medium']}", zorder=2)
    ax.axhline(CONFIG["threat_high"], color="red", linestyle="--",
               linewidth=0.9, label=f"High ≥ {CONFIG['threat_high']}", zorder=2)
    if "if_anomaly" in adf.columns:
        anomalies = adf[adf["if_anomaly"]]
        ax.scatter(anomalies["time"], anomalies["anomaly_score"],
                   color="red", s=14, zorder=4, label=f"Flagged ({len(anomalies)} pts)")
    ax.set_xlabel("Time", fontsize=7); ax.set_ylabel("Score (0–1)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=6, loc="upper left", ncol=3)
    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)