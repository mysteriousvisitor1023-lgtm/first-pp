"""
DuressDetect AI — app.py
═══════════════════════════════════════════════════════════════════════════════
Detects whether a person typed under psychological duress / fear by
analysing keystroke timing dynamics (dwell time, inter-key intervals,
flight time, pause patterns, error rate) using a Random Forest classifier.

Run locally:  streamlit run app.py
Deploy:       Push to GitHub → connect to Streamlit Cloud
═══════════════════════════════════════════════════════════════════════════════
"""

import os, sys, json, time
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# ── Path setup ───────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.feature_extractor import extract_features, features_to_vector
from utils.config import (
    FEATURE_NAMES, FEATURE_DISPLAY,
    COLOR_SAFE, COLOR_WARN, COLOR_DANGER, COLOR_ACCENT,
    MIN_KEYSTROKES_FOR_ANALYSIS,
)
from models.model_loader import build_and_evaluate
from data.synthetic_generator import generate_demo_session

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DuressDetect AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Rajdhani:wght@400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Rajdhani', sans-serif; }
code, pre, .monospace        { font-family: 'JetBrains Mono', monospace !important; }

/* Header */
.main-title {
  font-size: 2.6rem; font-weight: 700; color: #06b6d4;
  letter-spacing: 2px; text-transform: uppercase; margin-bottom: 0;
}
.main-sub {
  font-size: 1.05rem; color: #94a3b8; letter-spacing: 0.5px;
  margin-top: 4px; margin-bottom: 24px;
}

/* Metric cards */
.metric-card {
  background: #111827; border: 1px solid #1e2a3a;
  border-radius: 10px; padding: 16px 20px; text-align: center;
}
.metric-card .label { font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: 1px; }
.metric-card .value { font-size: 2rem; font-weight: 700; color: #06b6d4; }
.metric-card .sub   { font-size: 12px; color: #475569; margin-top: 2px; }

/* Result banner */
.result-safe   { background:#052e16; border:1.5px solid #22c55e; border-radius:12px; padding:20px 28px; }
.result-duress { background:#2d0a0a; border:1.5px solid #ef4444; border-radius:12px; padding:20px 28px; }
.result-title  { font-size:1.6rem; font-weight:700; letter-spacing:1px; }
.result-conf   { font-size:1rem; color:#94a3b8; margin-top:6px; }

/* Feature bar */
.feat-bar-wrap  { margin: 4px 0; }
.feat-bar-label { font-size:11px; color:#94a3b8; margin-bottom:2px; }
.feat-bar-track { background:#1e293b; border-radius:3px; height:7px; overflow:hidden; }
.feat-bar-fill  { height:100%; border-radius:3px; transition: width 0.6s ease; }

/* Section headers */
.section-hdr {
  font-size:1.25rem; font-weight:700; color:#e2e8f0;
  letter-spacing:1px; border-left:3px solid #06b6d4;
  padding-left:12px; margin:24px 0 12px;
}

/* Info box */
.info-box {
  background:#0f1d2c; border:1px solid #164e63;
  border-radius:8px; padding:14px 18px; font-size:0.92rem;
  color:#bae6fd; line-height:1.7;
}

/* Tag pills */
.tag {
  display:inline-block; background:#1e293b; color:#94a3b8;
  border-radius:20px; font-size:11px; padding:3px 10px; margin:3px 2px;
}

/* Sidebar style */
section[data-testid="stSidebar"] { background:#0d1117 !important; }

/* Tab overrides */
.stTabs [data-baseweb="tab"] { font-size:14px; letter-spacing:0.5px; }

/* Hide Streamlit branding */
#MainMenu {visibility:hidden;}
footer    {visibility:hidden;}
header    {visibility:hidden;}
</style>
""", unsafe_allow_html=True)


# ── Load model (cached — trains once on cold start) ──────────────────────────
@st.cache_resource(show_spinner=False)
def get_model():
    return build_and_evaluate()


# ── Component path ────────────────────────────────────────────────────────────
COMPONENT_PATH = os.path.join(ROOT, "components", "keystroke_recorder")
keystroke_recorder = components.declare_component(
    "keystroke_recorder", path=COMPONENT_PATH
)


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def run_analysis(events, model):
    """Run feature extraction + model prediction on a list of keystroke events."""
    features = extract_features(events)
    if features is None:
        return None, None, None

    X = features_to_vector(features).reshape(1, -1)
    prob_arr  = model.predict_proba(X)[0]
    label     = int(model.predict(X)[0])
    duress_p  = float(prob_arr[1])
    return features, label, duress_p


def render_gauge(duress_pct: float):
    """Render a compact SVG gauge showing duress percentage."""
    import math
    pct   = np.clip(duress_pct, 0, 1)
    angle = pct * 180                         # 0°→safe, 180°→full duress
    rad   = math.radians(angle - 90 + 180)    # -90° offset for left start
    px    = 100 + 70 * math.cos(math.radians(180 - angle))
    py    = 85  - 70 * math.sin(math.radians(180 - angle))

    color = COLOR_SAFE if pct < 0.4 else (COLOR_WARN if pct < 0.65 else COLOR_DANGER)

    svg = f"""
<svg viewBox="0 0 200 110" xmlns="http://www.w3.org/2000/svg" width="100%" style="max-width:260px">
  <defs>
    <linearGradient id="g1" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%"   stop-color="#22c55e"/>
      <stop offset="50%"  stop-color="#f59e0b"/>
      <stop offset="100%" stop-color="#ef4444"/>
    </linearGradient>
  </defs>
  <!-- track -->
  <path d="M 30 85 A 70 70 0 0 1 170 85" fill="none" stroke="#1e293b" stroke-width="12" stroke-linecap="round"/>
  <!-- filled arc -->
  <path d="M 30 85 A 70 70 0 0 1 170 85" fill="none" stroke="url(#g1)"
        stroke-width="12" stroke-linecap="round"
        stroke-dasharray="220" stroke-dashoffset="{220*(1-pct):.1f}"/>
  <!-- needle -->
  <line x1="100" y1="85" x2="{px:.1f}" y2="{py:.1f}"
        stroke="{color}" stroke-width="3" stroke-linecap="round"/>
  <circle cx="100" cy="85" r="5" fill="{color}"/>
  <!-- centre text -->
  <text x="100" y="105" text-anchor="middle" font-size="13" font-weight="700"
        fill="{color}" font-family="JetBrains Mono, monospace">{pct*100:.0f}%</text>
  <text x="100" y="115" text-anchor="middle" font-size="7" fill="#475569"
        font-family="Rajdhani, sans-serif" letter-spacing="1">DURESS PROBABILITY</text>
</svg>
"""
    return svg


def render_feature_bars(features: dict, model_fi: dict):
    """Render feature bars coloured by contribution & value."""
    MAX_TOP = 10
    sorted_feats = sorted(model_fi.items(), key=lambda x: -x[1])[:MAX_TOP]
    html_parts = []
    for fname, importance in sorted_feats:
        raw   = features.get(fname, 0)
        label = FEATURE_DISPLAY.get(fname, fname)
        # Normalise to 0-1 for bar width (rough scale)
        normalised = min(1.0, raw / max(1, {
            "wpm":150, "mean_iki":400, "mean_dwell":200, "mean_flight":300,
            "std_iki":200, "cv_iki":2, "backspace_rate":0.5,
            "pause_rate":0.5, "mean_pause_dur":3000, "max_pause_dur":5000,
            "rhythm_irregularity":2, "hesitation_index":1,
        }.get(fname, 100)))
        bar_color = f"linear-gradient(90deg,{COLOR_ACCENT},{COLOR_DANGER})" if normalised > 0.6 \
                    else f"linear-gradient(90deg,{COLOR_SAFE},{COLOR_ACCENT})"
        val_str = f"{raw:.3f}" if raw < 10 else f"{raw:.1f}"
        html_parts.append(f"""
<div class="feat-bar-wrap">
  <div class="feat-bar-label">{label}
    <span style="float:right;color:#e2e8f0;font-size:11px">{val_str}
      <span style="color:#475569;font-size:10px"> (fi={importance:.3f})</span>
    </span>
  </div>
  <div class="feat-bar-track">
    <div class="feat-bar-fill" style="width:{normalised*100:.1f}%;background:{bar_color}"></div>
  </div>
</div>
""")
    return "\n".join(html_parts)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN UI
# ═══════════════════════════════════════════════════════════════════════════════

# Header
st.markdown('<div class="main-title">🔍 DuressDetect AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="main-sub">Keystroke Dynamics Under Psychological Stress — Final Year Project</div>',
    unsafe_allow_html=True,
)

# Load model
with st.spinner("⚙️  Initialising model (first run only — ~8 seconds)…"):
    model, metrics = get_model()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_live, tab_insights, tab_demo, tab_about = st.tabs([
    "🔬  Live Analysis",
    "📊  Model Insights",
    "🎭  Demo Scenarios",
    "📖  About & Research",
])


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TAB 1 — LIVE ANALYSIS                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝
with tab_live:
    st.markdown('<div class="section-hdr">Live Keystroke Analysis</div>', unsafe_allow_html=True)

    col_l, col_r = st.columns([1.05, 0.95], gap="large")

    with col_l:
        st.markdown(
            '<div class="info-box">Type anything below — a sentence, a statement, a note. '
            'The AI does <b>not</b> read your words; it analyses only the <b>rhythm</b> of your '
            'keystrokes. When you\'ve typed at least 25 characters, click <b>Analyze Typing Pattern</b>.</div>',
            unsafe_allow_html=True,
        )
        st.write("")

        # ── Keystroke component ──────────────────────────────────────────────
        keystroke_data = keystroke_recorder(key="live_recorder", height=285)

    with col_r:
        st.markdown('<div class="section-hdr" style="margin-top:0">Result</div>',
                    unsafe_allow_html=True)

        if (
            keystroke_data
            and isinstance(keystroke_data, dict)
            and keystroke_data.get("action") == "analyze"
        ):
            events = keystroke_data.get("events", [])

            if len([e for e in events if e.get("type") == "down"]) < MIN_KEYSTROKES_FOR_ANALYSIS:
                st.warning(f"⚠️ Need at least {MIN_KEYSTROKES_FOR_ANALYSIS} keystrokes. Keep typing!")
            else:
                features, label, duress_p = run_analysis(events, model)

                if features is None:
                    st.error("Could not extract features. Please type a longer passage.")
                else:
                    # ── Gauge ────────────────────────────────────────────────
                    g_col, v_col = st.columns([1, 1.3])
                    with g_col:
                        st.markdown(render_gauge(duress_p), unsafe_allow_html=True)

                    with v_col:
                        if label == 1:
                            verdict_color = COLOR_DANGER
                            verdict_icon  = "🚨"
                            verdict_label = "DURESS DETECTED"
                            verdict_class = "result-duress"
                            interp = ("The typing pattern shows elevated stress indicators: "
                                      "high rhythm variability, increased error rate, and "
                                      "hesitation patterns consistent with psychological pressure.")
                        elif duress_p > 0.35:
                            verdict_color = COLOR_WARN
                            verdict_icon  = "⚠️"
                            verdict_label = "MILD STRESS SIGNALS"
                            verdict_class = "result-safe"
                            interp = ("Some indicators of stress are present but below the "
                                      "decision boundary. The writer may be moderately anxious "
                                      "or simply an irregular typist.")
                        else:
                            verdict_color = COLOR_SAFE
                            verdict_icon  = "✅"
                            verdict_label = "NORMAL TYPING"
                            verdict_class = "result-safe"
                            interp = ("Keystroke dynamics fall within the normal range. "
                                      "No significant stress or duress indicators detected.")

                        st.markdown(f"""
<div class="{verdict_class}">
  <div class="result-title" style="color:{verdict_color}">{verdict_icon} {verdict_label}</div>
  <div class="result-conf">Confidence: <b>{duress_p*100:.1f}% duress probability</b></div>
  <div style="font-size:0.88rem;color:#94a3b8;margin-top:10px">{interp}</div>
</div>
""", unsafe_allow_html=True)

                    # ── Key stats ────────────────────────────────────────────
                    st.write("")
                    c1, c2, c3, c4 = st.columns(4)
                    wpm_v = round(features.get("wpm", 0), 1)
                    bs_v  = round(features.get("backspace_rate", 0) * 100, 1)
                    pr_v  = round(features.get("pause_rate", 0) * 100, 1)
                    ri_v  = round(features.get("rhythm_irregularity", 0), 3)
                    for col, lbl, val, sub in [
                        (c1,"WPM",wpm_v,"typing speed"),
                        (c2,"Error%",bs_v,"backspace rate"),
                        (c3,"Pause%",pr_v,"hesitation rate"),
                        (c4,"Rhythm",ri_v,"irregularity"),
                    ]:
                        col.markdown(f"""
<div class="metric-card">
  <div class="label">{lbl}</div>
  <div class="value">{val}</div>
  <div class="sub">{sub}</div>
</div>""", unsafe_allow_html=True)

                    # ── Feature bars ─────────────────────────────────────────
                    st.write("")
                    st.markdown("**Top Feature Contributions:**")
                    fi_dict = metrics["feature_importances"]
                    st.markdown(render_feature_bars(features, fi_dict), unsafe_allow_html=True)

        elif keystroke_data and isinstance(keystroke_data, dict) \
                and keystroke_data.get("action") == "reset":
            st.info("Session reset. Start typing again.")
        else:
            st.markdown("""
<div style="color:#334155;font-size:0.9rem;margin-top:30px;text-align:center">
  👆 Type in the box on the left, then click <b>Analyze</b><br/>
  to see your keystroke stress profile
</div>
""", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TAB 2 — MODEL INSIGHTS                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝
with tab_insights:
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        PLOTLY = True
    except ImportError:
        PLOTLY = False

    st.markdown('<div class="section-hdr">Model Performance</div>', unsafe_allow_html=True)

    # ── Top metrics ───────────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    for col, lbl, val, sub in [
        (m1, "Accuracy",  f"{metrics['accuracy']*100:.1f}%",    "overall"),
        (m2, "AUC-ROC",   f"{metrics['auc']:.4f}",              "area under curve"),
        (m3, "Precision", f"{metrics['precision_duress']*100:.1f}%", "duress class"),
        (m4, "Recall",    f"{metrics['recall_duress']*100:.1f}%",    "duress class"),
        (m5, "CV Score",  f"{metrics['cv_mean']*100:.1f}%",     f"±{metrics['cv_std']*100:.1f}%"),
    ]:
        col.markdown(f"""
<div class="metric-card">
  <div class="label">{lbl}</div>
  <div class="value">{val}</div>
  <div class="sub">{sub}</div>
</div>""", unsafe_allow_html=True)

    st.write("")

    if PLOTLY:
        col_fi, col_roc = st.columns(2, gap="large")

        # ── Feature Importance ────────────────────────────────────────────────
        with col_fi:
            st.markdown('<div class="section-hdr">Feature Importance</div>',
                        unsafe_allow_html=True)
            fi_sorted = sorted(metrics["feature_importances"].items(), key=lambda x: x[1])
            labels    = [FEATURE_DISPLAY.get(k, k) for k, _ in fi_sorted]
            values    = [v for _, v in fi_sorted]
            fig_fi = go.Figure(go.Bar(
                x=values, y=labels, orientation="h",
                marker=dict(
                    color=values,
                    colorscale=[[0,"#164e63"],[0.5,"#06b6d4"],[1,"#22d3ee"]],
                    showscale=False,
                ),
                text=[f"{v:.3f}" for v in values],
                textposition="outside",
                textfont=dict(color="#94a3b8", size=10),
            ))
            fig_fi.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=True, gridcolor="#1e293b", color="#64748b"),
                yaxis=dict(showgrid=False, color="#94a3b8", tickfont=dict(size=10)),
                margin=dict(l=0, r=40, t=20, b=20),
                height=380,
            )
            st.plotly_chart(fig_fi, use_container_width=True)

        # ── ROC Curve ─────────────────────────────────────────────────────────
        with col_roc:
            st.markdown('<div class="section-hdr">ROC Curve</div>', unsafe_allow_html=True)
            fpr, tpr = metrics["roc_fpr"], metrics["roc_tpr"]
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                line=dict(color="#06b6d4", width=2.5),
                fill="tozeroy", fillcolor="rgba(6,182,212,0.07)",
                name=f"AUC = {metrics['auc']:.4f}",
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0,1], y=[0,1], mode="lines",
                line=dict(color="#334155", width=1, dash="dash"),
                name="Random Classifier",
            ))
            fig_roc.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(title="False Positive Rate", gridcolor="#1e293b", color="#64748b"),
                yaxis=dict(title="True Positive Rate",  gridcolor="#1e293b", color="#64748b"),
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8")),
                margin=dict(l=0, r=0, t=20, b=20),
                height=380,
            )
            st.plotly_chart(fig_roc, use_container_width=True)

        # ── Confusion Matrix ──────────────────────────────────────────────────
        st.markdown('<div class="section-hdr">Confusion Matrix (Test Set)</div>',
                    unsafe_allow_html=True)
        cm = metrics["confusion_matrix"]
        fig_cm = go.Figure(go.Heatmap(
            z=cm,
            x=["Predicted Normal","Predicted Duress"],
            y=["Actual Normal","Actual Duress"],
            colorscale=[[0,"#0f172a"],[1,"#06b6d4"]],
            showscale=False,
            text=[[str(v) for v in row] for row in cm],
            texttemplate="%{text}",
            textfont=dict(size=24, color="#e2e8f0"),
        ))
        fig_cm.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(color="#94a3b8"), yaxis=dict(color="#94a3b8"),
            margin=dict(l=0, r=0, t=20, b=20),
            height=280,
        )
        col_cm, col_cv = st.columns([1, 1])
        col_cm.plotly_chart(fig_cm, use_container_width=True)

        with col_cv:
            st.markdown('<div class="section-hdr">5-Fold Cross Validation</div>',
                        unsafe_allow_html=True)
            cv_vals = metrics["cv_scores"]
            fig_cv = go.Figure(go.Bar(
                x=[f"Fold {i+1}" for i in range(len(cv_vals))],
                y=[v*100 for v in cv_vals],
                marker=dict(color=[COLOR_ACCENT]*len(cv_vals)),
                text=[f"{v*100:.2f}%" for v in cv_vals],
                textposition="outside",
                textfont=dict(color="#94a3b8"),
            ))
            fig_cv.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(range=[min(cv_vals)*95, 101], gridcolor="#1e293b", color="#64748b"),
                xaxis=dict(color="#64748b"),
                margin=dict(l=0, r=0, t=20, b=20),
                height=280,
            )
            st.plotly_chart(fig_cv, use_container_width=True)

    else:
        # Fallback: plain tables
        fi_df = pd.DataFrame(
            sorted(metrics["feature_importances"].items(), key=lambda x: -x[1]),
            columns=["Feature","Importance"],
        )
        st.dataframe(fi_df, use_container_width=True)

    # ── Model details box ─────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">Model Architecture</div>', unsafe_allow_html=True)
    st.markdown(f"""
<div class="info-box">
<b>Pipeline:</b> StandardScaler → Random Forest Classifier<br/>
<b>Estimators:</b> 300 decision trees &nbsp;|&nbsp;
<b>Max depth:</b> 12 &nbsp;|&nbsp;
<b>Class weights:</b> balanced<br/>
<b>Features:</b> {metrics['n_features']} keystroke dynamics features &nbsp;|&nbsp;
<b>Training samples:</b> {metrics['n_train']:,} &nbsp;|&nbsp;
<b>Test samples:</b> {metrics['n_test']:,}<br/>
<b>Data:</b> Synthetic keystroke sessions generated from psychologically-grounded
stress / non-stress typing profiles with 6% label noise to model individual variation.
</div>
""", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TAB 3 — DEMO SCENARIOS                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝
with tab_demo:
    st.markdown('<div class="section-hdr">Pre-recorded Demo Sessions</div>', unsafe_allow_html=True)
    st.markdown(
        "These demos replay pre-generated keystroke sessions through the full pipeline, "
        "letting you see the system's output without having to type yourself.",
    )
    st.write("")

    col_n, col_d = st.columns(2, gap="large")

    for col, condition, btn_label, seed in [
        (col_n, "normal", "▶  Run Normal Typing Demo",  7),
        (col_d, "duress", "▶  Run Duress Typing Demo", 13),
    ]:
        with col:
            title_color = COLOR_SAFE if condition == "normal" else COLOR_DANGER
            title_text  = "Normal Typing Profile" if condition == "normal" else "Duress / Fear Typing Profile"
            st.markdown(
                f'<div class="section-hdr" style="border-color:{title_color};color:{title_color}">'
                f'{title_text}</div>',
                unsafe_allow_html=True,
            )
            if condition == "normal":
                st.markdown("""
**Characteristics simulated:**
- Consistent inter-key intervals (low CV)
- Low backspace rate (3–10%)
- Minimal long pauses
- Smooth, rhythmic typing
- Stable dwell times
""")
            else:
                st.markdown("""
**Characteristics simulated:**
- Highly erratic timing (high CV of IKI)
- High backspace rate (14–35%)
- Frequent & long hesitation pauses
- Tremor-like dwell time variation
- Burst-then-freeze rhythm pattern
""")

            if st.button(btn_label, key=f"demo_{condition}"):
                with st.spinner(f"Simulating {condition} typing session…"):
                    events   = generate_demo_session(condition=condition, seed=seed)
                    feats, label, duress_p = run_analysis(events, model)

                if feats is None:
                    st.error("Session too short. Try again.")
                else:
                    res_color = COLOR_SAFE if label == 0 else COLOR_DANGER
                    verdict   = "NORMAL" if label == 0 else "DURESS DETECTED"
                    st.markdown(f"""
<div style="border:1.5px solid {res_color};border-radius:10px;padding:16px;margin:10px 0;
            background:{'#052e16' if label==0 else '#2d0a0a'}">
  <div style="font-size:1.4rem;font-weight:700;color:{res_color}">{verdict}</div>
  <div style="color:#94a3b8;font-size:0.9rem;margin-top:4px">
    Duress probability: <b>{duress_p*100:.1f}%</b>
  </div>
</div>
""", unsafe_allow_html=True)
                    st.markdown(render_gauge(duress_p), unsafe_allow_html=True)

                    st.markdown("**Extracted Features:**")
                    demo_df = pd.DataFrame([
                        {
                            "Feature"    : FEATURE_DISPLAY.get(k, k),
                            "Value"      : round(v, 4),
                            "Importance" : round(metrics["feature_importances"].get(k, 0), 4),
                        }
                        for k, v in feats.items()
                    ]).sort_values("Importance", ascending=False)
                    st.dataframe(demo_df, use_container_width=True, hide_index=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TAB 4 — ABOUT & RESEARCH                                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝
with tab_about:
    c1, c2 = st.columns([1.6, 1], gap="large")

    with c1:
        st.markdown('<div class="section-hdr">Project Overview</div>', unsafe_allow_html=True)
        st.markdown("""
**DuressDetect AI** is a novel system that detects whether a person typed a message,
document, or confession **under psychological pressure or fear**, based solely on
the biometric patterns embedded in their keystroke timing — without reading the words.

This problem sits at the intersection of behavioural biometrics, forensic computing,
and affective computing. Prior work on keystroke dynamics focuses on **identity
authentication**; this project uniquely applies it to **emotional state inference**
— specifically the stress signature induced by duress or fear.

### Novel Research Contribution
> *"To our knowledge, no prior work has specifically targeted the detection of
>  **duress-induced typing behaviour** as opposed to identity verification or
>  general emotional arousal detection from keystroke dynamics."*

### Applications
- **Forensic analysis** — Was a confession or ransom note typed under coercion?
- **Workplace safety** — Distress detection for lone workers
- **Banking / legal** — Validating whether digital signatures were made freely
- **Mental health** — Passive monitoring for acute stress episodes
""")

        st.markdown('<div class="section-hdr">Features Extracted</div>', unsafe_allow_html=True)
        feat_table = pd.DataFrame([
            {"Feature": FEATURE_DISPLAY[k], "Code": k,
             "Importance": f"{metrics['feature_importances'].get(k,0):.4f}"}
            for k in FEATURE_NAMES
        ]).sort_values("Importance", ascending=False)
        st.dataframe(feat_table, use_container_width=True, hide_index=True)

    with c2:
        st.markdown('<div class="section-hdr">Research Background</div>', unsafe_allow_html=True)
        st.markdown("""
**Key References**

1. Monrose & Rubin (2000) — *Keystroke dynamics as a biometric for authentication*, Future Generation Computer Systems.

2. Loy et al. (2007) — *Keystroke dynamics in continuous authentication*, IEEE Transactions on Pattern Analysis.

3. Bergadano et al. (2002) — *User authentication through keystroke dynamics*, ACM Transactions on Information and System Security.

4. Epp, Lippold & Mandryk (2011) — *Identifying emotional states using keystroke dynamics*, CHI Conference on Human Factors in Computing Systems.

5. Fairhurst & Da Costa-Abreu (2011) — *Using keystroke dynamics for age and gender classification*, WOSSPA.

6. Tsimperidis et al. (2021) — *Keystroke dynamics as a biomarker for cognitive impairment*, Applied Sciences.

---
""")
        st.markdown('<div class="section-hdr">How It Works</div>', unsafe_allow_html=True)
        st.markdown("""
```
User types → JS captures
  keydown/keyup timestamps
       ↓
Feature extraction
  • dwell time
  • inter-key interval
  • flight time
  • pause patterns
  • error rate
       ↓
StandardScaler → Random Forest
  (300 trees, max depth 12)
       ↓
Duress probability 0–100%
       ↓
Verdict + Explanation
```
""")
        st.markdown('<div class="section-hdr">Dataset</div>', unsafe_allow_html=True)
        st.markdown("""
- **Type:** Synthetic, psychologically-grounded
- **Size:** 3,000 sessions (1,500 normal + 1,500 duress)
- **Label noise:** 6% (models individual variation)
- **Validation:** 5-fold stratified cross-validation
""")


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f'<div style="text-align:center;color:#334155;font-size:0.8rem;letter-spacing:0.5px">'
    f'DuressDetect AI · Final Year Project · '
    f'Accuracy {metrics["accuracy"]*100:.1f}% · AUC {metrics["auc"]:.4f} · '
    f'{metrics["n_features"]} features · Random Forest</div>',
    unsafe_allow_html=True,
)
