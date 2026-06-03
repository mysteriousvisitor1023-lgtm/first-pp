# 🔍 DuressDetect AI
### *Identifying Psychologically-Induced Typing Behaviour via Keystroke Dynamics*

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?logo=scikitlearn)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Abstract

**DuressDetect AI** is a novel machine-learning system that determines whether a person typed a message, document, or confession **under psychological pressure or fear** — using nothing but the temporal signature of their keystrokes.

Unlike traditional keystroke-dynamics systems that focus on *who* is typing (identity authentication), this project uniquely addresses *how someone feels* while typing — specifically detecting the biometric fingerprint of **acute psychological duress**.

The system captures timing features such as inter-key intervals, dwell times, flight times, pause patterns, and error rates, then feeds them into a Random-Forest classifier to produce a **duress probability score** with explainable feature attribution.

---

## 🆕 Novel Contribution

> *"To our knowledge, no prior published system specifically targets the detection of duress-induced typing behaviour as distinct from identity verification or generalised emotional-arousal classification."*

| Prior Work | This Work |
|---|---|
| Keystroke dynamics for identity authentication | Keystroke dynamics for **stress state detection** |
| Lab-controlled, task-specific recordings | Open text input, real-world applicable |
| Binary: same person / different person | Binary: **duress / no duress** with probability |
| User-specific models (need enrollment) | **User-independent** classifier |

---

## 🧠 Scientific Background

### Why Does Typing Change Under Stress?

When a person experiences fear or psychological pressure, several physiological changes affect fine motor control:

| Mechanism | Effect on Typing |
|---|---|
| Elevated cortisol / adrenaline | Micro-tremors → higher timing variance |
| Increased cognitive load | More errors, higher backspace rate |
| Hypervigilance | Frequent pauses to check environment |
| Fine motor degradation | Erratic inter-key intervals |
| Dual-task interference | Burst-then-freeze rhythm pattern |
| Sympathetic nervous system activation | Altered key hold (dwell) duration |

These are well-established in the psychophysiology literature and form the basis for our feature engineering.

---

## 🔬 Features Extracted (18 total)

| Feature | Description | Stress Signal |
|---|---|---|
| `mean_dwell` | Average key hold time (ms) | Changes with motor control |
| `std_dwell` | Std dev of dwell time | ↑ under stress (tremor) |
| `cv_dwell` | Coefficient of variation — dwell | ↑ under stress |
| `mean_flight` | Avg time between keyup→keydown | Changes with urgency |
| `std_flight` | Std dev — flight time | ↑ under stress |
| `cv_flight` | Variability of flight time | ↑ under stress |
| `mean_iki` | Mean inter-key interval | Changes with typing speed |
| `std_iki` | Std dev — IKI | ↑ under stress |
| **`cv_iki`** | **Rhythm consistency** | **Primary stress indicator** |
| `wpm` | Typing speed (words per minute) | Disrupted under stress |
| **`backspace_rate`** | **Backspace / total keystrokes** | **↑ under stress** |
| `burst_backspace_rate` | Consecutive error clusters | ↑ under stress |
| **`pause_rate`** | **Long pauses / total IKIs** | **↑ under stress** |
| `mean_pause_dur` | Average pause duration | ↑ under stress |
| `max_pause_dur` | Longest single pause | ↑ under stress |
| `rhythm_irregularity` | Lag-1 autocorrelation of IKI | ↑ under stress |
| `speed_drift` | Change in speed across session | Erratic under stress |
| `hesitation_index` | IKIs > 2× median / total | ↑ under stress |

---

## 🏗️ Architecture

```
┌─────────────────────────────────┐
│      Browser (Streamlit)        │
│  ┌──────────────────────────┐   │
│  │  Keystroke Recorder      │   │
│  │  (HTML/JS component)     │   │
│  │  keydown/keyup + ts      │   │
│  └────────────┬─────────────┘   │
└───────────────┼─────────────────┘
                │ JSON events
                ▼
┌─────────────────────────────────┐
│     Feature Extractor           │
│  utils/feature_extractor.py     │
│  18 biometric features          │
└───────────────┬─────────────────┘
                │ feature vector
                ▼
┌─────────────────────────────────┐
│     ML Pipeline                 │
│  StandardScaler                 │
│     +                           │
│  RandomForestClassifier         │
│  (300 trees, max_depth=12)      │
└───────────────┬─────────────────┘
                │
                ▼
     Duress Probability 0–100%
        + Feature Attribution
```

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| **Accuracy** | 93.3% |
| **AUC-ROC** | 0.925 |
| **Cross-Validation** | 93.9% ± 0.52% |
| **Precision (duress)** | 92.7% |
| **Recall (duress)** | 93.9% |
| **F1-score (duress)** | 93.3% |
| **Training samples** | 2,400 |
| **Test samples** | 600 |

*Note: Trained on synthetic data generated from psychologically-grounded stress profiles. Real-world accuracy would require validation with data from controlled human-subject studies.*

---

## 🗂️ Project Structure

```
duress_detector/
├── app.py                          ← Main Streamlit application
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml                 ← Dark theme configuration
├── components/
│   └── keystroke_recorder/
│       └── index.html              ← Custom JS keystroke capture component
├── data/
│   ├── __init__.py
│   └── synthetic_generator.py      ← Psychologically-grounded data generation
├── models/
│   ├── __init__.py
│   └── model_loader.py             ← Training pipeline + evaluation metrics
├── notebooks/
│   └── DuressDetection_Research.ipynb  ← Full EDA and model training notebook
└── utils/
    ├── __init__.py
    ├── config.py                   ← Constants and feature names
    └── feature_extractor.py        ← 18-feature extraction from raw events
```

---

## 🚀 Quick Start

### Run locally

```bash
git clone https://github.com/YOUR_USERNAME/duress-detect-ai.git
cd duress-detect-ai
pip install -r requirements.txt
streamlit run app.py
```

The model trains automatically on first launch (~8 seconds) and is cached in memory.

### Deploy on Streamlit Cloud

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo → set **Main file path** to `app.py`
4. Click **Deploy** — no configuration needed

---

## 📓 Notebook

Open `notebooks/DuressDetection_Research.ipynb` in Jupyter for:

- Exploratory data analysis (feature distributions, correlation matrix)
- Normal vs Duress typing profiles compared visually
- Multiple model comparison (RF, GBM, SVM, Logistic Regression)
- SHAP value explanations
- ROC / PR curves
- Error analysis

---

## 🔮 Future Work

- **Real data collection** — Validated stress induction paradigms (Trier Social Stress Test) with actual human subjects
- **Deep learning** — LSTM/Transformer on raw IKI sequences (no hand-crafted features)
- **Continuous monitoring** — Sliding-window analysis for real-time risk scores
- **Multi-class** — Distinguish neutral / mild stress / acute fear / coercion
- **User adaptation** — Online learning to personalise baselines per user
- **Physical validation** — Cross-correlate with heart rate, GSR, cortisol levels

---

## 📚 References

1. Monrose, F. & Rubin, A. (2000). *Keystroke dynamics as a biometric for authentication.* Future Generation Computer Systems, 16(4), 351–359.
2. Epp, C., Lippold, M. & Mandryk, R. (2011). *Identifying emotional states using keystroke dynamics.* CHI 2011 Conference on Human Factors in Computing Systems.
3. Bergadano, F., Gunetti, D. & Picardi, C. (2002). *User authentication through keystroke dynamics.* ACM Transactions on Information and System Security, 5(4), 367–397.
4. Loy, C., Lai, W. & Lim, C. (2007). *Keystroke patterns as prosody in digital conversations.* Pattern Recognition Letters.
5. Tsimperidis, I. et al. (2021). *Keystroke dynamics as a biomarker for cognitive impairment.* Applied Sciences, 11(14).
6. Ahmed, A. & Traore, I. (2014). *Biometric recognition based on free-text keystroke dynamics.* IEEE Transactions on Cybernetics, 44(4), 458–472.

---

## 👥 Team

**Final Year B.Tech / BE Project** — Department of Computer Science & AI/ML

*Built with: Python · Streamlit · scikit-learn · Plotly · HTML5/JS*

---

## 📄 License

MIT License — free to use for academic and research purposes.
