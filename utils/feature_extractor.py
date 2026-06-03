"""
feature_extractor.py
--------------------
Converts a raw list of keystroke events (keydown / keyup with timestamps)
into a fixed-length feature vector used by the duress-detection model.

Each event is a dict:
    { "type": "down"|"up", "key": "<key_name>", "timestamp": <float ms> }
"""

import numpy as np
from typing import List, Dict, Optional

from utils.config import (
    PAUSE_THRESHOLD_MS, BACKSPACE_KEYS,
    MIN_DWELL_MS, MAX_DWELL_MS,
    MIN_IKI_MS, MAX_IKI_MS,
    MIN_KEYSTROKES_FOR_ANALYSIS, FEATURE_NAMES,
)


def extract_features(events: List[Dict]) -> Optional[Dict[str, float]]:
    """
    Extract 18 keystroke-dynamics features from a list of key events.

    Parameters
    ----------
    events : list of dicts
        Raw keystroke events with keys: type, key, timestamp

    Returns
    -------
    dict  – feature_name → float value, or None if too few keystrokes
    """
    if not events:
        return None

    downs = [e for e in events if e["type"] == "down"]
    if len(downs) < MIN_KEYSTROKES_FOR_ANALYSIS:
        return None

    # ── 1. Dwell times (keydown→keyup for the same key) ──────────────────────
    dwell_times: List[float] = []
    pending: Dict[str, float] = {}          # key → last down-timestamp

    for e in events:
        k = e["key"]
        t = e["timestamp"]
        if e["type"] == "down":
            pending[k] = t
        elif e["type"] == "up" and k in pending:
            d = t - pending.pop(k)
            if MIN_DWELL_MS < d < MAX_DWELL_MS:
                dwell_times.append(d)

    # ── 2. Inter-key intervals (consecutive keydowns) ────────────────────────
    down_ts = np.array([e["timestamp"] for e in downs])
    ikis_raw = np.diff(down_ts)
    ikis = ikis_raw[(ikis_raw >= MIN_IKI_MS) & (ikis_raw <= MAX_IKI_MS)]

    # ── 3. Flight times (keyup → next keydown) ───────────────────────────────
    up_ts = np.array([e["timestamp"] for e in events if e["type"] == "up"])
    flight_times: List[float] = []
    ui = di = 0
    while ui < len(up_ts) and di < len(down_ts):
        if down_ts[di] > up_ts[ui]:
            ft = down_ts[di] - up_ts[ui]
            if 0 < ft < 3000:
                flight_times.append(ft)
            ui += 1
        else:
            di += 1

    # ── 4. Backspace / error metrics ─────────────────────────────────────────
    bs_mask = [e["key"] in BACKSPACE_KEYS for e in downs]
    n_bs    = sum(bs_mask)
    n_total = len(downs)

    backspace_rate = n_bs / n_total if n_total else 0.0

    # Burst backspace: consecutive backspaces / total backspaces
    bursts = 0
    for i in range(1, len(bs_mask)):
        if bs_mask[i] and bs_mask[i - 1]:
            bursts += 1
    burst_backspace_rate = bursts / max(n_bs, 1)

    # ── 5. Pause / hesitation metrics ────────────────────────────────────────
    long_pauses = ikis[ikis > PAUSE_THRESHOLD_MS] if len(ikis) > 0 else np.array([])
    pause_rate      = len(long_pauses) / max(len(ikis), 1)
    mean_pause_dur  = float(np.mean(long_pauses))  if len(long_pauses) > 0 else 0.0
    max_pause_dur   = float(np.max(long_pauses))   if len(long_pauses) > 0 else 0.0

    # ── 6. Typing speed (WPM) ────────────────────────────────────────────────
    if len(down_ts) > 1:
        total_sec = (down_ts[-1] - down_ts[0]) / 1000.0
        total_min = max(total_sec / 60.0, 1e-6)
        net_chars = n_total - n_bs
        wpm = (net_chars / 5.0) / total_min
    else:
        wpm = 0.0

    # ── 7. Rhythm irregularity (lag-1 autocorrelation of IKI) ────────────────
    if len(ikis) >= 6:
        mu, sig = np.mean(ikis), np.std(ikis) + 1e-10
        normed  = (ikis - mu) / sig
        lag1_corr = np.corrcoef(normed[:-1], normed[1:])[0, 1]
        rhythm_irregularity = float(np.clip(1.0 - lag1_corr, 0.0, 2.0))
    else:
        rhythm_irregularity = 1.0

    # ── 8. Speed drift: is typing getting faster or slower? ──────────────────
    if len(ikis) >= 9:
        third = max(len(ikis) // 3, 1)
        mu_first = np.mean(ikis[:third])
        mu_last  = np.mean(ikis[-third:])
        speed_drift = float((mu_last - mu_first) / (mu_first + 1e-10))
    else:
        speed_drift = 0.0

    # ── 9. Hesitation index: proportion of IKIs > 2× median ─────────────────
    if len(ikis) > 4:
        med = np.median(ikis)
        hesitation_index = float(np.mean(ikis > 2 * med))
    else:
        hesitation_index = 0.0

    # ── Helper: safe stats ───────────────────────────────────────────────────
    def _safe(arr, fn):
        return float(fn(arr)) if len(arr) > 0 else 0.0

    def _cv(arr):
        m = _safe(arr, np.mean)
        return float(np.std(arr) / (m + 1e-10)) if len(arr) > 1 else 0.0

    dwell_arr  = np.array(dwell_times) if dwell_times else np.array([0.0])
    flight_arr = np.array(flight_times) if flight_times else np.array([0.0])

    features = {
        "mean_dwell"          : _safe(dwell_arr, np.mean),
        "std_dwell"           : _safe(dwell_arr, np.std),
        "cv_dwell"            : _cv(dwell_arr),
        "mean_flight"         : _safe(flight_arr, np.mean),
        "std_flight"          : _safe(flight_arr, np.std),
        "cv_flight"           : _cv(flight_arr),
        "mean_iki"            : _safe(ikis, np.mean),
        "std_iki"             : _safe(ikis, np.std),
        "cv_iki"              : _cv(ikis),
        "wpm"                 : wpm,
        "backspace_rate"      : backspace_rate,
        "burst_backspace_rate": burst_backspace_rate,
        "pause_rate"          : pause_rate,
        "mean_pause_dur"      : mean_pause_dur,
        "max_pause_dur"       : max_pause_dur,
        "rhythm_irregularity" : rhythm_irregularity,
        "speed_drift"         : speed_drift,
        "hesitation_index"    : hesitation_index,
    }

    return features


def features_to_vector(features: Dict[str, float]) -> np.ndarray:
    """Convert feature dict → numpy array in canonical order."""
    return np.array([features[f] for f in FEATURE_NAMES], dtype=np.float32)
