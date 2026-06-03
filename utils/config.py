# ─────────────────────────────────────────────────────────────────────────────
# DuressDetect AI — Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Timing thresholds (milliseconds)
PAUSE_THRESHOLD_MS   = 500    # IKI > this → "long pause"
MIN_DWELL_MS         = 15     # ignore sub-15ms dwell (ghost press)
MAX_DWELL_MS         = 1200   # ignore held keys
MIN_IKI_MS           = 10
MAX_IKI_MS           = 6000

# Keys treated as error/correction
BACKSPACE_KEYS = {"Backspace", "Delete"}

# Minimum keystrokes required to make a prediction
MIN_KEYSTROKES_FOR_ANALYSIS = 25

# Ordered list of feature names used for model training & inference
FEATURE_NAMES = [
    "mean_dwell",
    "std_dwell",
    "cv_dwell",
    "mean_flight",
    "std_flight",
    "cv_flight",
    "mean_iki",
    "std_iki",
    "cv_iki",
    "wpm",
    "backspace_rate",
    "burst_backspace_rate",
    "pause_rate",
    "mean_pause_dur",
    "max_pause_dur",
    "rhythm_irregularity",
    "speed_drift",
    "hesitation_index",
]

# Human-readable display names
FEATURE_DISPLAY = {
    "mean_dwell"          : "Mean Key Hold Time (ms)",
    "std_dwell"           : "Std Dev — Key Hold Time",
    "cv_dwell"            : "Variability of Key Hold Time",
    "mean_flight"         : "Mean Flight Time (ms)",
    "std_flight"          : "Std Dev — Flight Time",
    "cv_flight"           : "Variability of Flight Time",
    "mean_iki"            : "Mean Inter-Key Interval (ms)",
    "std_iki"             : "Std Dev — Inter-Key Interval",
    "cv_iki"              : "Typing Rhythm Consistency",
    "wpm"                 : "Typing Speed (WPM)",
    "backspace_rate"      : "Error / Backspace Rate",
    "burst_backspace_rate": "Consecutive Error Burst Rate",
    "pause_rate"          : "Hesitation / Long Pause Rate",
    "mean_pause_dur"      : "Average Pause Duration (ms)",
    "max_pause_dur"       : "Longest Pause (ms)",
    "rhythm_irregularity" : "Rhythm Irregularity Score",
    "speed_drift"         : "Speed Acceleration / Drift",
    "hesitation_index"    : "Hesitation Index",
}

# Colours for Streamlit UI
COLOR_SAFE   = "#22c55e"   # green
COLOR_WARN   = "#f59e0b"   # amber
COLOR_DANGER = "#ef4444"   # red
COLOR_ACCENT = "#06b6d4"   # cyan
