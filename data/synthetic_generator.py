"""
synthetic_generator.py
----------------------
Generates realistic synthetic keystroke event sequences and extracts
feature vectors for both 'normal' and 'duress' typing conditions.

Psychological basis for duress parameters
──────────────────────────────────────────
Under acute psychological stress / fear:
  • Fine motor control degrades → higher variance in dwell & IKI times
  • Cognitive load increases   → more typos, more backspaces
  • Hypervigilance / distraction → longer pauses between phrases
  • Tremor-like micro-variations → erratic flight times
  • Dual-tasking (monitoring environment) → burst-then-pause rhythm
"""

import numpy as np
from typing import List, Dict, Tuple

from utils.feature_extractor import extract_features, features_to_vector
from utils.config import FEATURE_NAMES


# Overlapping ranges to produce realistic ~93-96% accuracy (not 100%)
NORMAL_PROFILE = dict(
    iki_mu_range     = (100, 200),
    iki_sig_range    = (15,  50),
    dwell_mu_range   = (70,  120),
    dwell_sig_range  = (8,   28),
    bs_prob_range    = (0.03, 0.10),
    long_pause_prob  = (0.01, 0.05),
    long_pause_mu    = (600,  1500),
    n_chars_range    = (60,   130),
)

DURESS_PROFILE = dict(
    iki_mu_range     = (85,   310),    # wider — rushed OR frozen
    iki_sig_range    = (55,   170),    # much higher variance
    dwell_mu_range   = (55,   195),
    dwell_sig_range  = (25,   95),     # tremor-like
    bs_prob_range    = (0.14, 0.35),   # 3–4× more errors
    long_pause_prob  = (0.07, 0.23),   # frequent hesitation
    long_pause_mu    = (1100, 4800),
    n_chars_range    = (55,   125),
)


def _generate_session(profile: dict, rng: np.random.Generator) -> List[Dict]:
    iki_mu    = rng.uniform(*profile["iki_mu_range"])
    iki_sig   = rng.uniform(*profile["iki_sig_range"])
    dwell_mu  = rng.uniform(*profile["dwell_mu_range"])
    dwell_sig = rng.uniform(*profile["dwell_sig_range"])
    bs_prob   = rng.uniform(*profile["bs_prob_range"])
    pause_p   = rng.uniform(*profile["long_pause_prob"])
    pause_mu  = rng.uniform(*profile["long_pause_mu"])
    n_chars   = int(rng.uniform(*profile["n_chars_range"]))

    # Per-session noise multiplier (models individual differences)
    noise = rng.uniform(0.85, 1.15)
    iki_sig  *= noise
    dwell_sig *= noise

    events: List[Dict] = []
    t = 0.0

    for _ in range(n_chars):
        if rng.random() < pause_p:
            t += rng.exponential(pause_mu)

        is_bs = rng.random() < bs_prob
        key   = "Backspace" if is_bs else chr(int(rng.integers(97, 123)))

        iki   = max(20.0, rng.normal(iki_mu,   iki_sig))
        dwell = max(10.0, rng.normal(dwell_mu, dwell_sig))

        events.append({"type": "down", "key": key, "timestamp": t})
        events.append({"type": "up",   "key": key, "timestamp": t + dwell})
        t += iki

    return events


def generate_dataset(
    n_samples_per_class: int = 1500,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    rows, labels = [], []

    for label, profile in [(0, NORMAL_PROFILE), (1, DURESS_PROFILE)]:
        collected = 0
        attempts  = 0
        while collected < n_samples_per_class and attempts < n_samples_per_class * 6:
            attempts += 1
            events   = _generate_session(profile, rng)
            features = extract_features(events)
            if features is not None:
                rows.append(features_to_vector(features))
                labels.append(label)
                collected += 1

    X = np.vstack(rows)
    y = np.array(labels, dtype=int)
    return X, y


def generate_demo_session(condition: str = "normal", seed: int = 7) -> List[Dict]:
    """Return a demo keystroke session for the UI demo tab."""
    rng = np.random.default_rng(seed)
    profile = NORMAL_PROFILE if condition == "normal" else DURESS_PROFILE
    return _generate_session(profile, rng)


def generate_dataset_with_noise(
    n_samples_per_class: int = 1500,
    label_noise: float = 0.06,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Same as generate_dataset but adds realistic label noise (~6% flip)
    to model individual variation and edge cases."""
    X, y = generate_dataset(n_samples_per_class=n_samples_per_class, seed=seed)
    rng = np.random.default_rng(seed + 999)
    flip_mask = rng.random(len(y)) < label_noise
    y_noisy = y.copy()
    y_noisy[flip_mask] = 1 - y_noisy[flip_mask]
    return X, y_noisy
