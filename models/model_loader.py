"""
model_loader.py — Trains the DuressDetect pipeline and returns metrics.
"""

import os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score, roc_curve,
)

from data.synthetic_generator import generate_dataset_with_noise
from utils.config import FEATURE_NAMES


def build_and_evaluate():
    print("[DuressDetect] Generating synthetic training data …")
    X, y = generate_dataset_with_noise(n_samples_per_class=1500, label_noise=0.06, seed=42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(
            n_estimators    = 300,
            max_depth       = 12,
            min_samples_leaf= 2,
            class_weight    = "balanced",
            random_state    = 42,
            n_jobs          = -1,
        )),
    ])

    print("[DuressDetect] Training classifier …")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    acc    = accuracy_score(y_test, y_pred)
    auc    = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm     = confusion_matrix(y_test, y_pred).tolist()

    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_sc  = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

    fi     = model.named_steps["clf"].feature_importances_
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    metrics = dict(
        accuracy         = float(acc),
        auc              = float(auc),
        precision_normal = report["0"]["precision"],
        recall_normal    = report["0"]["recall"],
        f1_normal        = report["0"]["f1-score"],
        precision_duress = report["1"]["precision"],
        recall_duress    = report["1"]["recall"],
        f1_duress        = report["1"]["f1-score"],
        confusion_matrix = cm,
        feature_importances = dict(zip(FEATURE_NAMES, fi.tolist())),
        cv_mean          = float(cv_sc.mean()),
        cv_std           = float(cv_sc.std()),
        cv_scores        = cv_sc.tolist(),
        roc_fpr          = fpr.tolist(),
        roc_tpr          = tpr.tolist(),
        n_train          = len(X_train),
        n_test           = len(X_test),
        n_features       = len(FEATURE_NAMES),
    )

    print(f"[DuressDetect] ✓ Accuracy={acc:.4f}  AUC={auc:.4f}  "
          f"CV={cv_sc.mean():.4f}±{cv_sc.std():.4f}")
    return model, metrics
