"""
Sentinel XGBoost Classifier v2.0

Supervised classification for notification triage.
Directly predicts: does this notification need intervention?

v1.0 replaced the VAE approach after analysis showed anomaly detection
couldn't separate FPs from real interventions (1.05x separation).
v2.0: event-grouped train/val/test split (no sibling-row leakage),
val-based early stopping with the saved model sliced to the selected
trees, honest per-day math from full-dataset counts, JSON config.
"""

import json
import time
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    f1_score, confusion_matrix,
    roc_auc_score, average_precision_score
)


# ============================================================================
# CONFIGURATION
# ============================================================================

from ml.constants import FEATURE_COLUMNS

# Artifacts live next to this file, regardless of cwd
ML_DIR = Path(__file__).resolve().parent


# ============================================================================
# MAIN TRAINING & EVALUATION
# ============================================================================

def train_and_evaluate():
    print("=" * 70)
    print("SENTINEL XGBOOST CLASSIFIER v2.0")
    print("=" * 70)

    # ========================================
    # LOAD DATA
    # ========================================
    print("\n📥 Loading data...")

    X = np.load(ML_DIR / 'X_all.npy')
    y = np.load(ML_DIR / 'y_all.npy', allow_pickle=True).astype(int)
    notif_types = np.load(ML_DIR / 'notif_types.npy', allow_pickle=True)
    groups = np.load(ML_DIR / 'groups.npy')

    meta_path = ML_DIR / 'dataset_meta.json'
    if not meta_path.exists():
        raise SystemExit(
            "dataset_meta.json not found — run `python -m ml.prepare_data` first "
            "(the per-day numbers need full-dataset counts, not the training sample)."
        )
    with open(meta_path) as f:
        dataset_meta = json.load(f)

    print(f"   Total samples: {len(X):,}")
    print(f"   Events: {len(np.unique(groups)):,}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Positive (needs intervention): {sum(y):,} ({sum(y)/len(y)*100:.1f}%)")
    print(f"   Negative (false positive): {sum(y==0):,} ({sum(y==0)/len(y)*100:.1f}%)")

    # ========================================
    # EVENT-GROUPED TRAIN/VAL/TEST SPLIT
    # ========================================
    # One notification event yields ~4-120 near-identical rows sharing a
    # label. Splitting by event_id keeps sibling rows on one side of every
    # boundary — a random row split would let the model memorize per-event
    # fingerprints and report inflated metrics.
    print("\n📊 Splitting data (grouped by event, ≈70/15/15)...")

    gss_test = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    trainval_idx, test_idx = next(gss_test.split(X, y, groups))

    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.15 / 0.85, random_state=43)
    tr_rel, val_rel = next(gss_val.split(
        X[trainval_idx], y[trainval_idx], groups[trainval_idx]))
    train_idx, val_idx = trainval_idx[tr_rel], trainval_idx[val_rel]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    notif_types_test = notif_types[test_idx]

    for name, arr in (("Train", y_train), ("Val", y_val), ("Test", y_test)):
        print(f"   {name}: {len(arr):,} samples ({arr.mean()*100:.1f}% positive)")

    # ========================================
    # TRAIN XGBOOST
    # ========================================
    print("\n🚀 Training XGBoost classifier...")

    # Calculate scale_pos_weight for imbalanced classes
    neg_count = sum(y_train == 0)
    pos_count = sum(y_train == 1)
    scale_pos_weight = neg_count / pos_count

    print(f"   Class balance - Neg: {neg_count:,}, Pos: {pos_count:,}")
    print(f"   scale_pos_weight: {scale_pos_weight:.2f}")

    # XGBoost parameters
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'aucpr',
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': scale_pos_weight,
        'tree_method': 'hist',  # Fast histogram-based
        'random_state': 42,
    }

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURE_COLUMNS)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=FEATURE_COLUMNS)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=FEATURE_COLUMNS)

    # Early stopping selects on the validation set; the test set stays
    # untouched until final evaluation
    start_time = time.time()

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=30,
        verbose_eval=50
    )

    train_time = time.time() - start_time
    best_iteration = model.best_iteration
    print(f"\n   Training completed in {train_time:.1f}s")
    print(f"   Best iteration: {best_iteration}")

    # Slice to the selected trees BEFORE saving — the artifact then contains
    # exactly the model that early stopping chose, and serving's plain
    # .predict() needs no iteration_range bookkeeping
    model = model[: best_iteration + 1]

    # ========================================
    # PREDICTIONS
    # ========================================
    print("\n🔍 Making predictions...")

    y_prob = model.predict(dtest)
    y_pred = (y_prob >= 0.5).astype(int)

    # ========================================
    # OVERALL RESULTS
    # ========================================
    print(f"\n{'=' * 70}")
    print("OVERALL RESULTS (event-grouped held-out test set)")
    print(f"{'=' * 70}")

    # Baseline metrics (all notifications flagged)
    baseline_precision = sum(y_test) / len(y_test)
    baseline_fp_rate = sum(y_test == 0) / len(y_test)

    # Model metrics
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = f1_score(y_test, y_pred)

    model_fp_rate = fp / (tp + fp) if (tp + fp) > 0 else 0
    fp_reduction = (baseline_fp_rate - model_fp_rate) / baseline_fp_rate * 100

    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    print(f"\n{'Metric':<30} {'Baseline':>15} {'XGBoost':>15} {'Change':>15}")
    print("-" * 75)
    print(f"{'False Positive Rate':<30} {baseline_fp_rate*100:>14.1f}% {model_fp_rate*100:>14.1f}% {fp_reduction:>14.1f}%")
    print(f"{'Precision':<30} {baseline_precision*100:>14.1f}% {precision*100:>14.1f}% {(precision-baseline_precision)/baseline_precision*100:>+14.1f}%")
    print(f"{'Recall':<30} {'100.0':>14}% {recall*100:>14.1f}%")
    print(f"{'F1 Score':<30} {'-':>15} {f1*100:>14.1f}%")
    print(f"{'ROC-AUC':<30} {'-':>15} {roc_auc:>14.3f}")
    print(f"{'PR-AUC':<30} {'-':>15} {pr_auc:>14.3f}")

    print(f"\n   Confusion Matrix:")
    print(f"   TP (caught real): {tp:,}")
    print(f"   FP (false alarm): {fp:,}")
    print(f"   TN (filtered FP): {tn:,}")
    print(f"   FN (missed real): {fn:,}")

    # ========================================
    # PER-NOTIFICATION-TYPE BREAKDOWN
    # ========================================
    print(f"\n{'=' * 70}")
    print("RESULTS BY NOTIFICATION TYPE")
    print(f"{'=' * 70}")

    # Group by notification type
    unique_types = {}
    for i, (ntype, subtype) in enumerate(notif_types_test):
        key = f"{ntype}" + (f"/{subtype}" if subtype else "")
        if key not in unique_types:
            unique_types[key] = []
        unique_types[key].append(i)

    print(f"\n{'Type':<40} {'Baseline FP':>12} {'XGBoost FP':>12} {'Reduction':>11} {'AUC':>7}")
    print("-" * 86)

    for key in sorted(unique_types.keys()):
        indices = unique_types[key]

        type_y = y_test[indices]
        type_pred = y_pred[indices]
        type_prob = y_prob[indices]

        # Baseline FP rate
        type_baseline_fp = sum(type_y == 0) / len(type_y) if len(type_y) > 0 else 0

        # Model FP rate (among predicted positives)
        type_pred_pos = sum(type_pred)
        if type_pred_pos > 0:
            type_model_fp = sum((type_pred == 1) & (type_y == 0)) / type_pred_pos
        else:
            type_model_fp = 0

        reduction = (type_baseline_fp - type_model_fp) / type_baseline_fp * 100 if type_baseline_fp > 0 else 0

        # Per-type discrimination — only defined when both classes appear
        if len(np.unique(type_y)) == 2:
            type_auc = f"{roc_auc_score(type_y, type_prob):.3f}"
        else:
            type_auc = "-"

        print(f"{key:<40} {type_baseline_fp*100:>11.1f}% {type_model_fp*100:>11.1f}% {reduction:>10.1f}% {type_auc:>7}")

    # ========================================
    # FEATURE IMPORTANCE
    # ========================================
    print(f"\n{'=' * 70}")
    print("FEATURE IMPORTANCE (Top 15)")
    print(f"{'=' * 70}")

    importance = model.get_score(importance_type='gain')
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    # Normalize to sum to 1
    total_importance = sum(v for _, v in sorted_importance)

    print(f"\n{'Rank':<6} {'Feature':<35} {'Importance':>12}")
    print("-" * 55)

    for i, (feature, score) in enumerate(sorted_importance[:15], 1):
        normalized = score / total_importance
        bar = "█" * int(normalized * 40)
        print(f"{i:<6} {feature:<35} {normalized:>11.1%} {bar}")

    # ========================================
    # OPERATOR IMPACT
    # ========================================
    # Per-day numbers come from the FULL dataset counts recorded by
    # prepare_data (not the training sample), divided by actual simulated
    # days. "Alerts" are 5-second notification samples; the events/day
    # line gives the count of distinct notification events.
    print(f"\n{'=' * 70}")
    print("OPERATOR IMPACT (full simulated fleet, per simulated day)")
    print(f"{'=' * 70}")

    full_rows = dataset_meta['full_rows']
    full_positives = dataset_meta['full_positives']
    full_events = dataset_meta['full_events']
    sim_days = dataset_meta['sim_days']

    baseline_alerts_per_day = round(full_rows / sim_days)
    baseline_fp_per_day = round((full_rows - full_positives) / sim_days)
    real_per_day = round(full_positives / sim_days)

    # Test-set rates projected onto the full per-day volume
    flag_rate = sum(y_pred) / len(y_test)
    fp_rate_of_all = fp / len(y_test)
    tp_rate_of_all = tp / len(y_test)

    model_alerts_per_day = round(flag_rate * baseline_alerts_per_day)
    model_fp_per_day = round(fp_rate_of_all * baseline_alerts_per_day)
    model_tp_per_day = round(tp_rate_of_all * baseline_alerts_per_day)

    print(f"\n   Simulation: {full_rows:,} notification samples "
          f"({full_events:,} events) over {sim_days} days")
    print(f"   Notification events per day: ~{round(full_events / sim_days):,}")

    print(f"\n{'Metric (per simulated day)':<40} {'Baseline':>15} {'XGBoost':>15}")
    print("-" * 70)
    print(f"{'Alert samples surfaced':<40} {baseline_alerts_per_day:>15,} {model_alerts_per_day:>15,}")
    print(f"{'False alarms surfaced':<40} {baseline_fp_per_day:>15,} {model_fp_per_day:>15,}")
    print(f"{'Real interventions caught':<40} {real_per_day:>15,} {model_tp_per_day:>15,}")

    alerts_filtered = baseline_alerts_per_day - model_alerts_per_day
    fp_filtered = baseline_fp_per_day - model_fp_per_day

    print(f"\n   🎯 XGBoost filters out {alerts_filtered:,} alert samples per day")
    print(f"   🎯 {fp_filtered:,} false alarms eliminated per day")
    print(f"   🎯 Operators handle {model_alerts_per_day/baseline_alerts_per_day*100:.0f}% of previous volume")

    # ========================================
    # SUMMARY TABLE (for README)
    # ========================================
    print(f"\n{'=' * 70}")
    print("SUMMARY (copy for README)")
    print(f"{'=' * 70}")

    print(f"""
| Metric | Baseline | XGBoost | Improvement |
|--------|----------|---------|-------------|
| False Positive Rate | {baseline_fp_rate*100:.1f}% | {model_fp_rate*100:.1f}% | ↓ {fp_reduction:.0f}% |
| Precision | {baseline_precision*100:.1f}% | {precision*100:.1f}% | ↑ {(precision-baseline_precision)/baseline_precision*100:.0f}% |
| Recall | 100% | {recall*100:.1f}% | - |
| F1 Score | - | {f1*100:.1f}% | - |
| ROC-AUC | - | {roc_auc:.3f} | - |
""")

    # ========================================
    # SAVE MODEL
    # ========================================
    print("\n💾 Saving model...")
    model.save_model(ML_DIR / 'xgboost_model.json')

    # Config as JSON (never pickle — committed artifacts shouldn't be
    # arbitrary-code-execution vectors on load)
    config = {
        'feature_columns': FEATURE_COLUMNS,
        'threshold': 0.5,
        'best_iteration': int(best_iteration),
        'n_trees': int(best_iteration) + 1,
        'roc_auc_test': round(float(roc_auc), 4),
        'fp_reduction_pct': round(float(fp_reduction), 1),
        'trained_rows': int(len(X_train)),
        'split': 'event-grouped 70/15/15',
    }
    with open(ML_DIR / 'model_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f"   Saved: xgboost_model.json ({config['n_trees']} trees)")
    print(f"   Saved: model_config.json")

    print(f"\n{'=' * 70}")
    print("TRAINING & EVALUATION COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    train_and_evaluate()
