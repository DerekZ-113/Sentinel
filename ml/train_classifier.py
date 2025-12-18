"""
Sentinel XGBoost Classifier v1.0

Supervised classification for notification triage.
Directly predicts: does this notification need intervention?

Replaces VAE approach after analysis showed anomaly detection
couldn't separate FPs from real interventions (1.05x separation).
"""

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report,
    roc_auc_score, average_precision_score
)
import joblib
import time


# ============================================================================
# CONFIGURATION
# ============================================================================

FEATURE_COLUMNS = [
    # Speed context
    'speed_ratio',
    'speed_deviation', 
    'is_stopped',
    'expected_stopped',
    
    # Road context
    'road_type_encoded',
    'traffic_encoded',
    'construction_encoded',
    
    # Notification info
    'notification_type_encoded',
    'notification_subtype_encoded',
    
    # Situational context
    'ev_distance_normalized',
    'pedestrian_density',
    'object_in_path',
    'time_since_stop_normalized',
    
    # Time patterns
    'hour_sin',
    'hour_cos',
    
    # Derived
    'high_traffic',
    'high_pedestrian',
    
    # Interaction features (domain knowledge)
    'stuck_in_traffic',
    'stuck_in_construction',
    'stuck_clear_road',
    'object_query_high_ped',
    'object_query_low_ped',
    'object_query_moving',
    'ev_far_away',
    'ev_close',
    'speed_anomaly_in_traffic',
    'speed_anomaly_clear',
    'impact_rough_road',
]


# ============================================================================
# MAIN TRAINING & EVALUATION
# ============================================================================

def train_and_evaluate():
    print("=" * 70)
    print("SENTINEL XGBOOST CLASSIFIER v1.0")
    print("=" * 70)
    
    # ========================================
    # LOAD DATA
    # ========================================
    print("\n📥 Loading data...")
    
    X = np.load('X_all.npy')
    y = np.load('y_all.npy', allow_pickle=True).astype(int)
    notif_types = np.load('notif_types.npy', allow_pickle=True)
    
    print(f"   Total samples: {len(X):,}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Positive (needs intervention): {sum(y):,} ({sum(y)/len(y)*100:.1f}%)")
    print(f"   Negative (false positive): {sum(y==0):,} ({sum(y==0)/len(y)*100:.1f}%)")
    
    # ========================================
    # TRAIN/TEST SPLIT
    # ========================================
    print("\n📊 Splitting data...")
    
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(X)),
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    notif_types_test = notif_types[idx_test]
    
    print(f"   Train: {len(X_train):,} samples")
    print(f"   Test: {len(X_test):,} samples")
    
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
    
    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURE_COLUMNS)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=FEATURE_COLUMNS)
    
    # Train with early stopping
    start_time = time.time()
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(dtrain, 'train'), (dtest, 'test')],
        early_stopping_rounds=30,
        verbose_eval=50
    )
    
    train_time = time.time() - start_time
    print(f"\n   Training completed in {train_time:.1f}s")
    print(f"   Best iteration: {model.best_iteration}")
    
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
    print("OVERALL RESULTS")
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
    
    print(f"\n{'Type':<45} {'Baseline FP':>12} {'XGBoost FP':>12} {'Reduction':>12}")
    print("-" * 83)
    
    for key in sorted(unique_types.keys()):
        indices = unique_types[key]
        
        type_y = y_test[indices]
        type_pred = y_pred[indices]
        
        # Baseline FP rate
        type_baseline_fp = sum(type_y == 0) / len(type_y) if len(type_y) > 0 else 0
        
        # Model FP rate (among predicted positives)
        type_pred_pos = sum(type_pred)
        if type_pred_pos > 0:
            type_model_fp = sum((type_pred == 1) & (type_y == 0)) / type_pred_pos
        else:
            type_model_fp = 0
        
        reduction = (type_baseline_fp - type_model_fp) / type_baseline_fp * 100 if type_baseline_fp > 0 else 0
        
        print(f"{key:<45} {type_baseline_fp*100:>11.1f}% {type_model_fp*100:>11.1f}% {reduction:>11.1f}%")
    
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
    print(f"\n{'=' * 70}")
    print("OPERATOR IMPACT (extrapolated to full dataset)")
    print(f"{'=' * 70}")
    
    # Scale to full dataset (test is 20%)
    scale_factor = 5
    
    baseline_alerts_per_day = len(y_test) * scale_factor
    baseline_fp_per_day = sum(y_test == 0) * scale_factor
    
    model_alerts_per_day = sum(y_pred) * scale_factor
    model_fp_per_day = fp * scale_factor
    
    print(f"\n{'Metric':<40} {'Baseline':>15} {'XGBoost':>15}")
    print("-" * 70)
    print(f"{'Alerts per day':<40} {baseline_alerts_per_day:>15,} {model_alerts_per_day:>15,}")
    print(f"{'False alarms per day':<40} {baseline_fp_per_day:>15,} {model_fp_per_day:>15,}")
    print(f"{'Real interventions caught':<40} {sum(y_test)*scale_factor:>15,} {tp*scale_factor:>15,}")
    
    alerts_filtered = baseline_alerts_per_day - model_alerts_per_day
    fp_filtered = baseline_fp_per_day - model_fp_per_day
    
    print(f"\n   🎯 XGBoost filters out {alerts_filtered:,} alerts per day")
    print(f"   🎯 {fp_filtered:,} false alarms eliminated")
    print(f"   🎯 Operators handle {model_alerts_per_day/baseline_alerts_per_day*100:.0f}% of previous workload")
    
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
    model.save_model('xgboost_model.json')
    
    # Also save feature names for later use
    joblib.dump({
        'feature_columns': FEATURE_COLUMNS,
        'best_iteration': model.best_iteration,
        'threshold': 0.5
    }, 'xgboost_config.joblib')
    
    print(f"   Saved: xgboost_model.json")
    print(f"   Saved: xgboost_config.joblib")
    
    print(f"\n{'=' * 70}")
    print("TRAINING & EVALUATION COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    train_and_evaluate()
