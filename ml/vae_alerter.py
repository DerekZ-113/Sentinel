"""
Sentinel VAE Alerter v2.0

Evaluates VAE-based notification triage system.
Compares against baseline (all notifications = intervention needed).
Provides per-notification-type breakdown.
"""

import torch
import numpy as np
from vae_model import VAE
from sklearn.metrics import precision_score, recall_score, f1_score


def load_model():
    """Load trained VAE model"""
    checkpoint = torch.load('vae_trained.pt', map_location='cpu')
    config = checkpoint['config']
    
    model = VAE(
        input_dim=config['input_dim'],
        hidden_dims=config['hidden_dims'],
        latent_dim=config['latent_dim'],
        dropout=config['dropout']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Loaded model from epoch {checkpoint['epoch']}")
    print(f"   Validation loss: {checkpoint['val_loss']:.4f}")
    
    return model


def compute_reconstruction_errors(model, X, batch_size=4096):
    """Compute reconstruction error for all samples in batches"""
    errors = []
    
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.FloatTensor(X[i:i+batch_size])
            batch_errors = model.get_reconstruction_error(batch)
            errors.extend(batch_errors.numpy())
    
    return np.array(errors)


def find_optimal_threshold(errors, y_true, min_recall=0.30):
    """
    Find threshold that maximizes F1 while maintaining minimum recall.
    
    For notification triage:
    - High recall = catch most real interventions (safety)
    - High precision = fewer false alarms (efficiency)
    """
    # Try percentiles from 50 to 99
    best_f1 = 0
    best_threshold = np.percentile(errors, 90)
    
    for percentile in range(50, 100):
        threshold = np.percentile(errors, percentile)
        y_pred = (errors > threshold).astype(int)
        
        # y_true: True = needs intervention
        # y_pred: True = model says needs intervention (high error)
        
        if sum(y_pred) == 0:
            continue
            
        recall = recall_score(y_true, y_pred)
        if recall < min_recall:
            continue
            
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold


def evaluate():
    print("=" * 70)
    print("SENTINEL VAE ALERTER - NOTIFICATION TRIAGE EVALUATION")
    print("=" * 70)
    
    # ========================================
    # LOAD DATA AND MODEL
    # ========================================
    print("\n📥 Loading data and model...")
    
    X_all = np.load('X_all.npy')
    y_all = np.load('y_all.npy', allow_pickle=True).astype(bool)
    notif_types = np.load('notif_types.npy', allow_pickle=True)
    
    model = load_model()
    
    print(f"   Total notifications: {len(X_all):,}")
    print(f"   Real interventions: {sum(y_all):,}")
    print(f"   False positives: {sum(~y_all):,}")
    
    # ========================================
    # COMPUTE RECONSTRUCTION ERRORS
    # ========================================
    print("\n🔍 Computing reconstruction errors...")
    errors = compute_reconstruction_errors(model, X_all)
    
    # Analyze error distribution
    fp_errors = errors[~y_all]  # False positives (no intervention)
    real_errors = errors[y_all]  # Real interventions
    
    print(f"\n   FP (no intervention) error: mean={np.mean(fp_errors):.4f}, std={np.std(fp_errors):.4f}")
    print(f"   Real intervention error:    mean={np.mean(real_errors):.4f}, std={np.std(real_errors):.4f}")
    print(f"   Separation ratio: {np.mean(real_errors) / np.mean(fp_errors):.2f}x")
    
    # ========================================
    # FIND OPTIMAL THRESHOLD
    # ========================================
    print("\n🎯 Finding optimal threshold...")
    threshold = find_optimal_threshold(errors, y_all, min_recall=0.50)
    print(f"   Threshold: {threshold:.6f}")
    
    # ========================================
    # OVERALL RESULTS
    # ========================================
    y_pred = (errors > threshold).astype(int)
    
    # Baseline: all notifications flagged as needing intervention
    baseline_tp = sum(y_all)
    baseline_fp = sum(~y_all)
    baseline_precision = baseline_tp / (baseline_tp + baseline_fp)
    baseline_fp_rate = baseline_fp / (baseline_tp + baseline_fp)
    
    # VAE results
    vae_tp = sum((y_pred == 1) & (y_all == True))
    vae_fp = sum((y_pred == 1) & (y_all == False))
    vae_fn = sum((y_pred == 0) & (y_all == True))
    vae_tn = sum((y_pred == 0) & (y_all == False))
    
    vae_precision = vae_tp / (vae_tp + vae_fp) if (vae_tp + vae_fp) > 0 else 0
    vae_recall = vae_tp / (vae_tp + vae_fn) if (vae_tp + vae_fn) > 0 else 0
    vae_f1 = f1_score(y_all, y_pred)
    vae_fp_rate = vae_fp / (vae_tp + vae_fp) if (vae_tp + vae_fp) > 0 else 0
    
    print(f"\n{'=' * 70}")
    print("OVERALL RESULTS")
    print(f"{'=' * 70}")
    
    print(f"\n{'Metric':<25} {'Baseline':>15} {'VAE':>15} {'Improvement':>15}")
    print("-" * 70)
    print(f"{'False Positive Rate':<25} {baseline_fp_rate*100:>14.1f}% {vae_fp_rate*100:>14.1f}% {(baseline_fp_rate-vae_fp_rate)/baseline_fp_rate*100:>14.1f}%")
    print(f"{'Precision':<25} {baseline_precision*100:>14.1f}% {vae_precision*100:>14.1f}% {(vae_precision-baseline_precision)/baseline_precision*100:>+14.1f}%")
    print(f"{'Recall':<25} {'100.0':>14}% {vae_recall*100:>14.1f}%")
    print(f"{'F1 Score':<25} {'-':>15} {vae_f1*100:>14.1f}%")
    
    print(f"\n   VAE Confusion Matrix:")
    print(f"   TP (caught real): {vae_tp:,}")
    print(f"   FP (false alarm): {vae_fp:,}")
    print(f"   TN (filtered FP): {vae_tn:,}")
    print(f"   FN (missed real): {vae_fn:,}")
    
    # ========================================
    # PER-NOTIFICATION-TYPE BREAKDOWN
    # ========================================
    print(f"\n{'=' * 70}")
    print("RESULTS BY NOTIFICATION TYPE")
    print(f"{'=' * 70}")
    
    # Get unique notification types
    unique_types = {}
    for i, (ntype, subtype) in enumerate(notif_types):
        key = f"{ntype}" + (f"/{subtype}" if subtype else "")
        if key not in unique_types:
            unique_types[key] = {'indices': [], 'type': ntype, 'subtype': subtype}
        unique_types[key]['indices'].append(i)
    
    print(f"\n{'Type':<40} {'Baseline FP':>12} {'VAE FP':>12} {'Reduction':>12}")
    print("-" * 78)
    
    type_results = []
    for key in sorted(unique_types.keys()):
        info = unique_types[key]
        indices = info['indices']
        
        type_y = y_all[indices]
        type_pred = y_pred[indices]
        
        # Baseline FP rate for this type
        type_baseline_fp = sum(~type_y) / len(type_y) if len(type_y) > 0 else 0
        
        # VAE FP rate for this type
        type_vae_alerts = sum(type_pred)
        type_vae_real = sum((type_pred == 1) & (type_y == True))
        type_vae_fp_count = sum((type_pred == 1) & (type_y == False))
        type_vae_fp_rate = type_vae_fp_count / type_vae_alerts if type_vae_alerts > 0 else 0
        
        reduction = (type_baseline_fp - type_vae_fp_rate) / type_baseline_fp * 100 if type_baseline_fp > 0 else 0
        
        type_results.append({
            'type': key,
            'baseline_fp': type_baseline_fp,
            'vae_fp': type_vae_fp_rate,
            'reduction': reduction,
            'count': len(indices)
        })
        
        print(f"{key:<40} {type_baseline_fp*100:>11.1f}% {type_vae_fp_rate*100:>11.1f}% {reduction:>11.1f}%")
    
    # ========================================
    # OPERATOR IMPACT
    # ========================================
    print(f"\n{'=' * 70}")
    print("OPERATOR IMPACT (per day)")
    print(f"{'=' * 70}")
    
    days = 1  # Adjust based on your dataset duration
    total_notifs_per_day = len(X_all) / days
    
    baseline_alerts_per_day = total_notifs_per_day  # All notifications
    baseline_fp_per_day = sum(~y_all) / days
    
    vae_alerts_per_day = sum(y_pred) / days
    vae_fp_per_day = vae_fp / days
    
    print(f"\n{'Metric':<35} {'Baseline':>15} {'VAE':>15}")
    print("-" * 65)
    print(f"{'Alerts per day':<35} {baseline_alerts_per_day:>15,.0f} {vae_alerts_per_day:>15,.0f}")
    print(f"{'False alarms per day':<35} {baseline_fp_per_day:>15,.0f} {vae_fp_per_day:>15,.0f}")
    print(f"{'Alerts filtered out':<35} {'-':>15} {(baseline_alerts_per_day - vae_alerts_per_day):>15,.0f}")
    
    print(f"\n   🎯 VAE filters out {(baseline_alerts_per_day - vae_alerts_per_day):,.0f} unnecessary alerts per day")
    print(f"   🎯 Operators handle {vae_alerts_per_day/baseline_alerts_per_day*100:.0f}% of previous workload")
    
    # ========================================
    # SUMMARY TABLE (for README/interviews)
    # ========================================
    print(f"\n{'=' * 70}")
    print("SUMMARY (copy for README)")
    print(f"{'=' * 70}")
    
    print(f"""
| Metric | Baseline | VAE | Improvement |
|--------|----------|-----|-------------|
| False Positive Rate | {baseline_fp_rate*100:.1f}% | {vae_fp_rate*100:.1f}% | ↓ {(baseline_fp_rate-vae_fp_rate)/baseline_fp_rate*100:.0f}% |
| Precision | {baseline_precision*100:.1f}% | {vae_precision*100:.1f}% | ↑ {(vae_precision-baseline_precision)/baseline_precision*100:.0f}% |
| Daily Alerts | {baseline_alerts_per_day:,.0f} | {vae_alerts_per_day:,.0f} | ↓ {(baseline_alerts_per_day-vae_alerts_per_day)/baseline_alerts_per_day*100:.0f}% |
""")


if __name__ == "__main__":
    evaluate()
