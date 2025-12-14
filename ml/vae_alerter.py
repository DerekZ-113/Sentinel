"""
VAE-Based Anomaly Detection - Production Version

Evaluates trained VAE on full dataset and compares to baseline alerter.
"""

import torch
import numpy as np
from vae_model import VAE

# ============================================================================
# CONFIGURATION (must match training)
# ============================================================================

class Config:
    input_dim = 7
    hidden_dims = [64, 32]
    latent_dim = 8
    dropout = 0.2
    model_path = 'vae_trained.pt'

config = Config()

# ============================================================================
# DEVICE SETUP
# ============================================================================

def get_device():
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🎮 Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("🍎 Using Apple MPS GPU")
    else:
        device = torch.device('cpu')
        print("💻 Using CPU")
    return device

# ============================================================================
# LOAD MODEL AND DATA
# ============================================================================

def load_trained_model(config, device):
    """Load the trained VAE model."""
    model = VAE(
        input_dim=config.input_dim,
        hidden_dims=config.hidden_dims,
        latent_dim=config.latent_dim,
        dropout=config.dropout
    )
    model.load_state_dict(torch.load(config.model_path, weights_only=True, map_location=device))
    model = model.to(device)
    model.eval()
    
    print(f"✅ Loaded trained model from {config.model_path}")
    return model

def load_all_data():
    """Load preprocessed data (all records, including anomalies)."""
    X_all = np.load('X_all.npy')
    y_all = np.load('y_all.npy')
    
    print(f"✅ Loaded {len(X_all):,} records")
    print(f"   Normal: {(y_all == False).sum():,}")
    print(f"   Anomaly: {(y_all == True).sum():,}")
    
    return X_all, y_all

# ============================================================================
# CALCULATE RECONSTRUCTION ERROR
# ============================================================================

def calculate_reconstruction_errors(model, X_all, device, batch_size=4096):
    """
    Calculate reconstruction error for each sample.
    Uses batching for memory efficiency on large datasets.
    """
    print("\n🔄 Calculating reconstruction errors...")
    
    X_tensor = torch.tensor(X_all, dtype=torch.float32)
    errors = []
    
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i+batch_size].to(device)
            reconstruction, _, _ = model(batch)
            
            # Mean squared error per sample
            batch_errors = torch.mean((batch - reconstruction) ** 2, dim=1)
            errors.append(batch_errors.cpu())
            
            # Progress
            if (i // batch_size) % 10 == 0:
                print(f"   Processed {min(i+batch_size, len(X_tensor)):,} / {len(X_tensor):,}", end='\r')
    
    errors = torch.cat(errors).numpy()
    print(f"\n   Error range: {errors.min():.6f} to {errors.max():.6f}")
    print(f"   Mean error: {errors.mean():.6f}")
    print(f"   Std error: {errors.std():.6f}")
    
    return errors

# ============================================================================
# THRESHOLD AND DETECTION
# ============================================================================

def find_optimal_threshold(errors, y_true):
    """Find threshold that maximizes detection while minimizing false positives."""
    print("\n🎯 Finding optimal threshold...")
    
    normal_errors = errors[y_true == False]
    anomaly_errors = errors[y_true == True]
    
    print(f"\n   Normal samples error:  mean={normal_errors.mean():.6f}, std={normal_errors.std():.6f}")
    print(f"   Anomaly samples error: mean={anomaly_errors.mean():.6f}, std={anomaly_errors.std():.6f}")
    print(f"   Separation ratio: {anomaly_errors.mean() / normal_errors.mean():.2f}x")
    
    results = []
    
    for percentile in [90, 95, 97, 99]:
        threshold = np.percentile(normal_errors, percentile)
        predictions = errors > threshold
        
        tp = ((predictions == True) & (y_true == True)).sum()
        fp = ((predictions == True) & (y_true == False)).sum()
        tn = ((predictions == False) & (y_true == False)).sum()
        fn = ((predictions == False) & (y_true == True)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        results.append({
            'percentile': percentile,
            'threshold': threshold,
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'fp_rate': fp_rate
        })
        
        print(f"\n   {percentile}th percentile (threshold={threshold:.6f}):")
        print(f"      Precision={precision:.1%}, Recall={recall:.1%}, F1={f1:.1%}")
    
    # Select best F1 score with recall > 30%
    valid_results = [r for r in results if r['recall'] > 0.3]
    if valid_results:
        best = max(valid_results, key=lambda x: x['f1'])
    else:
        best = results[-1]
    
    print(f"\n   ✅ Selected: {best['percentile']}th percentile (F1={best['f1']:.1%})")
    
    return best['threshold'], results

def evaluate_vae_alerter(errors, y_true, threshold):
    """Final evaluation with chosen threshold."""
    predictions = errors > threshold
    
    tp = ((predictions == True) & (y_true == True)).sum()
    fp = ((predictions == True) & (y_true == False)).sum()
    tn = ((predictions == False) & (y_true == False)).sum()
    fn = ((predictions == False) & (y_true == True)).sum()
    
    total_alerts = tp + fp
    precision = tp / total_alerts if total_alerts > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    fp_rate = fp / total_alerts if total_alerts > 0 else 0
    
    return {
        'total_alerts': int(total_alerts),
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'fp_rate': fp_rate
    }

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SENTINEL VAE ANOMALY DETECTION - PRODUCTION VERSION")
    print("=" * 60)
    
    # Setup
    device = get_device()
    model = load_trained_model(config, device)
    X_all, y_all = load_all_data()
    
    # Calculate errors
    errors = calculate_reconstruction_errors(model, X_all, device)
    
    # Find threshold
    threshold, all_results = find_optimal_threshold(errors, y_all)
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("📊 VAE ALERTER RESULTS")
    print("=" * 60)
    
    metrics = evaluate_vae_alerter(errors, y_all, threshold)
    
    print(f"Total alerts fired:    {metrics['total_alerts']:,}")
    print(f"True positives:        {metrics['true_positives']:,}")
    print(f"False positives:       {metrics['false_positives']:,}")
    print(f"")
    print(f"False positive rate:   {metrics['fp_rate']:.1%}")
    print(f"Precision:             {metrics['precision']:.1%}")
    print(f"Recall:                {metrics['recall']:.1%}")
    print(f"F1 Score:              {metrics['f1']:.1%}")
    
    # Compare to baseline
    print("\n" + "=" * 60)
    print("📈 COMPARISON TO BASELINE")
    print("=" * 60)
    print(f"                      Baseline    VAE        Change")
    print(f"                      --------    ----       ------")
    print(f"False Positive Rate:    34.5%     {metrics['fp_rate']:.1%}      ", end="")
    
    baseline_fp_rate = 0.345
    fp_change = (baseline_fp_rate - metrics['fp_rate']) / baseline_fp_rate * 100
    print(f"↓ {fp_change:.0f}%")
    
    print(f"Precision:              65.5%     {metrics['precision']:.1%}      ", end="")
    precision_change = (metrics['precision'] - 0.655) / 0.655 * 100
    print(f"↑ {precision_change:.0f}%")
    
    print("\n" + "=" * 60)
    print(f"🎯 SUMMARY: {fp_change:.0f}% reduction in false positives")
    print("=" * 60)
