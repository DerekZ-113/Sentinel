"""
VAE Training Script - Production Version

UPGRADES:
=========
1. CUDA GPU support (auto-detects NVIDIA GPU)
2. TensorBoard logging for visualization
3. Early stopping to prevent overfitting
4. Larger batch sizes for GPU efficiency
5. Learning rate scheduling
6. Model checkpointing (save best model)
7. Training/validation split
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
import os
from datetime import datetime

from vae_model import VAE, vae_loss, model_summary

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """
    All hyperparameters in one place.
    Makes experiments reproducible and easy to tune.
    """
    # Data
    batch_size = 2048        # Large batch for GPU efficiency
    validation_split = 0.1   # 10% for validation
    
    # Model architecture
    input_dim = 7
    hidden_dims = [64, 32]   # Deeper: 7 → 64 → 32 → 8 → 32 → 64 → 7
    latent_dim = 8
    dropout = 0.2
    
    # Training
    epochs = 100
    learning_rate = 0.001
    weight_decay = 1e-5      # L2 regularization
    kl_weight = 1.0          # Weight for KL loss
    
    # Early stopping
    patience = 10            # Stop if no improvement for 10 epochs
    min_delta = 0.001        # Minimum improvement to count
    
    # Paths
    model_save_path = 'vae_trained.pt'
    tensorboard_dir = 'runs'

config = Config()

# ============================================================================
# DEVICE SETUP
# ============================================================================

def get_device():
    """
    Auto-detect best available device.
    Priority: CUDA (NVIDIA) > MPS (Apple) > CPU
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🎮 Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("🍎 Using Apple MPS GPU")
    else:
        device = torch.device('cpu')
        print("💻 Using CPU")
    return device

# ============================================================================
# DATA LOADING
# ============================================================================

def load_training_data(config):
    """Load and prepare data with train/validation split."""
    print("\n📥 Loading preprocessed data...")
    
    X_train = np.load('X_train.npy')
    print(f"   Total samples: {len(X_train):,}")
    
    # Convert to tensor
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    dataset = TensorDataset(X_tensor)
    
    # Train/validation split
    val_size = int(len(dataset) * config.validation_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"   Training samples: {train_size:,}")
    print(f"   Validation samples: {val_size:,}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,        # Set to 4 on Linux for parallel loading
        pin_memory=True       # Faster GPU transfer
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=True
    )
    
    print(f"   Batch size: {config.batch_size}")
    print(f"   Training batches: {len(train_loader)}")
    print(f"   Validation batches: {len(val_loader)}")
    
    return train_loader, val_loader

# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_epoch(model, train_loader, optimizer, device, config):
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    
    for batch in train_loader:
        x = batch[0].to(device)
        
        optimizer.zero_grad()
        reconstruction, mu, logvar = model(x)
        loss, recon_loss, kl_loss = vae_loss(
            reconstruction, x, mu, logvar, config.kl_weight
        )
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
    
    n = len(train_loader)
    return total_loss/n, total_recon/n, total_kl/n

def validate(model, val_loader, device, config):
    """Validate model. Returns average loss."""
    model.eval()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    
    with torch.no_grad():
        for batch in val_loader:
            x = batch[0].to(device)
            reconstruction, mu, logvar = model(x)
            loss, recon_loss, kl_loss = vae_loss(
                reconstruction, x, mu, logvar, config.kl_weight
            )
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
    
    n = len(val_loader)
    return total_loss/n, total_recon/n, total_kl/n

def train_vae(model, train_loader, val_loader, config, device):
    """
    Full training loop with:
    - TensorBoard logging
    - Early stopping
    - Model checkpointing
    - Learning rate scheduling
    """
    model = model.to(device)
    
    # Optimizer with weight decay (L2 regularization)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Learning rate scheduler (reduce LR when loss plateaus)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # TensorBoard writer
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f'{config.tensorboard_dir}/vae_{timestamp}')
    
    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"\n🚀 Starting training...")
    print(f"   Device: {device}")
    print(f"   Epochs: {config.epochs}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Learning rate: {config.learning_rate}")
    print(f"   Early stopping patience: {config.patience}")
    print("-" * 60)
    
    start_time = time.time()
    
    for epoch in range(config.epochs):
        # Train
        train_loss, train_recon, train_kl = train_epoch(
            model, train_loader, optimizer, device, config
        )
        
        # Validate
        val_loss, val_recon, val_kl = validate(
            model, val_loader, device, config
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Log to TensorBoard
        writer.add_scalars('Loss', {
            'train': train_loss,
            'val': val_loss
        }, epoch)
        writer.add_scalars('Reconstruction', {
            'train': train_recon,
            'val': val_recon
        }, epoch)
        writer.add_scalars('KL', {
            'train': train_kl,
            'val': val_kl
        }, epoch)
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)
        
        # Early stopping check
        if val_loss < best_val_loss - config.min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), config.model_save_path)
        else:
            patience_counter += 1
        
        # Print progress
        elapsed = time.time() - start_time
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{config.epochs} | "
                  f"Train: {train_loss:8.1f} | "
                  f"Val: {val_loss:8.1f} | "
                  f"Best: {best_val_loss:8.1f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
                  f"Time: {elapsed:.1f}s")
        
        # Early stopping
        if patience_counter >= config.patience:
            print(f"\n⏹️  Early stopping at epoch {epoch+1} (no improvement for {config.patience} epochs)")
            break
    
    writer.close()
    total_time = time.time() - start_time
    
    print("-" * 60)
    print(f"✅ Training complete in {total_time:.1f} seconds")
    print(f"💾 Best model saved to {config.model_save_path}")
    print(f"📊 TensorBoard logs: {config.tensorboard_dir}/vae_{timestamp}")
    print(f"   Run: tensorboard --logdir={config.tensorboard_dir}")
    
    # Load best model before returning
    model.load_state_dict(torch.load(config.model_save_path, weights_only=True))
    
    return model

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SENTINEL VAE TRAINING - PRODUCTION VERSION")
    print("=" * 60)
    
    # Setup
    device = get_device()
    train_loader, val_loader = load_training_data(config)
    
    # Create model
    model = VAE(
        input_dim=config.input_dim,
        hidden_dims=config.hidden_dims,
        latent_dim=config.latent_dim,
        dropout=config.dropout
    )
    model_summary(model)
    
    # Train
    model = train_vae(model, train_loader, val_loader, config, device)
    
    print("\n✅ Training pipeline complete!")
    print("   Next: Run vae_alerter.py to evaluate anomaly detection")
