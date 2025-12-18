"""
Sentinel VAE Training v2.0

GPU-accelerated training with:
- TensorBoard logging
- Early stopping
- Learning rate scheduling
- Model checkpointing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime
from vae_model import VAE, vae_loss


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Data
    input_dim = 28          # Number of features (auto-detected from data)
    
    # Architecture (sized for 28 features)
    hidden_dims = [256, 128]
    latent_dim = 32
    dropout = 0.2
    
    # Training
    batch_size = 2048
    learning_rate = 0.001
    epochs = 100
    beta = 1.0              # KL weight
    
    # Early stopping
    patience = 50
    min_delta = 0.0001
    
    # Paths
    model_path = 'vae_trained.pt'
    
    @classmethod
    def print_config(cls):
        print("\n📋 Training Configuration:")
        print(f"   Input dim: {cls.input_dim}")
        print(f"   Hidden dims: {cls.hidden_dims}")
        print(f"   Latent dim: {cls.latent_dim}")
        print(f"   Dropout: {cls.dropout}")
        print(f"   Batch size: {cls.batch_size}")
        print(f"   Learning rate: {cls.learning_rate}")
        print(f"   Max epochs: {cls.epochs}")
        print(f"   Early stopping patience: {cls.patience}")


# ============================================================================
# TRAINING
# ============================================================================

def train():
    print("=" * 60)
    print("SENTINEL VAE TRAINING v2.0")
    print("=" * 60)
    
    # Device selection
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"\n🚀 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("\n💻 Using CPU")
    
    Config.print_config()
    
    # ========================================
    # LOAD DATA
    # ========================================
    print("\n📥 Loading data...")
    X_train = np.load('X_train.npy')
    
    # Update input dim based on actual data
    Config.input_dim = X_train.shape[1]
    print(f"   Training samples: {X_train.shape[0]:,}")
    print(f"   Features: {X_train.shape[1]}")
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X_train)
    
    # Train/validation split (90/10)
    n_samples = len(X_tensor)
    n_val = int(n_samples * 0.1)
    indices = torch.randperm(n_samples)
    
    train_indices = indices[n_val:]
    val_indices = indices[:n_val]
    
    X_train_split = X_tensor[train_indices]
    X_val_split = X_tensor[val_indices]
    
    print(f"   Train split: {len(X_train_split):,}")
    print(f"   Validation split: {len(X_val_split):,}")
    
    # Data loaders
    train_loader = DataLoader(
        TensorDataset(X_train_split),
        batch_size=Config.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val_split),
        batch_size=Config.batch_size,
        shuffle=False
    )
    
    # ========================================
    # MODEL SETUP
    # ========================================
    print("\n🏗️  Building model...")
    model = VAE(
        input_dim=Config.input_dim,
        hidden_dims=Config.hidden_dims,
        latent_dim=Config.latent_dim,
        dropout=Config.dropout
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # TensorBoard
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f'runs/sentinel_vae_{timestamp}')
    
    # ========================================
    # TRAINING LOOP
    # ========================================
    print("\n🚀 Starting training...")
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(Config.epochs):
        # Training
        model.train()
        train_losses = []
        
        for batch in train_loader:
            x = batch[0].to(device)
            
            optimizer.zero_grad()
            reconstruction, mu, logvar = model(x)
            loss, recon_loss, kl_loss = vae_loss(reconstruction, x, mu, logvar, Config.beta)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        
        # Validation
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                reconstruction, mu, logvar = model(x)
                loss, _, _ = vae_loss(reconstruction, x, mu, logvar, Config.beta)
                val_losses.append(loss.item())
        
        avg_val_loss = np.mean(val_losses)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Logging
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Print progress
        if epoch % 10 == 0 or epoch == Config.epochs - 1:
            print(f"Epoch {epoch:3d} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping check
        if avg_val_loss < best_val_loss - Config.min_delta:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'config': {
                    'input_dim': Config.input_dim,
                    'hidden_dims': Config.hidden_dims,
                    'latent_dim': Config.latent_dim,
                    'dropout': Config.dropout,
                }
            }, Config.model_path)
        else:
            patience_counter += 1
            if patience_counter >= Config.patience:
                print(f"\n⏹️  Early stopping at epoch {epoch} (patience={Config.patience})")
                break
    
    writer.close()
    
    # ========================================
    # SUMMARY
    # ========================================
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"   Best validation loss: {best_val_loss:.4f}")
    print(f"   Model saved to: {Config.model_path}")
    print(f"   TensorBoard logs: runs/sentinel_vae_{timestamp}")
    print(f"\n   View with: tensorboard --logdir=runs")


if __name__ == "__main__":
    train()
