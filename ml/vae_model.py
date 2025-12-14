"""
VAE (Variational Autoencoder) for Anomaly Detection - Production Version

UPGRADES FROM BASIC VERSION:
============================
1. Deeper architecture (3 layers instead of 1)
2. Dropout for regularization (prevents overfitting)
3. Batch normalization (stabilizes training)
4. Configurable architecture via parameters
5. GPU-ready (CUDA/MPS compatible)

ARCHITECTURE:
=============
Input (7) → 64 → 32 → Latent (8) → 32 → 64 → Output (7)
                ↓
         With dropout + batch norm between layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# VAE MODEL - PRODUCTION VERSION
# ============================================================================

class VAE(nn.Module):
    """
    Variational Autoencoder with deeper architecture and regularization.
    
    NEW PYTORCH CONCEPTS:
    ====================
    
    nn.Dropout(p=0.2):
        Randomly zeros 20% of inputs during training.
        Prevents overfitting by forcing redundancy.
        Automatically disabled during eval() mode.
    
    nn.BatchNorm1d(features):
        Normalizes layer outputs to mean=0, std=1.
        Stabilizes and accelerates training.
        Has learnable scale/shift parameters.
    
    nn.Sequential(*layers):
        Chains multiple layers into one callable.
        Cleaner than calling each layer manually.
    """
    
    def __init__(self, input_dim=7, hidden_dims=[64, 32], latent_dim=8, dropout=0.2):
        """
        Args:
            input_dim: Number of input features (7 for our data)
            hidden_dims: List of hidden layer sizes [64, 32]
            latent_dim: Size of latent space (8)
            dropout: Dropout probability (0.2 = 20%)
        """
        super(VAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # ========================
        # ENCODER
        # ========================
        # Build encoder layers dynamically
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),  # Normalize activations
                nn.ReLU(),
                nn.Dropout(dropout)           # Regularization
            ])
            prev_dim = hidden_dim
        
        # nn.Sequential chains all layers together
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space projections (mu and logvar)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # ========================
        # DECODER
        # ========================
        # Mirror the encoder architecture
        decoder_layers = []
        prev_dim = latent_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Output projection
        self.fc_out = nn.Linear(hidden_dims[0], input_dim)
    
    def encode(self, x):
        """Encode input to latent distribution parameters."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Sample from latent distribution using reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent representation to reconstruction."""
        h = self.decoder(z)
        return torch.sigmoid(self.fc_out(h))
    
    def forward(self, x):
        """Full forward pass: encode → sample → decode"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar

# ============================================================================
# LOSS FUNCTION
# ============================================================================

def vae_loss(reconstruction, x, mu, logvar, kl_weight=1.0):
    """
    VAE Loss = Reconstruction Loss + KL Divergence
    
    NEW: kl_weight parameter allows tuning the balance.
    Lower kl_weight = focus more on reconstruction accuracy.
    """
    # Reconstruction loss (binary cross entropy)
    recon_loss = F.binary_cross_entropy(reconstruction, x, reduction='sum')
    
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Weighted total
    total_loss = recon_loss + kl_weight * kl_loss
    
    return total_loss, recon_loss, kl_loss

# ============================================================================
# MODEL SUMMARY UTILITY
# ============================================================================

def count_parameters(model):
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_summary(model):
    """Print model architecture summary."""
    print("\n" + "=" * 60)
    print("MODEL ARCHITECTURE")
    print("=" * 60)
    print(model)
    print("-" * 60)
    print(f"Total trainable parameters: {count_parameters(model):,}")
    print("=" * 60)

# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("VAE MODEL TEST - PRODUCTION VERSION")
    print("=" * 60)
    
    # Detect device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🎮 Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("🍎 Using Apple MPS GPU")
    else:
        device = torch.device('cpu')
        print("💻 Using CPU")
    
    # Create model
    model = VAE(
        input_dim=7,
        hidden_dims=[64, 32],
        latent_dim=8,
        dropout=0.2
    ).to(device)
    
    model_summary(model)
    
    # Test forward pass
    print("\n🧪 Testing forward pass...")
    dummy_input = torch.rand(32, 7).to(device)  # Batch of 32
    
    model.eval()  # Set to eval mode (disables dropout)
    with torch.no_grad():
        reconstruction, mu, logvar = model(dummy_input)
    
    print(f"   Input shape:  {dummy_input.shape}")
    print(f"   Output shape: {reconstruction.shape}")
    print(f"   Latent shape: {mu.shape}")
    
    # Test loss
    loss, recon, kl = vae_loss(reconstruction, dummy_input, mu, logvar)
    print(f"\n📉 Loss: {loss.item():.2f} (Recon: {recon.item():.2f}, KL: {kl.item():.2f})")
    
    print("\n✅ Model test passed!")
