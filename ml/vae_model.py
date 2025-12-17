"""
Sentinel VAE Model v2.0

Variational Autoencoder for notification triage.
Learns patterns of false positives (no intervention needed).
High reconstruction error = likely needs real intervention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    """
    Variational Autoencoder for notification triage.
    
    Architecture: Input → Encoder → Latent (mu, logvar) → Decoder → Output
    
    The model learns to reconstruct "false positive" patterns.
    Real interventions have high reconstruction error because
    the model hasn't learned those patterns.
    """
    
    def __init__(self, input_dim=18, hidden_dims=[128, 64], latent_dim=16, dropout=0.2):
        super(VAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # ========================================
        # ENCODER
        # ========================================
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space (mean and log variance)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # ========================================
        # DECODER
        # ========================================
        decoder_layers = []
        decoder_dims = [latent_dim] + hidden_dims[::-1]  # Reverse hidden dims
        
        for i in range(len(decoder_dims) - 1):
            decoder_layers.extend([
                nn.Linear(decoder_dims[i], decoder_dims[i+1]),
                nn.BatchNorm1d(decoder_dims[i+1]),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
        
        # Final output layer (no activation - will apply sigmoid in loss)
        decoder_layers.append(nn.Linear(decoder_dims[-1], input_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Print architecture
        total_params = sum(p.numel() for p in self.parameters())
        print(f"VAE Architecture: {input_dim} → {hidden_dims} → {latent_dim} → {hidden_dims[::-1]} → {input_dim}")
        print(f"Total parameters: {total_params:,}")

    def encode(self, x):
        """Encode input to latent space parameters"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = mu + std * epsilon
        Allows backprop through sampling.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Decode latent vector to reconstruction"""
        return self.decoder(z)

    def forward(self, x):
        """Full forward pass: encode → sample → decode"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar

    def get_reconstruction_error(self, x):
        """
        Get per-sample reconstruction error for anomaly detection.
        Higher error = less like training data = more likely real intervention.
        """
        self.eval()
        with torch.no_grad():
            reconstruction, _, _ = self.forward(x)
            # MSE per sample
            error = torch.mean((x - reconstruction) ** 2, dim=1)
        return error


def vae_loss(reconstruction, x, mu, logvar, beta=1.0):
    """
    VAE loss = Reconstruction loss + KL divergence
    
    Args:
        reconstruction: Decoder output
        x: Original input
        mu: Latent mean
        logvar: Latent log variance
        beta: Weight for KL term (beta-VAE)
    
    Returns:
        total_loss, reconstruction_loss, kl_loss
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(reconstruction, x, reduction='mean')
    
    # KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss


# ============================================================================
# QUICK TEST
# ============================================================================

if __name__ == "__main__":
    print("Testing VAE model...")
    
    # Create model
    model = VAE(input_dim=18, hidden_dims=[128, 64], latent_dim=16)
    
    # Test forward pass
    x = torch.randn(32, 18)  # Batch of 32, 18 features
    reconstruction, mu, logvar = model(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Reconstruction shape: {reconstruction.shape}")
    print(f"Mu shape: {mu.shape}")
    print(f"Logvar shape: {logvar.shape}")
    
    # Test loss
    loss, recon, kl = vae_loss(reconstruction, x, mu, logvar)
    print(f"\nLoss: {loss.item():.4f} (Recon: {recon.item():.4f}, KL: {kl.item():.4f})")
    
    # Test reconstruction error
    error = model.get_reconstruction_error(x)
    print(f"Reconstruction error shape: {error.shape}")
    print(f"Mean error: {error.mean().item():.4f}")
    
    print("\n✅ Model test passed!")
