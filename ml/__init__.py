"""
Sentinel ML Module

Components:
- prepare_data.py: Feature engineering and data preparation
- vae_model.py: Variational Autoencoder architecture
- train_vae.py: GPU-accelerated training pipeline
- vae_alerter.py: Anomaly detection and evaluation
"""

from .vae_model import VAE, vae_loss
