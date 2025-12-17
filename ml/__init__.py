"""
Sentinel ML Module

Context-aware notification triage for autonomous vehicle fleets.

Components:
- prepare_data.py: Feature engineering pipeline
- vae_model.py: Variational Autoencoder architecture
- train_vae.py: GPU-accelerated training
- vae_alerter.py: Evaluation and results
"""

from .vae_model import VAE, vae_loss
