"""
Training Script for Curriculum RL Policy (Decision Transformer)

This script implements the training pipeline for the "Optimizing Spaced Interleaving with RL"
system. It performs the following steps:

1. Data Generation: Creates a synthetic dataset of student interactions simulating
   realistic forgetting curves and learning dynamics.
2. Training: Trains a Decision Transformer model on this offline dataset to learn
   optimal scheduling policies.
3. Export: Exports the trained model weights to a format compatible with DTLite
   for lightweight production inference.

Usage:
    python -m app.adaptive.offline_rl.train_crl_policy --output_dir ./models --epochs 50
"""

import os
import argparse
import logging
import torch
import numpy as np
from datetime import datetime
from pathlib import Path

from app.adaptive.offline_rl.decision_transformer import (
    DecisionTransformer,
    DecisionTransformerConfig,
    DecisionTransformerTrainer,
)
from app.adaptive.offline_rl.data_pipeline import DataPipeline, TrajectoryDataset
from app.adaptive.offline_rl.dt_lite import DTLite, DTLiteConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Train Curriculum RL Policy")
    parser.add_argument("--output_dir", type=str, default="./models", help="Directory to save models")
    parser.add_argument("--num_users", type=int, default=500, help="Number of synthetic users")
    parser.add_argument("--interactions", type=int, default=100, help="Interactions per user")
    parser.add_argument("--concepts", type=int, default=20, help="Number of concepts")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    
    args = parser.parse_args()
    
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Setup device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # --------------------------------------------------------------------------
    # 1. Data Generation
    # --------------------------------------------------------------------------
    logger.info("Step 1: Generating synthetic training data...")
    pipeline = DataPipeline()
    dataset = pipeline.create_synthetic_dataset(
        num_users=args.num_users,
        interactions_per_user=args.interactions,
        num_concepts=args.concepts,
        seed=args.seed
    )
    
    # Save dataset statistics
    stats = dataset.compute_statistics()
    logger.info(f"Dataset Stats: {stats}")
    
    # --------------------------------------------------------------------------
    # 2. Model Initialization
    # --------------------------------------------------------------------------
    logger.info("Step 2: Initializing Decision Transformer...")
    
    # Determine state dimension from data
    sample_traj = dataset.trajectories[0]
    state_dim = sample_traj.transitions[0].state.shape[0]
    
    config = DecisionTransformerConfig(
        state_dim=state_dim,
        action_dim=args.concepts,
        hidden_dim=128,
        num_layers=4,
        num_heads=4,
        max_length=20,  # Context length
        max_episode_length=args.interactions + 10,
        batch_size=args.batch_size,
        learning_rate=1e-4,
        weight_decay=1e-4,
        warmup_steps=1000,
        max_steps=args.epochs * (len(dataset) // args.batch_size),
    )
    
    model = DecisionTransformer(config)
    
    # --------------------------------------------------------------------------
    # 3. Training
    # --------------------------------------------------------------------------
    logger.info(f"Step 3: Training for {args.epochs} epochs...")
    trainer = DecisionTransformerTrainer(model, config, device=device)
    
    # Calculate steps
    steps_per_epoch = len(dataset) // args.batch_size
    total_steps = args.epochs * steps_per_epoch
    
    history = trainer.train(
        dataset,
        num_steps=total_steps,
        save_interval=steps_per_epoch,  # Save every epoch
        save_path=str(output_path / "dt_model.pt")
    )
    
    logger.info(f"Training complete. Final loss: {history['loss'][-1]:.4f}")
    
    # --------------------------------------------------------------------------
    # 4. Export for DT-Lite
    # --------------------------------------------------------------------------
    logger.info("Step 4: Exporting weights for DT-Lite...")
    
    weights_path = output_path / "dt_lite_weights.npz"
    config_path = output_path / "dt_lite_config.json"
    
    model.export_weights(str(weights_path))
    
    # Create DTLite config matching the trained model
    lite_config = DTLiteConfig(
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        context_length=config.max_length,
        temperature=config.temperature,
        top_k=config.top_k,
        target_return=config.target_return
    )
    lite_config.save(str(config_path))
    
    logger.info(f"Exported weights to {weights_path}")
    logger.info(f"Exported config to {config_path}")
    
    # --------------------------------------------------------------------------
    # 5. Verification
    # --------------------------------------------------------------------------
    logger.info("Step 5: Verifying DTLite loading...")
    
    try:
        dt_lite = DTLite.load(str(weights_path), lite_config)
        
        # Test inference
        mock_state = np.zeros(config.state_dim)
        action, probs = dt_lite.select_action(mock_state)
        
        logger.info(f"Verification successful. Selected action: {action}, Probs shape: {probs.shape}")
        
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        raise

if __name__ == "__main__":
    main()
