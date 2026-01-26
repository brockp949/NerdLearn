# Curriculum RL Training Guide

This document describes how to train the Curriculum Reinforcement Learning (CRL) policy using the "Optimizing Spaced Interleaving with RL" framework.

## Overview

The system uses Offline Reinforcement Learning (Decision Transformer) to learn optimal scheduling policies from student interaction data. The pipeline consists of:

1.  **Data Generation/Extraction**: Creating trajectories from interaction logs.
2.  **Training**: Training a transformer model to predict optimal actions.
3.  **Inference**: Converting the model to a lightweight format (`DTLite`) for low-latency production use.

## Training Script

A dedicated training script is available at `apps/api/app/adaptive/offline_rl/train_crl_policy.py`.

### Usage

Run the script from the `apps/api` directory:

```bash
cd apps/api
python -m app.adaptive.offline_rl.train_crl_policy \
  --output_dir ../../models/crl_v1 \
  --num_users 1000 \
  --epochs 50 \
  --batch_size 64
```

### Parameters

- `--output_dir`: Directory to save trained model and config (default: `./models`)
- `--num_users`: Number of synthetic users to generate (default: 500)
- `--interactions`: Number of interactions per user (default: 100)
- `--concepts`: Number of unique concepts in the curriculum (default: 20)
- `--epochs`: Number of training epochs (default: 10)
- `--batch_size`: Batch size for training (default: 64)
- `--seed`: Random seed for reproducibility (default: 42)
- `--no_cuda`: Force CPU training even if GPU is available

## Output Artifacts

The script produces the following files in the output directory:

1.  `dt_model.pt`: Full PyTorch model checkpoint (for resuming training).
2.  `dt_lite_weights.npz`: Exported weights for inference.
3.  `dt_lite_config.json`: Configuration file for the inference engine.

## Production Deployment

To use the trained model in production:

1.  Copy `dt_lite_weights.npz` and `dt_lite_config.json` to your model storage location (e.g., `apps/api/models/`).
2.  Update `apps/api/app/config.py` or environment variables to point to these files:

```env
CRL_MODEL_PATH=apps/api/models/dt_lite_weights.npz
CRL_CONFIG_PATH=apps/api/models/dt_lite_config.json
```

3.  Restart the `orchestrator` service.

## Synthetic Data Generation

The current pipeline uses a synthetic data generator that simulates:
- **Forgetting Curves**: Exponential decay of memory strength.
- **Learning Rates**: Variable learning rates per student.
- **Spacing Effect**: Benefits from spaced repetition.

Future improvements will involve extracting real user logs from the `ReviewLog` database table.
