"""Retired RMA distillation entry point.

KUAFU deploys the causal PPO Actor directly.  Old distillation checkpoints use the
incompatible 157-dimensional RMA interface and are explicitly legacy-v0.
"""

if __name__ == "__main__":
    raise SystemExit("RMA distillation is retired; train and export the 140-dimensional PPO Actor directly.")
