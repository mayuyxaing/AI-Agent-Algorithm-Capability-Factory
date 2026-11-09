"""Synthetic, explicitly non-production customer churn benchmark."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

NUMERIC = ["tenure", "monthly_charges", "support_calls"]
CATEGORICAL = ["contract", "payment_method"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "churn"


def generate(path: Path, rows: int = 1200, seed: int = 42, missing_rate: float = .035,
             logit_shift: float = 0.0, interaction_strength: float = 0.0) -> None:
    if not 0 <= missing_rate <= .6:
        raise ValueError("missing_rate must be between 0 and 0.6")
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 73, rows)
    charges = rng.uniform(20, 120, rows)
    calls = rng.poisson(1.5, rows)
    contract = rng.choice(["monthly", "annual", "two_year"], rows, p=[.6, .25, .15])
    payment = rng.choice(["automatic", "manual"], rows)
    logits = (-1.8 - .035 * tenure + .018 * (charges - 50) + .6 * calls
              + 1.5 * (contract == "monthly") + .35 * (payment == "manual"))
    logits = logits + logit_shift + interaction_strength * (calls >= 3) * (contract == "monthly")
    churn = rng.binomial(1, 1 / (1 + np.exp(-logits)))
    frame = pd.DataFrame(dict(tenure=tenure.astype(float), monthly_charges=charges,
                              support_calls=calls.astype(float), contract=contract,
                              payment_method=payment, churn=churn))
    for col in FEATURES:
        frame.loc[rng.random(rows) < missing_rate, col] = np.nan
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def load_split(path: Path, seed: int = 42):
    frame = pd.read_csv(path)
    missing = set(FEATURES + [TARGET]) - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {sorted(missing)}")
    if frame[TARGET].isna().any() or set(frame[TARGET].unique()) != {0, 1}:
        raise ValueError("churn must contain both binary classes 0 and 1, without nulls")
    if len(frame) < 100 or frame[TARGET].value_counts().min() < 10:
        raise ValueError("At least 100 rows and 10 examples per class are required")
    train, rest = train_test_split(frame, test_size=.4, stratify=frame[TARGET], random_state=seed)
    valid, test = train_test_split(rest, test_size=.5, stratify=rest[TARGET], random_state=seed)
    return tuple(x.reset_index(drop=True) for x in (train, valid, test))
