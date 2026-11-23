"""Simulate click logs with position bias, and correct for it with inverse propensity weighting.

Users mostly look at the top results: P(examine rank r) = (1 / r) ** eta.
A click needs both: the user examined the result AND found it relevant.
So a relevant product shown at rank 30 rarely gets clicked, and a model trained on raw
clicks learns "whatever the old ranker put on top" instead of "what is relevant".

IPS fix: weight each click by 1 / P(examine its rank), so clicks at low positions count more.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CLICK_PROB = np.array([0.05, 0.15, 0.35, 0.6, 0.85])  # by relevance grade 0..4


def simulate_clicks(df: pd.DataFrame, logging_score: str, eta: float = 1.0, sessions: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    out["rank"] = out.groupby("qid")[logging_score].rank(ascending=False, method="first").astype(int)
    out["propensity"] = (1.0 / out["rank"]) ** eta
    rel = out["relevance"].to_numpy()
    clicks = np.zeros(len(out))
    for _ in range(sessions):  # each query is searched a few times
        examined = rng.random(len(out)) < out["propensity"].to_numpy()
        clicks += examined & (rng.random(len(out)) < CLICK_PROB[rel])
    out["clicks"] = clicks
    out["clicked"] = (clicks > 0).astype(int)
    return out


def ips_weights(clicked: pd.DataFrame, clip: float = 0.02) -> np.ndarray:
    """Clicked rows get 1 / propensity (clipped so one deep click can't dominate); others get 1."""
    w = np.where(clicked["clicked"] == 1, 1.0 / np.maximum(clicked["propensity"], clip), 1.0)
    return w / w.mean()
