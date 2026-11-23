"""Ranking metrics computed per query, then averaged over queries."""
from __future__ import annotations

import numpy as np
import pandas as pd


def dcg(rels: np.ndarray, k: int, gain: str = "exponential") -> float:
    """gain="exponential" uses 2^rel - 1 (LightGBM / LambdaMART convention, rewards top grades more);
    gain="linear" uses rel as-is (what sklearn.metrics.ndcg_score does)."""
    rels = rels[:k].astype(float)
    g = 2.0**rels - 1 if gain == "exponential" else rels
    return float((g / np.log2(np.arange(2, len(rels) + 2))).sum())


def ndcg_at_k(rels_in_ranked_order: np.ndarray, k: int, gain: str = "exponential") -> float:
    ideal = dcg(np.sort(rels_in_ranked_order)[::-1], k, gain)
    return dcg(rels_in_ranked_order, k, gain) / ideal if ideal > 0 else 0.0


def mean_ndcg(df: pd.DataFrame, score_col: str, k: int = 10, rel_col: str = "relevance") -> float:
    vals = []
    for _, g in df.groupby("qid", sort=False):
        ranked = g.sort_values(score_col, ascending=False, kind="stable")[rel_col].to_numpy()
        vals.append(ndcg_at_k(ranked, k))
    return float(np.mean(vals))


def mean_reciprocal_rank(df: pd.DataFrame, score_col: str, rel_col: str = "relevance", threshold: int = 3) -> float:
    """1 / position of the first highly relevant result (relevance >= threshold), averaged over queries."""
    vals = []
    for _, g in df.groupby("qid", sort=False):
        ranked = g.sort_values(score_col, ascending=False, kind="stable")[rel_col].to_numpy()
        hits = np.flatnonzero(ranked >= threshold)
        vals.append(1.0 / (hits[0] + 1) if len(hits) else 0.0)
    return float(np.mean(vals))
