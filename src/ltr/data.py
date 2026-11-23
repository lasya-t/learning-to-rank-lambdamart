"""Synthetic product-search logs: queries, candidate products, graded relevance.

Each query returns ~40 candidates. Relevance (0-4) comes from a hidden function that
mixes text match with things text match can't see: rating, popularity, price fit and
stock, with an interaction (cheap items only help for "budget" queries).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ["text_match", "title_match", "rating", "log_reviews", "price_z", "in_stock",
            "days_since_launch", "is_budget_query", "brand_match"]


def make_queries(n_queries: int = 2000, docs_per_query: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for q in range(n_queries):
        n = rng.integers(docs_per_query // 2, docs_per_query + 1)
        budget = rng.random() < 0.3
        text = rng.beta(2, 3, n)
        title = np.clip(text + rng.normal(0, 0.15, n), 0, 1)
        rating = np.clip(rng.normal(4.0, 0.6, n), 1, 5)
        reviews = rng.gamma(1.5, 400, n)
        price_z = rng.normal(0, 1, n)
        stock = rng.random(n) < 0.85
        age = rng.integers(0, 900, n)
        brand = rng.random(n) < 0.2
        score = (
            3.0 * text + 1.0 * title + 0.8 * (rating - 4) + 0.25 * np.log1p(reviews)
            - 1.2 * price_z * budget + 0.2 * price_z * (not budget)
            + 1.0 * stock + 0.6 * brand - 0.0006 * age + rng.normal(0, 0.35, n)
        )
        rel = np.digitize(score, np.quantile(score, [0.55, 0.75, 0.88, 0.96]))  # 0..4, few highly relevant
        rows.append(pd.DataFrame({
            "qid": q, "text_match": text, "title_match": title, "rating": rating,
            "log_reviews": np.log1p(reviews), "price_z": price_z, "in_stock": stock.astype(int),
            "days_since_launch": age, "is_budget_query": int(budget), "brand_match": brand.astype(int),
            "relevance": rel,
        }))
    return pd.concat(rows, ignore_index=True)
