"""Three ways to learn a ranking from the same features.

pointwise  - regress the relevance grade of each product on its own (LGBMRegressor)
LambdaMART - gradient-boosted trees whose gradients are pairwise swaps weighted by how much
             each swap would change NDCG, so it focuses on getting the top of the list right
"""
from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

from .data import FEATURES

warnings.filterwarnings("ignore", message=".*eval_set.*")  # lightgbm/sklearn version noise


def group_sizes(df: pd.DataFrame) -> np.ndarray:
    # LightGBM needs rows of a query to be contiguous and the size of each group
    return df.groupby("qid", sort=False).size().to_numpy()


def train_pointwise(train: pd.DataFrame, seed=0):
    m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31, verbose=-1, random_state=seed)
    return m.fit(train[FEATURES], train["relevance"])


def train_lambdamart(train: pd.DataFrame, valid: pd.DataFrame, seed=0, weight=None, label="relevance"):
    m = lgb.LGBMRanker(objective="lambdarank", n_estimators=1000, learning_rate=0.05, num_leaves=31,
                       lambdarank_truncation_level=20, verbose=-1, random_state=seed)
    m.fit(train[FEATURES], train[label], group=group_sizes(train), sample_weight=weight,
          eval_set=[(valid[FEATURES], valid["relevance"])], eval_group=[group_sizes(valid)],
          eval_at=[10], callbacks=[lgb.early_stopping(50, verbose=False)])
    return m
