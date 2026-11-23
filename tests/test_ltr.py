import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

from ltr.clicks import ips_weights, simulate_clicks
from ltr.data import FEATURES, make_queries
from ltr.metrics import mean_ndcg, mean_reciprocal_rank, ndcg_at_k
from ltr.models import group_sizes, train_lambdamart


def test_ndcg_linear_matches_sklearn():
    rng = np.random.default_rng(1)
    for _ in range(20):
        rel, s = rng.integers(0, 5, 25), rng.random(25)
        assert np.isclose(ndcg_at_k(rel[np.argsort(-s)], 10, "linear"), ndcg_score([rel], [s], k=10))


def test_perfect_and_reversed_ranking():
    rel = np.array([4, 3, 2, 1, 0])
    assert ndcg_at_k(rel, 5) == 1.0
    assert ndcg_at_k(rel[::-1], 5) < 0.6


def test_mrr():
    df = pd.DataFrame({"qid": [0, 0, 0, 1, 1], "relevance": [0, 3, 0, 4, 0], "s": [3, 2, 1, 2, 1]})
    assert mean_reciprocal_rank(df, "s") == (1 / 2 + 1) / 2


def test_groups_are_contiguous_sizes():
    df = make_queries(20)
    assert group_sizes(df).sum() == len(df) and len(group_sizes(df)) == 20


def test_clicks_are_position_biased():
    df = make_queries(400)
    df["old"] = df["text_match"]
    logs = simulate_clicks(df, "old")
    top = logs[(logs.relevance >= 3) & (logs["rank"] <= 3)].clicked.mean()
    deep = logs[(logs.relevance >= 3) & (logs["rank"] > 15)].clicked.mean()
    assert top > 3 * deep


def test_ips_upweights_deep_clicks():
    df = make_queries(200)
    df["old"] = df["text_match"]
    logs = simulate_clicks(df, "old")
    w = ips_weights(logs)
    clicked = logs.clicked == 1
    assert w[clicked & (logs["rank"] > 10)].mean() > w[clicked & (logs["rank"] <= 3)].mean()


def test_lambdamart_beats_text_only():
    df = make_queries(600)
    train, valid, test = df[df.qid < 400], df[(df.qid >= 400) & (df.qid < 500)], df[df.qid >= 500].copy()
    test["lm"] = train_lambdamart(train, valid).predict(test[FEATURES])
    test["text"] = test["text_match"]
    assert mean_ndcg(test, "lm") > mean_ndcg(test, "text") + 0.15
