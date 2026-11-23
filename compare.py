"""Ranking quality on held-out queries: hand-tuned text score vs pointwise vs LambdaMART."""
import numpy as np

from ltr.data import FEATURES, make_queries
from ltr.metrics import mean_ndcg, mean_reciprocal_rank
from ltr.models import train_lambdamart, train_pointwise

df = make_queries(3000)
qids = df["qid"].unique()
rng = np.random.default_rng(0)
rng.shuffle(qids)
tr_q, va_q, te_q = np.split(qids, [int(0.7 * len(qids)), int(0.8 * len(qids))])
train, valid, test = (df[df.qid.isin(q)].sort_values("qid", kind="stable") for q in (tr_q, va_q, te_q))
test = test.copy()
print(f"{len(qids):,} queries, {len(df):,} query-product rows; test on {len(te_q)} unseen queries\n")

test["random"] = rng.random(len(test))
test["text_only"] = test["text_match"] + 0.3 * test["title_match"]
lm = train_lambdamart(train, valid)
test["pointwise"] = train_pointwise(train).predict(test[FEATURES])
test["lambdamart"] = lm.predict(test[FEATURES])

print(f"{'ranker':28s} {'NDCG@5':>7s} {'NDCG@10':>8s} {'MRR (rel>=3)':>13s}")
for col, label in [("random", "random order"), ("text_only", "text match only"),
                   ("pointwise", "pointwise regression"), ("lambdamart", f"LambdaMART ({lm.best_iteration_} trees)")]:
    print(f"{label:28s} {mean_ndcg(test, col, 5):7.4f} {mean_ndcg(test, col, 10):8.4f} {mean_reciprocal_rank(test, col):13.4f}")

imp = sorted(zip(FEATURES, lm.booster_.feature_importance("gain")), key=lambda t: -t[1])
total = sum(v for _, v in imp)
print("\nLambdaMART feature importance (gain share):")
for f, v in imp:
    print(f"  {f:18s} {v / total:.3f}")
