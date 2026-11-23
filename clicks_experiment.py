"""Train on click logs instead of human labels. Does position bias hurt, and does IPS fix it?

The click logs come from the old 'text match only' ranker. Every model is evaluated on the
TRUE relevance grades of unseen queries, which the click-trained models never saw.
"""
import lightgbm as lgb
import numpy as np

from ltr.clicks import ips_weights, simulate_clicks
from ltr.data import FEATURES, make_queries
from ltr.metrics import mean_ndcg
from ltr.models import train_lambdamart

df = make_queries(3000)
df["old_ranker"] = df["text_match"] + 0.3 * df["title_match"]
qids = df["qid"].unique()
rng = np.random.default_rng(0)
rng.shuffle(qids)
tr_q, va_q, te_q = np.split(qids, [int(0.7 * len(qids)), int(0.8 * len(qids))])
logs = simulate_clicks(df[df.qid.isin(tr_q)].sort_values("qid", kind="stable"), "old_ranker", eta=1.0)
valid = df[df.qid.isin(va_q)].sort_values("qid", kind="stable")
test = df[df.qid.isin(te_q)].copy()
print(f"click logs: {len(logs):,} impressions, {int(logs.clicked.sum()):,} clicked; "
      f"{(logs[logs.relevance >= 3]['rank'] > 10).mean():.0%} of highly relevant products sat below rank 10\n")


def pointwise(label, weight=None):
    m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, verbose=-1, random_state=0)
    return m.fit(logs[FEATURES], logs[label], sample_weight=weight)


models = {
    "old ranker (logged)": None,
    "pointwise on raw clicks": pointwise("clicked"),
    "pointwise on clicks + IPS": pointwise("clicked", ips_weights(logs)),
    "LambdaMART on raw clicks": train_lambdamart(logs, valid, label="clicked"),
    "LambdaMART on true labels (upper bound)": train_lambdamart(df[df.qid.isin(tr_q)].sort_values("qid", kind="stable"), valid),
}
print(f"{'model':42s} {'NDCG@10 (true relevance)':>26s}")
for name, m in models.items():
    test["s"] = test["old_ranker"] if m is None else m.predict(test[FEATURES])
    print(f"{name:42s} {mean_ndcg(test, 's', 10):26.4f}")

# what did the raw-click model learn? it over-trusts the feature the old ranker used
raw, ips = models["pointwise on raw clicks"], models["pointwise on clicks + IPS"]
share = lambda m: m.booster_.feature_importance("gain")[FEATURES.index("text_match")] / m.booster_.feature_importance("gain").sum()  # noqa: E731
print(f"\ntext_match share of gain: raw clicks {share(raw):.2f}, with IPS {share(ips):.2f}, "
      f"true-label LambdaMART {share(models['LambdaMART on true labels (upper bound)']):.2f}")
