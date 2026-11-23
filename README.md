# Learning to Rank: LambdaMART, and Learning from Biased Clicks

Ranks products for search queries with LightGBM's LambdaMART. It then tackles the harder, more realistic problem: learning from **click logs**, where users rarely look past the first few results.

## Data

3,000 synthetic product-search queries, each with 20–40 candidate products (~90k rows) and graded relevance from 0 to 4. Relevance comes from a hidden function mixing text match with things text match can't see: rating, review count, stock, brand, and **price, which counts against a product only for "budget" queries** (an interaction). Train, validation and test splits are made **by query**, so test queries are never seen in training.

## Part 1: rankers trained on relevance labels (600 test queries)

| Ranker | NDCG@5 | NDCG@10 | MRR (rel ≥ 3) |
|---|---|---|---|
| random order | 0.181 | 0.256 | 0.295 |
| text match only (the "old" search) | 0.611 | 0.680 | 0.769 |
| pointwise regression (LightGBM) | 0.905 | 0.926 | 0.982 |
| **LambdaMART** (early stopped at 255 trees) | **0.907** | **0.927** | 0.979 |

Learned rankers lift NDCG@10 from 0.68 to 0.93. **With clean, graded labels, LambdaMART and pointwise regression end up nearly tied**: when each item's grade is directly learnable, the pairwise objective has little extra to offer. LambdaMART's real advantage is in optimizing the top of the list when labels are relative within a query.

## Part 2: learning from click logs

Human relevance labels are expensive; clicks are free but biased. Clicks were simulated from the old text-match ranker:
- users examine rank *r* with probability 1/r
- they click examined results with probability rising with relevance

```
click logs: 62,830 impressions, 6,239 clicked
22% of highly relevant products sat below rank 10, where almost nobody looks
```

All models below are scored on the **true** relevance of unseen queries:

| Model | NDCG@10 |
|---|---|
| old ranker (produced the logs) | 0.680 |
| pointwise on raw clicks | 0.816 |
| LambdaMART on raw clicks | 0.813 |
| **pointwise on clicks + IPS** | **0.864** |
| *LambdaMART on true labels (upper bound)* | *0.927* |

**What position bias does:** a model trained on raw clicks learns *"whatever the old ranker put on top"*. It gives `text_match` (the old ranker's main signal) **61%** of its split gain, against **37%** in the true-label model.

**Inverse propensity weighting** counts each click as 1/P(seen at that rank), so a click at rank 20 counts 20× a click at rank 1. That brings `text_match` back down to **30%** and recovers **close to half the gap** between raw clicks and true labels.

## Implementation notes
- NDCG supports both gain conventions: `2^rel − 1` (LightGBM / LambdaMART) and linear `rel` (scikit-learn's `ndcg_score`). The linear version is tested against scikit-learn on random rankings.
- LightGBM needs each query's rows to be contiguous, plus a `group` array of sizes. `group_sizes()` builds it, and a test checks it.
- IPS weights are clipped at propensity 0.02 so a single deep click can't dominate training.

## How to run

```bash
pip install -r requirements.txt
PYTHONPATH=src python compare.py            # part 1
PYTHONPATH=src python clicks_experiment.py  # part 2
pytest
```

## Next steps
- Estimate propensities from the logs with randomized interventions (swap experiments) instead of assuming 1/r
- Add an unbiased LambdaMART variant that applies the IPS weights per pair
