"""
evaluate.py — Precision@5, Recall@5, NDCG@5 for the Kiva partner recommender.

Ground truth: partners that actually funded >= 1 loan for a given (sector, country)
query, derived from kiva_loans.csv joined with loan_themes_by_region.csv.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OrdinalEncoder


# ── Load model ─────────────────────────────────────────────────────────────────
with open("model/similarity_matrix.pkl", "rb") as f:
    model = pickle.load(f)

partners = model["partners"]    # rows from loan_themes_by_region (after filtering)
encoder  = model["encoder"]    # fitted OneHotEncoder

# ── Build ground truth ─────────────────────────────────────────────────────────
# Merge actual loans with partner names via partner_id / Partner ID
loans   = pd.read_csv("data/kiva_loans.csv")
partner_map = (partners[["Partner ID", "Field Partner Name"]]
               .drop_duplicates()
               .rename(columns={"Partner ID": "partner_id"}))

loans_merged = loans.merge(partner_map, on="partner_id", how="inner")

# For each (sector, country) query: set of partner names that actually operated there
ground_truth = (
    loans_merged
    .groupby(["sector", "country"])["Field Partner Name"]
    .apply(set)
    .reset_index()
    .rename(columns={"Field Partner Name": "relevant_partners"})
)

# Keep only queries where at least 1 relevant partner exists in our catalog
ground_truth = ground_truth[ground_truth["relevant_partners"].apply(len) > 0]

print(f"Evaluation queries: {len(ground_truth):,}  "
      f"(unique sector×country pairs with known-relevant partners)")


# ── Recommender function ───────────────────────────────────────────────────────
# Deduplicated partner catalog (same logic as app.py)
catalog = (partners
           .sort_values("amount", ascending=False)
           .drop_duplicates(subset=["Field Partner Name"])
           .reset_index(drop=True))

catalog_vecs = encoder.transform(
    catalog[["sector", "country", "Loan Theme Type"]].fillna("Unknown")
)


def recommend(sector: str, country: str, k: int = 5) -> list[str]:
    """Return top-k Field Partner Names for the given profile."""
    borrower_vec = encoder.transform(
        pd.DataFrame([[sector, country, sector]],
                     columns=["sector", "country", "Loan Theme Type"])
    )
    scores = cosine_similarity(borrower_vec, catalog_vecs)[0]

    filtered = catalog.copy()
    filtered["_score"] = scores
    filtered = filtered[filtered["country"] == country]
    filtered = filtered.sort_values(["_score", "amount"], ascending=[False, False])
    return filtered["Field Partner Name"].head(k).tolist()


# ── Metric helpers ─────────────────────────────────────────────────────────────
def precision_at_k(recommended: list, relevant: set, k: int = 5) -> float:
    hits = sum(1 for p in recommended[:k] if p in relevant)
    return hits / k


def recall_at_k(recommended: list, relevant: set, k: int = 5) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for p in recommended[:k] if p in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: list, relevant: set, k: int = 5) -> float:
    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, p in enumerate(recommended[:k])
        if p in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


# ── Evaluate ───────────────────────────────────────────────────────────────────
K = 5
rows = []

for _, qrow in ground_truth.iterrows():
    sec, cty, relevant = qrow["sector"], qrow["country"], qrow["relevant_partners"]

    # Skip if no partners in our catalog operate in this country
    if catalog[catalog["country"] == cty].empty:
        continue

    recs = recommend(sec, cty, k=K)
    rows.append({
        "sector":    sec,
        "country":   cty,
        "relevant":  len(relevant),
        "P@5":       precision_at_k(recs, relevant, K),
        "R@5":       recall_at_k(recs, relevant, K),
        "NDCG@5":    ndcg_at_k(recs, relevant, K),
    })

results_df = pd.DataFrame(rows)

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n── Overall metrics ──────────────────────────────────────")
print(f"  Queries evaluated : {len(results_df):,}")
print(f"  Precision@5       : {results_df['P@5'].mean():.4f}")
print(f"  Recall@5          : {results_df['R@5'].mean():.4f}")
print(f"  NDCG@5            : {results_df['NDCG@5'].mean():.4f}")

print("\n── NDCG@5 by sector ─────────────────────────────────────")
by_sector = (results_df.groupby("sector")["NDCG@5"]
             .agg(["mean", "count"])
             .rename(columns={"mean": "NDCG@5", "count": "queries"})
             .sort_values("NDCG@5", ascending=False))
print(by_sector.to_string())

print("\n── NDCG@5 by country (top 15) ───────────────────────────")
by_country = (results_df.groupby("country")["NDCG@5"]
              .agg(["mean", "count"])
              .rename(columns={"mean": "NDCG@5", "count": "queries"})
              .sort_values("NDCG@5", ascending=False)
              .head(15))
print(by_country.to_string())

# Save full results for inspection
os.makedirs("results", exist_ok=True)
results_df.to_csv("results/eval_results.csv", index=False)
print("\nFull results saved to results/eval_results.csv")
