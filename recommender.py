"""
Kiva Partner Recommender — core recommendation logic.

=======================================================================
STEP 1 — Formal Recommendation Task Definition
=======================================================================
Let:
  U = set of borrowers (users)
  P = set of Kiva field partners (items)
  q ∈ Q = a borrower query: (country, sector, loan_amount, term_months, borrower_type)
  H(u) = loan history of returning borrower u (empty set for cold-start users)
  s(p, q) = compatibility score between partner p and query q
  r(p, q) = post-bias-correction re-ranked score

Objective:
  Given q (and optionally H(u)), return an ordered list R = [p1, p2, p3, p4, p5]
  such that:
    (1) r(pi, q) ≥ r(pj, q) for i < j          [ranked by corrected score]
    (2) at least 1 partner is outside top-volume quartile   [diversity constraint]
    (3) similarity threshold: s(pi, q) ≥ 0.3    [no unsuitable matches]
    (4) gender is not a feature in s(·) or r(·) [fairness-by-design]

=======================================================================
STEP 2 — Known Problems and Design Responses
=======================================================================
| Problem                  | Where it appears             | Design decision                                      |
|--------------------------|------------------------------|------------------------------------------------------|
| Cold-start (new user)    | No H(u) available            | CBF on query features q; MPI regional prior          |
| Sparsity                 | Borrower-partner matrix thin | KNN on partner-side (I-I); fallback to rule-based    |
| Popularity bias          | Top partners dominate output | Diversity constraint: 1 non-top-quartile partner     |
| Gender proxy bias        | Sector/region correlated     | Gender excluded from s(·); retained as audit axis    |
| Geographic cold-catalog  | No partner for exact country | Neighbourhood-country fallback (Step 6 below)        |
| Data staleness (→2018)   | Evaluation looks outdated    | Synthetic data extension (Step 7 below)              |
| Bias in generated data   | Simulator inherits history   | Explicit decorrelation step in simulation (Step 7)   |
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


# =======================================================================
# STEP 3 — I-I (item-item) and U-U (user-user) collaborative filtering
# =======================================================================

def build_item_item_model(partner_feature_matrix, partner_ids, k=10):
    """
    partner_feature_matrix : np.ndarray, shape (n_partners, n_features)
                             One-hot or TF-IDF encoded partner features
    partner_ids            : list of partner IDs matching matrix rows
    Returns a fitted NearestNeighbors model and the index→id mapping.
    """
    model = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute")
    model.fit(partner_feature_matrix)
    return model, partner_ids


def get_ii_scores(liked_partner_ids, all_partner_ids, model, partner_feature_matrix, k=5):
    """
    Given a list of partner IDs the borrower has used before,
    returns a dict {partner_id: ii_score} for all partners.
    Score = average inverse-rank position across neighbourhoods of liked partners.
    """
    scores = {pid: 0.0 for pid in all_partner_ids}
    id_to_idx = {pid: i for i, pid in enumerate(all_partner_ids)}
    for liked_pid in liked_partner_ids:
        if liked_pid not in id_to_idx:
            continue
        idx = id_to_idx[liked_pid]
        query_vec = partner_feature_matrix[idx].reshape(1, -1)
        distances, indices = model.kneighbors(query_vec, n_neighbors=k + 1)
        for rank, (dist, neighbour_idx) in enumerate(
            zip(distances[0][1:], indices[0][1:]), start=1
        ):
            neighbour_pid = all_partner_ids[neighbour_idx]
            scores[neighbour_pid] += 1.0 / rank
    return scores


def build_user_user_model(borrower_feature_matrix, borrower_ids, k=10):
    """
    borrower_feature_matrix : np.ndarray, shape (n_borrowers, n_features)
                              Encode: country, sector, loan_amount_bin, term_bin, borrower_type
                              Do NOT include gender column.
    """
    model = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute")
    model.fit(borrower_feature_matrix)
    return model, borrower_ids


def get_uu_partner_scores(query_vec, uu_model, borrower_ids, borrower_to_partners, k=10):
    """
    query_vec           : np.ndarray, shape (1, n_features) — current borrower encoded
    borrower_to_partners: dict {borrower_id: [partner_id, ...]} from loan history
    Returns dict {partner_id: uu_score} aggregated from similar borrowers' choices.
    """
    distances, indices = uu_model.kneighbors(query_vec, n_neighbors=k + 1)
    scores = {}
    for rank, (dist, idx) in enumerate(zip(distances[0][1:], indices[0][1:]), start=1):
        neighbour_id = borrower_ids[idx]
        weight = 1.0 / rank
        for pid in borrower_to_partners.get(neighbour_id, []):
            scores[pid] = scores.get(pid, 0.0) + weight
    return scores


# =======================================================================
# STEP 4 — Hybrid scorer: blend CBF + U-U + I-I
# =======================================================================

def hybrid_score(cbf_scores, uu_scores, ii_scores,
                 w_cbf=0.6, w_uu=0.2, w_ii=0.2,
                 has_history=False):
    """
    cbf_scores : dict {partner_id: float}  — content-based similarity
    uu_scores  : dict {partner_id: float}  — user-user CF signal
    ii_scores  : dict {partner_id: float}  — item-item CF signal
    has_history: bool — if False, collapse all weight to CBF (cold-start)
    Returns dict {partner_id: float} of blended scores, normalised to [0,1].
    """
    if not has_history:
        w_cbf, w_uu, w_ii = 1.0, 0.0, 0.0

    all_ids = set(cbf_scores) | set(uu_scores) | set(ii_scores)

    def norm(d):
        max_v = max(d.values(), default=1.0)
        return {k: v / max_v for k, v in d.items()} if max_v > 0 else d

    cbf_n, uu_n, ii_n = norm(cbf_scores), norm(uu_scores), norm(ii_scores)

    blended = {}
    for pid in all_ids:
        blended[pid] = (
            w_cbf * cbf_n.get(pid, 0.0)
            + w_uu * uu_n.get(pid, 0.0)
            + w_ii * ii_n.get(pid, 0.0)
        )
    return blended


# =======================================================================
# STEP 5 — Bias-aware re-ranking
# =======================================================================

# Sectors found by EDA to be >80% one gender in training data — fill in as needed
GENDER_PROXY_SECTORS = []

REGION_NEIGHBOURS = {
    "Kenya":     ["Uganda", "Tanzania", "Rwanda", "Ethiopia"],
    "Indonesia": ["Philippines", "Cambodia", "Vietnam", "Timor-Leste"],
    "Mexico":    ["Guatemala", "Honduras", "El Salvador", "Nicaragua"],
    "Brazil":    ["Peru", "Colombia", "Bolivia", "Paraguay"],
}


def reweight_for_gender_proxy(scores, partner_sector_map, penalty=0.85):
    """
    Applies a small penalty to partners whose dominant sector is a
    known gender-correlated proxy, to counteract implicit bias.
    penalty: multiplier < 1.0 (0.85 = 15% discount)
    """
    adjusted = {}
    for pid, score in scores.items():
        sector = partner_sector_map.get(pid, "")
        if sector in GENDER_PROXY_SECTORS:
            adjusted[pid] = score * penalty
        else:
            adjusted[pid] = score
    return adjusted


def apply_neighbourhood_fallback(scores, query_country, partner_country_map,
                                 threshold=0.3, neighbour_weight=0.7):
    """
    If no partner in query_country exceeds threshold, expand to regional neighbours
    with a neighbour_weight discount on their scores.
    Returns (adjusted_scores, fallback_triggered).
    """
    in_country = {
        pid: s for pid, s in scores.items()
        if partner_country_map.get(pid) == query_country and s >= threshold
    }
    if in_country:
        return scores, False

    neighbours = REGION_NEIGHBOURS.get(query_country, [])
    expanded = {}
    for pid, s in scores.items():
        if partner_country_map.get(pid) in neighbours:
            expanded[pid] = s * neighbour_weight
        elif partner_country_map.get(pid) == query_country:
            expanded[pid] = s

    if expanded:
        return expanded, True
    return scores, False


def enforce_diversity(ranked_partners, partner_volume_quartile_map, top_n=5):
    """
    ranked_partners         : list of (partner_id, score) sorted descending
    partner_volume_quartile_map: dict {partner_id: int} where 4 = top volume quartile
    Ensures at least 1 of the top_n results is NOT in the top volume quartile.
    Returns list of (partner_id, score, is_diversity_pick) tuples.
    """
    primary = [(pid, s) for pid, s in ranked_partners if partner_volume_quartile_map.get(pid, 4) == 4]
    diverse = [(pid, s) for pid, s in ranked_partners if partner_volume_quartile_map.get(pid, 4) < 4]

    result = []
    primary_iter = iter(primary)
    diverse_iter = iter(diverse)
    diversity_added = False

    for i in range(top_n):
        if i == top_n - 1 and not diversity_added:
            pick = next(diverse_iter, None) or next(primary_iter, None)
            if pick:
                result.append((*pick, True))
                diversity_added = True
        else:
            pick = next(primary_iter, None)
            if pick:
                result.append((*pick, False))

    return result


def rerank(blended_scores, query_country,
           partner_sector_map, partner_country_map, partner_volume_quartile_map):
    """Full re-ranking pipeline: gender proxy re-weight → threshold → fallback → sort → diversity."""
    scores = reweight_for_gender_proxy(blended_scores, partner_sector_map)
    scores = {pid: s for pid, s in scores.items() if s >= 0.3}
    scores, fallback_triggered = apply_neighbourhood_fallback(
        scores, query_country, partner_country_map
    )
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    final = enforce_diversity(ranked, partner_volume_quartile_map)
    return final, fallback_triggered


# =======================================================================
# STEP 6 — History-based features for returning users
# =======================================================================

def extract_history_features(loan_history_df, borrower_id):
    """
    loan_history_df: DataFrame with columns [borrower_id, partner_id,
                     loan_amount, term_in_months, sector, repayment_interval,
                     funded_amount, status]
    Returns a dict of derived history signals for this borrower.
    """
    hist = loan_history_df[loan_history_df["borrower_id"] == borrower_id]
    if hist.empty:
        return {"has_history": False}

    completed = hist[hist["status"] == "paid"]
    return {
        "has_history":        True,
        "n_loans":            len(hist),
        "preferred_partners": hist["partner_id"].value_counts().index[:3].tolist(),
        "avg_loan_amount":    hist["loan_amount"].mean(),
        "avg_term_months":    hist["term_in_months"].mean(),
        "preferred_sectors":  hist["sector"].value_counts().index[:2].tolist(),
        "repayment_rate":     len(completed) / len(hist),
        "preferred_interval": (
            hist["repayment_interval"].mode()[0] if not hist.empty else None
        ),
    }


def apply_history_boosts(blended_scores, history):
    """
    Boosts scores for previously successful partners and returns the updated
    dict. Pass has_history=False history dict to skip silently.
    """
    if not history.get("has_history"):
        return blended_scores
    boosted = dict(blended_scores)
    for pid in history.get("preferred_partners", []):
        if pid in boosted:
            boosted[pid] *= 1.2
    return boosted


# =======================================================================
# STEP 7 — Data simulation to extend beyond 2018
# =======================================================================

def simulate_loans(base_df, target_years=range(2019, 2026), seed=42):
    """
    Extends the Kiva loan dataset to cover 2019–2025 by resampling
    from the 2016–2018 distribution with controlled drift.

    IMPORTANT: This simulation inherits the gender/sector/geography
    distribution of the training data. Biases present in 2014-2018
    records (e.g. women receiving smaller loans, rural borrowers
    underrepresented) will persist in generated records unless
    explicitly corrected via the decorrelation block below.

    Usage:
        full_df, synthetic_df = simulate_loans(kiva_loans_df)
        train_df = full_df[~full_df['is_simulated']]     # real data for training
        eval_df  = full_df[full_df['is_simulated'] & (full_df['disbursed_year'] >= 2023)]
    """
    rng = np.random.default_rng(seed)
    records = []

    base = base_df[base_df["disbursed_year"] >= 2016].copy()

    for year in target_years:
        n = int(len(base) * rng.uniform(0.9, 1.1))
        sample = base.sample(n=n, replace=True, random_state=seed + year).copy()
        sample["disbursed_year"] = year

        years_ahead = year - 2018
        sample["loan_amount"] = sample["loan_amount"] * (1.03 ** years_ahead)

        # --- BIAS INHERITANCE NOTE ---
        # The lines above preserve all correlations from the base data including:
        #   - gender × loan_amount correlation
        #   - country × sector concentration
        #   - partner popularity skew
        #
        # DECORRELATION BLOCK (optional, recommended for fairness evaluation):
        # sample['loan_amount'] = rng.permutation(sample['loan_amount'].values)
        # This breaks the gender-loan_amount correlation while preserving marginals.
        # Do NOT use for realism simulation; DO use when generating fairness audit sets.

        records.append(sample)

    simulated_df = pd.concat(records, ignore_index=True)
    simulated_df["is_simulated"] = True
    base_df = base_df.copy()
    base_df["is_simulated"] = False
    combined = pd.concat([base_df, simulated_df], ignore_index=True)
    return combined, simulated_df


# =======================================================================
# Helper: compute female-majority share for audit panel
# =======================================================================

def compute_female_share(final_recommendations, partner_metadata):
    """
    final_recommendations: list of (partner_id, score, is_diversity_pick)
    partner_metadata      : dict or DataFrame with 'borrower_genders' or similar
    Returns fraction of recommended partners whose primary borrower base is female.
    """
    if not final_recommendations:
        return 0.0
    female_count = 0
    for pid, _score, _div in final_recommendations:
        meta = partner_metadata.get(pid, {}) if isinstance(partner_metadata, dict) else {}
        if meta.get("female_majority", False):
            female_count += 1
    return female_count / len(final_recommendations)
