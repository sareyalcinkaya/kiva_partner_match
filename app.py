import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OrdinalEncoder
import shap
from lime import lime_tabular

from recommender import (
    hybrid_score,
    rerank,
    extract_history_features,
    apply_history_boosts,
    compute_female_share,
    REGION_NEIGHBOURS,
)

st.set_page_config(page_title="Kiva Partner Matcher", layout="centered")

# ── Load data ──────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("model/similarity_matrix.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
partners = model["partners"]
encoder = model["encoder"]

@st.cache_data(show_spinner=False)
def load_ground_truth():
    """Build (sector, country) → set of relevant partner names from actual loans."""
    loans = pd.read_csv("data/kiva_loans.csv")
    partner_map = (partners[["Partner ID", "Field Partner Name"]]
                   .drop_duplicates()
                   .rename(columns={"Partner ID": "partner_id"}))
    merged = loans.merge(partner_map, on="partner_id", how="inner")
    return (merged
            .groupby(["sector", "country"])["Field Partner Name"]
            .apply(set)
            .to_dict())

ground_truth = load_ground_truth()


# ── Ranking metrics ────────────────────────────────────────────────────────────
def precision_at_k(recommended: list, relevant: set, k: int = 5) -> float:
    return sum(1 for p in recommended[:k] if p in relevant) / k

def recall_at_k(recommended: list, relevant: set, k: int = 5) -> float:
    if not relevant:
        return 0.0
    return sum(1 for p in recommended[:k] if p in relevant) / len(relevant)

def ndcg_at_k(recommended: list, relevant: set, k: int = 5) -> float:
    dcg  = sum(1.0 / np.log2(i + 2) for i, p in enumerate(recommended[:k]) if p in relevant)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0

# ── Explanation helpers ────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def compute_explanations(borrower_tuple, p_sector, p_country, p_theme, _encoder, _partners):
    """
    Compute SHAP and LIME feature contributions for one partner recommendation.
    borrower_tuple: (sector, country, theme) — hashable so Streamlit can cache it.
    """
    p_vec = _encoder.transform([[str(p_sector), str(p_country), str(p_theme)]])

    bg = (_partners[["sector", "country", "Loan Theme Type"]]
          .fillna("Unknown")
          .astype(str)
          .sample(min(120, len(_partners)), random_state=0))

    # OrdinalEncoder so SHAP/LIME receive numeric arrays
    ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    bg_int = ord_enc.fit_transform(bg.values).astype(float)
    instance_int = ord_enc.transform([list(borrower_tuple)]).astype(float)

    _cols = ["sector", "country", "Loan Theme Type"]

    def score_fn(X_int):
        scores = []
        for row in X_int:
            idx = np.clip(row.astype(int), 0, None)
            row_str = ord_enc.inverse_transform(idx.reshape(1, -1))[0]
            df = pd.DataFrame([row_str], columns=_cols)
            encoded = _encoder.transform(df)
            scores.append(cosine_similarity(encoded, p_vec)[0][0])
        return np.array(scores)

    feature_names = ["Sector", "Country", "Loan Theme"]

    # SHAP via KernelExplainer (exact enough for 3 features)
    shap_exp = shap.KernelExplainer(score_fn, bg_int)
    shap_vals = shap_exp.shap_values(instance_int, nsamples=100, silent=True)
    shap_result = {
        "values": np.array(shap_vals).flatten().tolist(),
        "features": feature_names,
        "base": float(shap_exp.expected_value),
        "instance": list(borrower_tuple),
    }

    # LIME via LimeTabularExplainer
    cat_names = {i: [str(v) for v in ord_enc.categories_[i]] for i in range(3)}
    lime_exp_obj = lime_tabular.LimeTabularExplainer(
        bg_int,
        feature_names=feature_names,
        categorical_features=[0, 1, 2],
        categorical_names=cat_names,
        mode="regression",
        random_state=42,
    )
    lime_exp = lime_exp_obj.explain_instance(instance_int[0], score_fn, num_features=3)
    lime_result = {
        "features": [f for f, _ in lime_exp.as_list()],
        "values": [v for _, v in lime_exp.as_list()],
    }

    return shap_result, lime_result


def _bar_chart(ax, features, values, title, xlabel):
    colors = ["#2196F3" if v >= 0 else "#EF5350" for v in values]
    ax.barh(features, values, color=colors, edgecolor="none", height=0.5)
    ax.axvline(0, color="#555", linewidth=0.8, linestyle="--")
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)


def show_explanations(borrower_tuple, row, partners_df, enc):
    partner_theme = str(row.get("Loan Theme Type", "Unknown") or "Unknown")
    with st.spinner("Computing SHAP & LIME explanations…"):
        shap_result, lime_result = compute_explanations(
            borrower_tuple,
            row["sector"], row["country"], partner_theme,
            enc, partners_df,
        )

    st.markdown("**Why was this partner recommended?**")
    st.caption(
        "**SHAP** measures each feature's push/pull on the score relative to the "
        "average partner. **LIME** fits a local linear model around your profile "
        "to show which features matter most for this specific match."
    )

    col_s, col_l = st.columns(2)

    with col_s:
        fig, ax = plt.subplots(figsize=(3.5, 2.2))
        _bar_chart(
            ax,
            shap_result["features"],
            shap_result["values"],
            f"SHAP  (base {shap_result['base']:.2f})",
            "Contribution to score",
        )
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_l:
        fig, ax = plt.subplots(figsize=(3.5, 2.2))
        _bar_chart(
            ax,
            lime_result["features"],
            lime_result["values"],
            "LIME  (local linear fit)",
            "Weight",
        )
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # Plain-English summary
    top_feature = shap_result["features"][
        int(np.argmax(np.abs(shap_result["values"])))
    ]
    top_val = shap_result["values"][
        int(np.argmax(np.abs(shap_result["values"])))
    ]
    direction = "boosted" if top_val > 0 else "reduced"
    st.caption(
        f"The strongest driver is **{top_feature}**, which {direction} the match "
        f"score by {abs(top_val):.0%}. Blue bars push the score up; red bars pull it down."
    )


# ── Screen 1: Profile Form ─────────────────────────────────────────────────────
st.title("Kiva partner matcher")
st.caption("No registration required")

# Country outside form so theme list reacts to it
country = st.selectbox("Country", sorted(partners["country"].dropna().unique()))

country_themes = sorted(
    partners[partners["country"] == country]["Loan Theme Type"].dropna().unique()
)
theme_options = ["No preference"] + country_themes

with st.form("profile_form"):
    sector = st.selectbox("Sector", ["Agriculture", "Food", "Retail", "Services",
                                      "Education", "Health", "Housing", "Other"])
    theme  = st.selectbox("Loan theme type", theme_options,
                          help="Narrows results to partners that specialise in this theme")
    amount = st.slider("Loan amount (USD)", 25, 5000, 400, step=25)
    term   = st.selectbox("Repayment term", ["6 months", "12 months", "18 months", "24 months"])
    btype  = st.radio("Borrower type", ["Individual", "Group"], horizontal=True)
    submit = st.form_submit_button("Find partners")

# ── Screen 2: Results ──────────────────────────────────────────────────────────
if submit:
    st.divider()

    theme_label = theme if theme != "No preference" else "any theme"
    st.info(f"**Profile:** {country} · {sector} · {theme_label} · ${amount:,} · {term} · {btype}")
    st.caption("Recommendations are gender-neutral — gender is not used to rank partners.")

    # Cold-start: no borrower history in this prototype — H(u) = ∅
    history = {"has_history": False}

    # Step 8a — show which CF path was used
    if history["has_history"]:
        st.info(
            f"Personalised using your {history['n_loans']} previous Kiva loans "
            f"(hybrid CBF + U-U + I-I mode)"
        )
    else:
        st.info("No loan history found. Recommendations are based on your profile only (CBF mode).")

    theme_input = theme if theme != "No preference" else sector
    borrower_tuple = (sector, country, theme_input)   # used for caching too
    borrower_vec = encoder.transform([[str(sector), str(country), str(theme_input)]])
    partner_vecs = encoder.transform(
        partners[["sector", "country", "Loan Theme Type"]].fillna("Unknown").astype(str)
    )
    cbf_scores_arr = cosine_similarity(borrower_vec, partner_vecs)[0]

    # Build CBF score dict keyed by row index (proxy for partner_id)
    cbf_scores = {i: float(s) for i, s in enumerate(cbf_scores_arr)}

    # Hybrid blend (cold-start collapses to pure CBF)
    blended = hybrid_score(cbf_scores, {}, {}, has_history=False)
    blended = apply_history_boosts(blended, history)

    # Build lookup maps for re-ranking
    partner_sector_map   = {i: str(row["sector"])  for i, row in partners.iterrows()}
    partner_country_map  = {i: str(row["country"]) for i, row in partners.iterrows()}
    volume_threshold     = partners["amount"].quantile(0.75)
    partner_volume_q_map = {
        i: (4 if row.get("amount", 0) >= volume_threshold else 1)
        for i, row in partners.iterrows()
    }

    # Re-rank with bias-awareness and diversity enforcement
    final_tuples, fallback_triggered = rerank(
        blended, country,
        partner_sector_map, partner_country_map, partner_volume_q_map,
    )

    # Step 8c — neighbourhood fallback notice
    if fallback_triggered:
        neighbour_list = ", ".join(REGION_NEIGHBOURS.get(country, ["neighbouring countries"]))
        st.warning(
            f"No partners found active in {country}. "
            f"Showing results from neighbouring countries ({neighbour_list})."
        )

    # Reconstruct results DataFrame from re-ranked tuples
    if final_tuples:
        result_indices = [pid for pid, _s, _div in final_tuples]
        results = partners.loc[result_indices].copy().reset_index(drop=True)
        results["score"] = [s for _pid, s, _div in final_tuples]
        results["is_diversity_pick"] = [div for _pid, _s, div in final_tuples]
        results = results.drop_duplicates(subset=["Field Partner Name"]).head(5)
    else:
        # Fallback: original CBF sort when re-ranker returns nothing
        filtered = partners.copy()
        filtered["score"] = cbf_scores_arr
        filtered = filtered[filtered["country"] == country]
        filtered = (filtered
                    .sort_values(["score", "amount"], ascending=[False, False])
                    .drop_duplicates(subset=["Field Partner Name"]))
        top4 = filtered.head(4)
        volume_threshold2 = filtered["amount"].quantile(0.75)
        diversity_pick = filtered[filtered["amount"] < volume_threshold2].iloc[4:10].head(1)
        results = pd.concat([top4, diversity_pick]).reset_index(drop=True)
        results["is_diversity_pick"] = [False] * 4 + [True] * len(diversity_pick)

    st.subheader("Top 5 recommended partners")

    # ── Ranking quality metrics ────────────────────────────────────────────────
    relevant = ground_truth.get((sector, country), set())
    recommended_names = results["Field Partner Name"].tolist()
    p5    = precision_at_k(recommended_names, relevant)
    r5    = recall_at_k(recommended_names, relevant)
    ndcg5 = ndcg_at_k(recommended_names, relevant)

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Precision@5", f"{p5:.0%}",
               help="Fraction of the 5 shown partners that are genuinely relevant for your profile.")
    mc2.metric("Recall@5",    f"{r5:.0%}",
               help="Fraction of all relevant partners that appear in the top 5.")
    mc3.metric("NDCG@5",      f"{ndcg5:.4f}",
               help="Normalised Discounted Cumulative Gain — rewards relevant partners ranked higher. 1.0 = perfect order.")

    if not relevant:
        st.caption("ℹ️ No historical loans found for this exact sector × country pair — metrics shown as 0.")
    st.divider()

    for i, row in results.iterrows():
        is_diversity_pick = bool(row.get("is_diversity_pick", i == 4))
        score_pct = int(row["score"] * 100)

        label = f"{'⭐ ' if i < 2 else ''}**{row['Field Partner Name']}**"
        if is_diversity_pick:
            label = f"🌍 **{row['Field Partner Name']}** — Regional specialist pick"

        with st.expander(label, expanded=(i == 0)):
            st.progress(min(score_pct / 100, 1.0), text=f"Match score: {score_pct}%")

            # Step 8b — diversity pick label
            if is_diversity_pick:
                st.caption(
                    "🌍 Regional specialist pick — included to show alternatives "
                    "beyond the most popular partners."
                )
            else:
                st.write(f"This partner focuses on **{row['sector']}** loans in "
                         f"**{row['country']}** and has a strong record with "
                         f"borrowers in your loan range.")

            col1, col2 = st.columns(2)
            col1.metric("Sector", row["sector"])
            col2.metric("Loan Theme", row.get("Loan Theme Type", "General"))
            st.caption(f"Region: {row.get('region', 'N/A')}")
            st.markdown("[View on Kiva.org →](https://www.kiva.org/partners)")

            st.divider()
            show_explanations(borrower_tuple, row, partners, encoder)

    st.divider()

    # Step 8d — Gender fairness audit panel
    with st.expander("Gender fairness audit"):
        st.write(
            "Gender is not used in scoring. "
            "Audit: % of recommended partners whose primary borrower base is female:"
        )
        audit_tuples = [
            (i, row["score"], bool(row.get("is_diversity_pick", False)))
            for i, row in results.iterrows()
        ]
        female_pct = compute_female_share(audit_tuples, {})
        st.metric("Female-majority partners in results", f"{female_pct:.0%}")
        st.caption(
            "Note: female-majority classification requires partner-level gender data "
            "not present in the 2014–2018 Kiva dataset. Extend partner_metadata to "
            "enable this metric."
        )

    st.divider()

    # Why not lookup
    st.subheader("Why did a specific partner not appear?")
    query = st.text_input("Type a partner name...")
    if query:
        match = partners[partners["Field Partner Name"].str.contains(
            query, case=False, na=False)]
        if match.empty:
            st.warning("Partner not found in the catalog.")
        else:
            row = match.iloc[0]
            if row["country"] != country:
                st.info(f"**{row['Field Partner Name']}** does not operate in {country}.")
            elif row.get("amount", 0) < amount * 0.5:
                st.info(f"**{row['Field Partner Name']}**'s typical loan size "
                        f"may be below your requested amount.")
            else:
                st.info(f"**{row['Field Partner Name']}** was not in the top 5 "
                        f"matches for your profile. Their sector or region may "
                        f"not closely match your request.")

    st.caption("⚠️ Partner data is from 2014–2018. Verify status on Kiva.org before applying.")
