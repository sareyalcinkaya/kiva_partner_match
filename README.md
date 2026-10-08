# Kiva Partner Matcher

A recommender system that helps micro-entrepreneurs in low and middle income countries find the right microfinance field partner, with a plain-language reason for every match.

Built with the Kiva "Data Science for Good" dataset. Supports SDG 1 (No Poverty, target 1.4) and SDG 8 (Decent Work and Economic Growth, targets 8.3 and 8.10).

## The problem

Roughly 1.4 billion adults are still unbanked, and many micro-entrepreneurs cannot get credit that fits their business. The shortage is not only of capital. Borrowers face an information gap: they cannot tell which of many microfinance institutions serves their country, sector and loan size.

That gap is not neutral:

- Outreach through field agents and word of mouth favours urban borrowers, people already connected to an institution, and people comfortable with applications.
- Women, rural and first-time borrowers face several of these barriers at once.
- Borrowers shut out of formal credit often fall back on informal lenders at high rates.
- A poor match can push a household into over-indebtedness, so errors cost more than a bad movie suggestion would.

Kiva's interface is built for lenders, who browse loans and choose whom to fund. Borrowers have no equivalent tool for choosing a partner.

Learning from historical matches makes this harder. Partners that optimise repayment favour borrowers with credit history and collateral, and sector and region correlate with gender. A naive model would learn those patterns as "preferences" and reproduce them at scale.

## The solution

Given a borrower profile (country, sector, loan theme, amount, term, borrower type), the system returns the **five best-fit Kiva field partners**, ranked, each with an explanation. It is designed as an add-on that sits next to Kiva's existing platform.

```
Borrower profile ──► Content-based scoring ──► Hybrid blend ──► Bias-aware re-ranking ──► Top 5 + explanations
 (cold start OK)      one-hot sector x          CBF + user-user   threshold, neighbour       SHAP, LIME,
                      country x theme,          + item-item       fallback, diversity        "why not?" lookup
                      cosine similarity         (needs history)   constraint
```

Key design choices:

- **Cold start first.** No account or history is needed. The history-based paths (user-user and item-item collaborative filtering, history boosts) exist in `recommender.py` and switch on only when a borrower has past loans.
- **Gender is excluded from scoring.** It is kept as an audit dimension, so fairness can be checked without feeding it into the ranking.
- **Diversity constraint.** At least one of the five partners must come from outside the top volume quartile, which counters popularity bias.
- **Neighbour-country fallback.** If no catalogued partner serves the borrower's country, results expand to neighbouring countries with a 30% score discount, and the interface says so.
- **Explainability.** Each partner card shows SHAP and LIME feature contributions and a plain-language summary. A "why did this partner not appear?" lookup explains omissions.
- **Borrower welfare as the objective.** Partner and platform preferences enter as constraints, not as the thing being optimised.

More detail is in [docs/design.md](docs/design.md).

## Results

Offline evaluation over 875 sector x country queries, with ground truth taken from the partners that actually funded loans for that sector and country (`results/eval_results.csv`).

| Metric | Score |
|---|---|
| Precision@5 | 0.412 |
| Recall@5 | 0.962 |
| NDCG@5 | 0.934 |

- Most queries have only one to three relevant partners (mean 2.2), so Precision@5 cannot exceed about 0.43 on this set. 0.412 is close to that ceiling.
- NDCG@5 by sector ranges from 0.895 (Entertainment) to 0.959 (Agriculture).
- The weakest countries are Mexico (0.489), Indonesia (0.618), Kenya (0.648), South Africa (0.659) and Kyrgyzstan (0.664).
- Three queries score zero because no suitable partner exists in the catalogue: Entertainment x Ghana, Health x Brazil, Transportation x Israel.

## Limitations

Please read these before citing the numbers.

- **The evaluation is partly circular.** "Relevant" means a partner has funded a loan in that sector and country, and the evaluated recommender already restricts results to the borrower's country. The scores show the system surfaces partners that really operate there. They do not show better outcomes for borrowers.
- **`evaluate.py` scores the content-based path only.** It does not apply the full re-ranking pipeline with the diversity constraint.
- **Borrower type, repayment term and loan amount are shown in the interface but are not encoded** in the feature matrix. The encoder uses sector, country and loan theme.
- **The gender audit panel is a placeholder.** The dataset has no partner-level female-majority flag, so it shows 0%. The `borrower_genders` column in `kiva_loans.csv` could be aggregated per partner to fill it in.
- **The gender-proxy reweighting list is empty.** `GENDER_PROXY_SECTORS` in `recommender.py` needs to be filled in from an exploratory analysis.
- **The MPI poverty index is loaded but unused** in scoring.
- **The data ends in 2018.** `simulate_loans()` can extend it to 2025 for stress tests. Simulated rows are flagged `is_simulated=True`, inherit historical biases unless the decorrelation step is applied, and must not be used for training.
- **SHAP and LIME use a 120-partner background sample** for speed, so explanations are approximate.

## Run it

Python 3.10 or later.

```bash
git clone <this repository>
cd kiva-partner-matcher
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

1. Download the four Kaggle CSV files into `data/` (see [data/README.md](data/README.md)).
2. Build the model (once, or after any data change):
   ```bash
   python process.py
   ```
3. Start the app:
   ```bash
   streamlit run app.py
   ```
4. Optional, reproduce the offline evaluation:
   ```bash
   python evaluate.py
   ```

Use the same Python environment for steps 2 and 3. Mixing environments can cause a pandas `StringDtype` error when the model pickle loads.

## Repository layout

```
app.py             Streamlit interface and explanations
process.py         Filters partners, builds features and the similarity matrix
recommender.py     Collaborative filtering paths, hybrid scorer, re-ranking, simulation
evaluate.py        Precision@5, Recall@5, NDCG@5
results/           eval_results.csv from the last evaluation run
docs/design.md     Task definition, problem to design mapping, pipeline
data/              Place the Kaggle CSV files here (not committed)
```

## Next steps

- Encode loan amount, term and borrower type in the features.
- Fill the gender audit with per-partner shares from `borrower_genders` and set the proxy-sector list from data.
- Use the MPI score as a regional poverty prior.
- Evaluate with outcome-oriented measures and a held-out time split, not only partner presence.
- Add borrower accounts so the collaborative filtering paths become active.

## Data and credits

Data: [Kiva Data Science for Good](https://www.kaggle.com/datasets/kiva/data-science-for-good-kiva-crowdfunding), Kaggle, 2018. Not affiliated with Kiva.

Author: Sare Melek Yalcinkaya. Started as a Recommender Systems course project at TU Wien. Licensed under MIT.
