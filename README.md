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

## My solution

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

## Design

### Formal task

Let `P` be the set of Kiva field partners and `q = (country, sector, loan theme, amount, term, borrower type)` a borrower query.
Return an ordered list `R = [p1 ... p5]` such that:

1. partners are ranked by their corrected score `r(p, q)`,
2. at least one partner is outside the top volume quartile (diversity constraint),
3. every candidate has similarity `s(p, q) >= 0.3`,
4. gender is not an input to `s` or `r`.

### Known problems and the design response

| Problem | Where it shows up | Response |
|---|---|---|
| Cold start | A new borrower has no loan history | Content-based scoring on the query profile |
| Sparsity | The borrower-partner matrix is thin | Item-item neighbours on the partner side, rule-based fallback |
| Popularity bias | High-volume partners dominate any ranking | Diversity constraint: the last slot goes to a lower-volume partner |
| Gender proxy bias | Sector and region correlate with gender | Gender excluded from scoring and kept as an audit dimension |
| Empty catalogue for a country | Some countries have no catalogued partner | Neighbouring-country fallback with a 30% score discount |
| Data ends in 2018 | Evaluation looks outdated | Optional synthetic extension (`simulate_loans`), flagged `is_simulated` |
| Bias inherited by simulated data | Resampling copies historical inequities | Documented decorrelation step for fairness audit sets |

### Pipeline

1. `process.py` filters partners with 100 or more loans, one-hot encodes `sector x country x loan theme`, and saves a partner feature matrix.
2. `recommender.hybrid_score` blends content-based, user-user and item-item scores (weights 0.6 / 0.2 / 0.2). With no history it collapses to content-based only.
3. `recommender.rerank` applies the gender-proxy reweighting hook, the 0.3 threshold, the neighbour-country fallback, sorting, and the diversity constraint.
4. `app.py` shows the top 5 with a match score, SHAP and LIME explanations, a "why not this partner?" lookup and a fairness audit panel.

### Why template and feature-based explanations

Borrowers and Kiva staff need to see why a partner was suggested. Explanations come from feature contributions (SHAP, LIME) and plain-language summaries, not from a generative model, so every explanation can be traced back to inputs.

### Stakeholder tension

A partner optimising repayment prefers borrowers with credit history and collateral, which are the traits that exclude the most vulnerable. A model trained on past matches would reproduce that. The design therefore treats the borrower's access to a suitable partner as the objective, and treats partner and platform preferences as constraints.

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

## What I learned

**About the problem**
- In credit, a bad recommendation has real costs. A wrong partner can mean a rejected application or a household taking on debt it cannot carry, so fairness and transparency had to be design requirements from the start, not extras.
- Historical data carries historical inequity. In the Kiva data, male borrowers receive larger loans than female borrowers within the same sector and country, and a model that learns from past matches would treat that as a preference.
- Whose utility to optimise is a design decision. Partners, donors and the platform want different things than the borrower, so I made borrower welfare the goal and treated the others as constraints.

**About building the recommender**
- Cold start decides the architecture. Most real users have no history, so a content-based model had to carry the system, with user-user and item-item collaborative filtering waiting for borrower accounts.
- Removing gender from the features is not enough, because sector and region act as proxies. That is why gender is kept as an audit dimension instead of simply dropped. It is also why I am careful to say the system is gender-neutral by design, not proven fair.
- Re-ranking is a practical lever. A threshold, a neighbour-country fallback and a diversity slot let me counter popularity bias and empty catalogues without retraining anything.
- Explanations should be traceable. Feature-based explanations (SHAP, LIME and plain-language summaries) are easier to audit than generated text.

**About evaluation**
- High scores can mislead. NDCG@5 of 0.934 looked strong until I checked how "relevant" was defined: it is partly circular. Reading the metric's construction mattered more than the number.
- Averages hide failures. Mexico (0.489), Indonesia (0.618) and Kenya (0.648) sit far below the mean, and three queries score zero because the catalogue has no suitable partner. Looking at breakdowns by country and sector found problems the overall score did not show.
- Precision@5 depends on the data. With about two relevant partners per query, it cannot exceed roughly 0.43, so a low-looking value is not automatically a weak model.
- Synthetic data needs care. Resampling the past reproduces its biases unless I break them on purpose, and simulated rows must never be used for training.

**About engineering**
- Make the pipeline reproducible: one Python environment for training and serving (a pickle built in one environment failed to load in another), correct file paths, and pinned or compatible dependencies.
- Be explicit about what is real and what is a placeholder. Listing the simplifications openly makes the work more credible.

## Data

The Kiva data is not included in this repository. Download the [Kiva Data Science for Good](https://www.kaggle.com/datasets/kiva/data-science-for-good-kiva-crowdfunding) dataset from Kaggle and place these four files in `data/`:

| File | Content |
|---|---|
| `kiva_loans.csv` | Individual loan records, 2014 to mid 2018 (about 672k rows) |
| `loan_themes_by_region.csv` | Partner level features: sector, country, loan theme, region, MPI |
| `loan_theme_ids.csv` | Loan to theme mapping |
| `kiva_mpi_region_locations.csv` | Multidimensional Poverty Index by region |

Check the dataset's license terms on Kaggle before redistributing it.

## Run it

Python 3.10 or later.

```bash
git clone <this repository>
cd kiva-partner-matcher
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

1. Download the four Kaggle CSV files into `data/` (see [Data](#data)).
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
