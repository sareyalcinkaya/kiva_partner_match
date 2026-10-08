# Design notes

## Formal task

Let `P` be the set of Kiva field partners and `q = (country, sector, loan theme, amount, term, borrower type)` a borrower query.
Return an ordered list `R = [p1 ... p5]` such that:

1. partners are ranked by their corrected score `r(p, q)`,
2. at least one partner is outside the top volume quartile (diversity constraint),
3. every candidate has similarity `s(p, q) >= 0.3`,
4. gender is not an input to `s` or `r`.

## Known problems and the design response

| Problem | Where it shows up | Response |
|---|---|---|
| Cold start | A new borrower has no loan history | Content-based scoring on the query profile |
| Sparsity | The borrower-partner matrix is thin | Item-item neighbours on the partner side, rule-based fallback |
| Popularity bias | High-volume partners dominate any ranking | Diversity constraint: the last slot goes to a lower-volume partner |
| Gender proxy bias | Sector and region correlate with gender | Gender excluded from scoring and kept as an audit dimension |
| Empty catalogue for a country | Some countries have no catalogued partner | Neighbouring-country fallback with a 30% score discount |
| Data ends in 2018 | Evaluation looks outdated | Optional synthetic extension (`simulate_loans`), flagged `is_simulated` |
| Bias inherited by simulated data | Resampling copies historical inequities | Documented decorrelation step for fairness audit sets |

## Pipeline

1. `process.py` filters partners with 100 or more loans, one-hot encodes `sector x country x loan theme`, and saves a partner feature matrix.
2. `recommender.hybrid_score` blends content-based, user-user and item-item scores (weights 0.6 / 0.2 / 0.2). With no history it collapses to content-based only.
3. `recommender.rerank` applies the gender-proxy reweighting hook, the 0.3 threshold, the neighbour-country fallback, sorting, and the diversity constraint.
4. `app.py` shows the top 5 with a match score, SHAP and LIME explanations, a "why not this partner?" lookup and a fairness audit panel.

## Why template and feature-based explanations

Borrowers and Kiva staff need to see why a partner was suggested. Explanations come from feature contributions (SHAP, LIME) and plain-language summaries, not from a generative model, so every explanation can be traced back to inputs.

## Stakeholder tension

A partner optimising repayment prefers borrowers with credit history and collateral, which are the traits that exclude the most vulnerable. A model trained on past matches would reproduce that. The design therefore treats the borrower's access to a suitable partner as the objective, and treats partner and platform preferences as constraints.
