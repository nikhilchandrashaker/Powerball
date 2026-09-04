# POWER — Powerball Occurrence & Winning-pattern Evaluation Research

Rigorous randomness audit of Powerball draw history (1992–2026). Full write-up in `REPORT.md`.

## Files

**Report & visuals**
- `REPORT.md` — full write-up: methodology, results, RANDOM score table, conclusions
- `plot_random_score.png` — RANDOM score by era
- `plot_white_freq_current_era.png` — observed white-ball frequency vs. uniform expectation, current era
- `plot_sum_distribution.png` — observed vs. simulated draw-sum distribution, current era
- `plot_gap_pvalues.png` — "overdue number" significance test by era
- `plot_ml_vs_baseline.png` — ML model vs. random baseline, log loss by CV fold

**Pipeline (run in this order)**
1. `eras.py` — loads `powerball_all.csv`, assigns each draw to one of the 7 verified rule-change eras, sanity-checks against observed max values → writes `powerball_with_era.csv`
2. `frequency.py` — chi-square uniformity test per era, white balls + Powerball → `frequency_results.csv`
3. `gaps.py` — logistic regression testing whether gap-since-last-seen predicts next appearance ("overdue" test) → `gap_results.csv`
4. `combos.py` — odd/even split, sum, range, consecutive pairs, decade spread, repeats-from-previous-draw, all vs. simulated random baselines → `powerball_with_features.csv`
5. `random_score.py` — combines F/G/A/C/T p-values into the composite RANDOM score per era → `random_scores.csv`
6. `ml_experiment.py` — builds a long-format panel (draw × number), trains logistic regression with time-series CV, compares log loss / AUC to the constant-probability baseline → `ml_results.json`
7. `make_plots.py` — generates the PNGs above from the CSVs/JSON produced by steps 1–6

**Data**
- `powerball_with_features.csv` — cleaned draws with era labels and all engineered combination-structure features (this is the main analysis-ready dataset)

## Re-running on updated data

The source dataset auto-updates after every Mon/Wed/Sat drawing. To refresh everything:

```bash
# replace powerball_all.csv with the latest export, then:
python3 eras.py
python3 frequency.py
python3 gaps.py
python3 combos.py
python3 random_score.py
python3 ml_experiment.py
python3 make_plots.py
```

Each script reads the previous step's output, so re-running the full chain recomputes every table, score, and plot with no code changes — including the RANDOM score for the current era as it accumulates more draws.

## Dependencies

`pandas`, `numpy`, `scipy`, `statsmodels`, `scikit-learn`, `matplotlib`
(`pip install pandas numpy scipy statsmodels scikit-learn matplotlib`)
