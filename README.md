# Model versus market: a match forecasting model against Polymarket at the 2026 World Cup

Capstone project, Professional Certificate in Data Analytics. The executive report is a
GitHub Pages site (`docs/index.html`); a PDF export is included as a fallback
(`WC2026_capstone_executive_report.pdf`).

## Question

Did a result-forecasting model built from public match history produce better predictions
than Polymarket over the 104 matches of the 2026 FIFA World Cup, and would trading its picks
have made money?

## Data sources

| Source | Role | Rows |
|---|---|---|
| [martj42/international_results](https://github.com/martj42/international_results) (CSV) | Training corpus, official results | 49,547 matches, 9 columns (count taken from the CSV on 10 September 2026; the repository README text lags behind the file) |
| [Polymarket](https://docs.polymarket.com) Gamma API + CLOB API | Independent benchmark: pre-kickoff prices | 312 markets, 28,247 price observations |
| [openfootball/worldcup.json](https://github.com/openfootball/worldcup.json) 2026 (JSON) | Auxiliary: stages, venues, 90-minute scores | 104 matches, 17 columns |

## How to run

```bash
pip install -r requirements.txt
python 00_verify_sources.py     # checks the three sources are live and match the local copies
python 01_build_database.py     # downloads the sources, builds wc2026.db (Polymarket fetch is cached per match; re-run if interrupted)
python 02_model.py              # Elo + form features, hold-out benchmark, final models, 104 predictions
python 03_evaluate.py           # merge with prices, metrics, betting simulation, figures in ./assets
python 04_build_report.py       # renders docs/index.html from the computed results
python 05_export_prose.py       # exports the report text to docs/prose_for_review.txt
```

`capstone.ipynb` runs the four scripts in order from Jupyter.

`01_build_database.py` accepts an optional batch size (`python 01_build_database.py 40`) to fetch
Polymarket in chunks; every fetched match is cached under `data/polymarket_raw/`, so the script can
be re-run until all 104 matches are present.

## Repository layout

```
00_verify_sources.py   01_build_database.py   02_model.py   03_evaluate.py
04_build_report.py     05_export_prose.py    capstone.ipynb
wc2026.db              SQLite: matches_history, wc2026_matches, polymarket_markets, polymarket_prices,
                       elo_pre_tournament, model_predictions, evaluation
data/                  raw copies of every source (CSV, JSON, per-match Polymarket JSON)
results/               metrics, ledgers, calibration, predictions (CSV/JSON)
assets/                figures (PNG)
docs/                  GitHub Pages site (index.html, assets/, data/)
PROPOSAL.md            one-page project proposal (Modules 21-22)
```

## Publishing the report on GitHub Pages

1. Create a new public repository, for example `wc2026-capstone`, and push this folder to `main`.
2. In the repository, open Settings > Pages, set Source to "Deploy from a branch", branch `main`,
   folder `/docs`, and save.
3. The report appears at `https://<username>.github.io/wc2026-capstone/` after a minute or two.
4. Optionally link it from the portfolio site.

## Main results

| Forecaster | Accuracy | Log loss | Brier |
|---|---|---|---|
| Ensemble (gradient boosting + Poisson) | 60.6% | 0.906 | 0.535 |
| Polymarket (normalised pre-kickoff prices) | 63.5% | 0.840 | 0.495 |

Flat 150 USD on the model's pick in every match: -291 USD on 15,600 (-1.9%).
Only when the model's edge over the market exceeds five points: +246 USD on 3,750 (+6.6%, 25 bets,
95% CI -1,346 to +1,894). Market favourite every match (control): +520 USD (+3.3%).
