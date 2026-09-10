# Project proposal: model versus market at the 2026 World Cup

**Research question.** Did a match-result forecasting model built from public international football data produce better predictions than Polymarket, a liquid prediction market, over the 104 matches of the 2026 FIFA World Cup, and would trading the model's picks have been profitable?

**Business context.** During the tournament I used a model of this kind to place positions of 100 to 200 USD per match on Polymarket without keeping records. The project rebuilds the model under a strict pre-tournament training cut-off and evaluates it against the market with full ground truth, which is the same question any analyst faces when a private model competes with an established benchmark.

**Data sources (two primary, one auxiliary).**

1. International football results, martj42/international_results (GitHub, CSV). 49,547 senior international matches from 1872 to August 2026, 9 columns: date, home_team, away_team, home_score, away_score, tournament, city, country, neutral. Cleaning: harmonise two team names; exclude matches played after the tournament; note that knockout scores include extra time.
2. Polymarket match markets (Gamma API and CLOB API, JSON). One event per World Cup match with three binary markets (home win, draw, away win); 312 markets, 28,247 price observations at 12-hour resolution from market opening to kickoff, plus metadata (question, kickoff time, resolution, volume). Cleaning: event identifiers mix FIFA and ISO country codes and use UTC dates; market questions use alternative spellings (Türkiye, IR Iran, Côte d'Ivoire, Cabo Verde); markets resolve on the 90-minute result.
3. Auxiliary: openfootball/worldcup.json 2026 (GitHub, JSON). 104 matches with stage, round, venue, half-time, full-time and extra-time scores; defines the 90-minute result used for evaluation.

**Merge plan.** Fixture key: (home_team, away_team, kickoff date) with a team-name and country-code mapping table. Results file to fixture list on team names and date; fixture list to Polymarket events on country codes and UTC date, with a text-search fallback; market type identified from the question text. Pre-kickoff price = last recorded price before the kickoff timestamp. Validation: every market's resolution is checked against the official 90-minute score.

**Analysis and techniques (Python, SQLite).** Elo ratings with importance-weighted K-factors and five-match form features; HistGradientBoostingClassifier for the three-way result, benchmarked against multinomial logistic regression and base rates on a 2022 to 2026 temporal hold-out; Poisson goals model (statsmodels GLM) for expected goals and a second set of outcome probabilities; ensemble by averaging. Sequential prediction of the 104 matches with ratings updated after each match and no retraining. Evaluation with accuracy, log loss and multi-class Brier score against normalised market probabilities, paired bootstrap for the difference, calibration curves, and a betting simulation with a flat 150 USD stake under four explicit rules (model pick every match; model pick when edge exceeds five points; largest edge; market favourite as control).

**Deliverable.** Executive report as a GitHub Pages site with a PDF export: title, executive summary, introduction, methods, results with captioned figures, conclusion with recommendations.
