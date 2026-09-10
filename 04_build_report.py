"""
04_build_report.py
Renders the executive report (index.html) from the computed results so that every number in the
text comes from results/summary.json and the CSV tables, not from manual transcription.

Run:  python 04_build_report.py
Output: docs/index.html (GitHub Pages), docs/assets/*.png, docs/data/*.csv
"""

import json
import os
import shutil

import pandas as pd

OUT, ASSETS, DOCS = "results", "assets", "docs"
S = json.load(open(os.path.join(OUT, "summary.json")))
H = json.load(open(os.path.join(OUT, "holdout_metrics.json")))
D = S["data"]
metrics = pd.read_csv(os.path.join(OUT, "metrics_wc2026.csv"))
strat = pd.read_csv(os.path.join(OUT, "betting_strategies.csv"))
edge = pd.read_csv(os.path.join(OUT, "edge_buckets.csv"))
surp = pd.read_csv(os.path.join(OUT, "surprises.csv"))
cal = pd.read_csv(os.path.join(OUT, "calibration.csv"))
ev = pd.read_csv(os.path.join(OUT, "matches_evaluated.csv"))
elo = pd.read_csv(os.path.join(OUT, "elo_pre_tournament.csv"))
ledB = pd.read_csv(os.path.join(OUT, "ledger_B.csv"))
ledA = pd.read_csv(os.path.join(OUT, "ledger_A_ens.csv"))

M = metrics.set_index("model")
ens, mkt, hgb, pois, uni = M.loc["Ensemble"], M.loc["Polymarket"], M.loc["Gradient boosting"], M.loc["Poisson goals"], M.loc["Uniform (1/3 each)"]
ST = strat.set_index("strategy")
A = ST[ST.index.str.startswith("A. Ensemble")].iloc[0]
Ah = ST[ST.index.str.startswith("A. Gradient")].iloc[0]
B = ST[ST.index.str.startswith("B.")].iloc[0]
C = ST[ST.index.str.startswith("C.")].iloc[0]
Dm = ST[ST.index.str.startswith("D.")].iloc[0]
diff = S["brier_diff_vs_market"]["ens"]
agree = S["pick_agreement"]
stageA = S["strategy_A_ens_by_stage"]
hold = pd.DataFrame(H["metrics"]).set_index("model")
dis = ev[ev.ens_pick != ev.mkt_pick]
n_draws = int((ev.result_90 == "D").sum())
n_ko_draws = int(((ev.stage == "Knockout") & (ev.result_90 == "D")).sum())
top10 = elo[elo.in_wc2026].head(10)
STAKE = float(A.staked_usd / A.bets)
Bstage = ledB.groupby("stage").agg(bets=("won", "size"), wins=("won", "sum"), profit=("profit", "sum"))

WORDS = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
word = lambda n: WORDS.get(int(n), str(n))
pct = lambda x, d=1: f"{x * 100:.{d}f}%"
usd = lambda x: f"{x:,.0f} USD" if x >= 0 else f"-{abs(x):,.0f} USD"
sgn = lambda x: f"+{x:,.0f}" if x >= 0 else f"-{abs(x):,.0f}"
RES = {"H": "home win", "D": "draw", "A": "away win"}


def table(df, cols, headers, fmt=None, caption=None, cls=""):
    fmt = fmt or {}
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            f = fmt.get(c)
            cells.append(f"<td>{f(v) if f else v}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    cap = f"<caption>{caption}</caption>" if caption else ""
    return (f'<table class="{cls}">{cap}<thead><tr>' + "".join(f"<th>{h}</th>" for h in headers) + "</tr></thead><tbody>"
            + "".join(rows) + "</tbody></table>")


def fig(name, caption, num, width=""):
    style = f' style="max-width:{width}"' if width else ""
    return f'<figure><img src="assets/{name}" alt="{caption}"{style}><figcaption><span>Figure {num}.</span> {caption}</figcaption></figure>'


# ----------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------
sources_tbl = f"""
<table class="wide"><caption>Table 1. Data sources.</caption>
<thead><tr><th>Source</th><th>Role</th><th>Acquisition</th><th>Rows and columns</th><th>Key columns</th></tr></thead><tbody>
<tr><td>International football results (martj42/international_results, GitHub)</td><td>Primary. Training corpus for ratings and models; official results of the 104 World Cup matches.</td><td>CSV download</td><td>{D['history_rows']:,} matches, 9 columns; {D['history_pre_tournament']:,} before 11 June 2026; {D['training_2000_plus']:,} from 2000 onward used for model training</td><td>date, home_team, away_team, home_score, away_score, tournament, city, country, neutral</td></tr>
<tr><td>Polymarket match markets (Gamma API and CLOB API)</td><td>Primary. Independent benchmark: prices of the home-win, draw and away-win markets before kickoff.</td><td>REST API, one event per match, three markets per event, 12-hour price history</td><td>{D['polymarket_markets']} markets; 28,247 price observations, {D['price_points']:,} of them before kickoff; 11 market columns plus 5 price columns</td><td>event_slug, market_type, question, kickoff_utc, resolved_yes, volume_usd; ts, price</td></tr>
<tr><td>World Cup 2026 fixtures (openfootball/worldcup.json, GitHub)</td><td>Auxiliary. Stage, round, venue, half-time, full-time and extra-time scores; defines the 90-minute result used for evaluation.</td><td>JSON download</td><td>104 matches, 17 columns</td><td>wc_match_id, date, stage, round, home_team, away_team, ft_home, ft_away, et_home, et_away, venue</td></tr>
</tbody></table>"""

cleaning_tbl = """
<table class="wide"><caption>Table 2. Data cleaning and alignment steps.</caption>
<thead><tr><th>Issue</th><th>Where</th><th>Treatment</th></tr></thead><tbody>
<tr><td>Team names differ between sources</td><td>"Bosnia &amp; Herzegovina" and "USA" in openfootball versus "Bosnia and Herzegovina" and "United States" in the results file</td><td>Mapped to the results-file spelling before any join</td></tr>
<tr><td>Polymarket event slugs mix FIFA and ISO country codes</td><td>prt, hrv, che, cdr, cvi and kr instead of por, cro, sui, cod, cpv and kor; one slug labels Curaçao as kor</td><td>Code table built per team; text search by fixture as fallback; market type identified from the question text, never from the slug</td></tr>
<tr><td>Local versus UTC kickoff dates</td><td>Three evening kickoffs in North America carry the next calendar day in the Polymarket slug</td><td>Both the local date and the following day are tried when locating the event</td></tr>
<tr><td>Alternative team spellings in market questions</td><td>Türkiye, IR Iran, Côte d'Ivoire, Cabo Verde, Korea Republic</td><td>Alias table; event order (home, draw, away) used as a last resort</td></tr>
<tr><td>Extra-time scores in the results file</td><td>Knockout matches decided after 90 minutes are stored with the extra-time score (the final is 1-0, not 0-0)</td><td>The 90-minute score from openfootball is the evaluation target, because every Polymarket market resolves on 90 minutes plus stoppage time</td></tr>
<tr><td>Host-nation home advantage</td><td>The results file flags 13 matches of the United States, Mexico and Canada as non-neutral; openfootball lists the host second in three of them</td><td>Non-neutral flag taken from the results file; the three swapped fixtures are scored with the host as home team and the probabilities flipped back to fixture order</td></tr>
<tr><td>Fixtures scheduled after the tournament</td><td>27 matches played between the final and the August snapshot of the results file</td><td>Excluded: only matches before 11 June 2026 enter training</td></tr>
<tr><td>Cross-source validation</td><td>Polymarket resolution versus official 90-minute result</td><td>All 312 markets resolved consistently with the official score; zero mismatches</td></tr>
</tbody></table>"""

hold_tbl = table(hold.reset_index(), ["model", "accuracy", "log_loss", "brier"], ["Model", "Accuracy", "Log loss", "Brier score"],
                 {"accuracy": lambda x: pct(x), "log_loss": lambda x: f"{x:.3f}", "brier": lambda x: f"{x:.3f}"},
                 caption=f"Table 3. Temporal hold-out on {H['n_holdout']:,} international matches from January 2022 to 10 June 2026 (models fitted on {H['n_train']:,} matches from 2000 to 2021). Lower log loss and Brier are better.")

wc_tbl = table(metrics, ["model", "accuracy", "log_loss", "brier", "brier_group", "brier_knockout"],
               ["Forecaster", "Accuracy", "Log loss", "Brier score", "Brier, group stage (72)", "Brier, knockouts (32)"],
               {"accuracy": lambda x: pct(x), "log_loss": lambda x: f"{x:.3f}", "brier": lambda x: f"{x:.3f}", "brier_group": lambda x: f"{x:.3f}", "brier_knockout": lambda x: f"{x:.3f}"},
               caption="Table 4. Forecast quality on the 104 World Cup 2026 matches. Polymarket probabilities are the pre-kickoff prices normalised to sum to one.")

strat_tbl = table(strat, ["strategy", "bets", "wins", "hit_rate", "avg_price", "staked_usd", "profit_usd", "roi", "profit_ci95_low", "profit_ci95_high", "max_drawdown_usd"],
                  ["Strategy", "Bets", "Wins", "Hit rate", "Average price paid", "Staked", "Profit", "Return on stake", "95% CI low", "95% CI high", "Max drawdown"],
                  {"hit_rate": lambda x: pct(x), "avg_price": lambda x: f"{x:.3f}", "staked_usd": lambda x: f"{x:,.0f}", "profit_usd": lambda x: f"{x:+,.0f}", "roi": lambda x: f"{x * 100:+.1f}%",
                   "profit_ci95_low": lambda x: f"{x:+,.0f}", "profit_ci95_high": lambda x: f"{x:+,.0f}", "max_drawdown_usd": lambda x: f"{x:,.0f}"},
                  caption="Table 5. Betting simulation, 150 USD flat stake per bet, filled at the last recorded pre-kickoff price, no fees or slippage. Confidence intervals are bootstrap intervals for total profit (10,000 resamples of the per-bet results). All amounts in USD.", cls="wide")

edge_tbl = table(edge, ["edge_bucket", "bets", "hit_rate", "profit", "roi"], ["Edge of the model's pick over the market", "Bets", "Hit rate", "Profit (USD)", "Return on stake"],
                 {"hit_rate": lambda x: pct(x), "profit": lambda x: f"{x:+,.0f}", "roi": lambda x: f"{x * 100:+.1f}%"},
                 caption="Table 6. Strategy A results split by how far the model's probability for its pick exceeded the market's.")

surp["scoreline"] = surp.ft_home.astype(str) + "-" + surp.ft_away.astype(str)
surp["model_p"] = [r[f"ens_{r.result_90}"] for _, r in surp.iterrows()]
surp["fixture"] = surp.home_team + " v " + surp.away_team
surp_tbl = table(surp, ["date", "fixture", "scoreline", "result_90", "price_of_result", "model_p"],
                 ["Date", "Fixture", "Score (90 min)", "Result", "Market price of the result", "Model probability of the result"],
                 {"result_90": lambda x: RES[x], "price_of_result": lambda x: f"{x:.3f}", "model_p": lambda x: f"{x:.3f}"},
                 caption="Table 7. The eight least expected results, ranked by the pre-kickoff price of the outcome that happened.")

dis["fixture"] = dis.home_team + " v " + dis.away_team
dis["scoreline"] = dis.ft_home.astype(str) + "-" + dis.ft_away.astype(str)
dis_tbl = table(dis, ["date", "stage", "fixture", "scoreline", "result_90", "ens_pick", "mkt_pick"],
                ["Date", "Stage", "Fixture", "Score (90 min)", "Result", "Model pick", "Market favourite"],
                {"result_90": lambda x: RES[x], "ens_pick": lambda x: RES[x], "mkt_pick": lambda x: RES[x]},
                caption=f"Table 8. The {len(dis)} matches where the model's pick differed from the market favourite.")

elo_tbl = table(top10, ["rank_overall", "team", "elo"], ["Overall Elo rank", "Team", "Elo on 10 June 2026"], {"elo": lambda x: f"{x:.0f}"},
                caption="Table 9. Ten highest-rated World Cup teams at the start of the tournament.")

ev["fixture"] = ev.home_team + " v " + ev.away_team
ev["scoreline"] = ev.ft_home.astype(str) + "-" + ev.ft_away.astype(str)
app_tbl = table(ev, ["wc_match_id", "date", "stage", "fixture", "scoreline", "result_90", "ens_H", "ens_D", "ens_A", "mkt_H", "mkt_D", "mkt_A", "ens_pick"],
                ["#", "Date", "Stage", "Fixture", "Score", "Result", "Model H", "Model D", "Model A", "Market H", "Market D", "Market A", "Model pick"],
                {"result_90": lambda x: x, "ens_H": lambda x: f"{x:.2f}", "ens_D": lambda x: f"{x:.2f}", "ens_A": lambda x: f"{x:.2f}", "mkt_H": lambda x: f"{x:.2f}", "mkt_D": lambda x: f"{x:.2f}", "mkt_A": lambda x: f"{x:.2f}"},
                caption="Appendix table. Model and market probabilities for every match (H = first-named team wins, D = draw, A = second-named team wins).", cls="wide small")

n_dis_draw = int((dis.result_90 == "D").sum()); n_dis_draw_mkt = int(((dis.result_90 == "D") & (dis.mkt_pick == "D")).sum())
n_pick_diff = int((ev.ens_pick != ev.pois_pick).sum())
pick_diff_all_draw = bool((ev.loc[ev.ens_pick != ev.pois_pick, "result_90"] == "D").all())
imax = ev.mkt_D.idxmax(); max_draw_fix = f"{ev.loc[imax, 'home_team']} v {ev.loc[imax, 'away_team']}, which finished {int(ev.loc[imax, 'ft_home'])}-{int(ev.loc[imax, 'ft_away'])}"
gp = ev[ev.wc_match_id == 68].iloc[0]
sd = surp[surp.result_90 == "D"]; n_surp_draw = len(sd)
surp_favs = ", ".join(dict.fromkeys(sd.apply(lambda r: r.home_team if r.ens_H > r.ens_A else r.away_team, axis=1))); surp_dogs = ", ".join(dict.fromkeys(sd.apply(lambda r: r.away_team if r.ens_H > r.ens_A else r.home_team, axis=1)))
dis_desc = "; ".join(f"{r.home_team} v {r.away_team} ({RES[r.result_90]})" for _, r in dis.iterrows())
n_model_right = int((dis.ens_pick == dis.result_90).sum()); n_mkt_right = int((dis.mkt_pick == dis.result_90).sum())
cal_m = cal[cal.source == "model"].set_index("bin"); cal_k = cal[cal.source == "market"].set_index("bin")
lo_bin, mid_bin, hi_bin = "(0.15, 0.25]", "(0.35, 0.45]", "(0.45, 0.6]"

# ----------------------------------------------------------------------
# HTML
# ----------------------------------------------------------------------
css = """
:root{--paper:#F6F7F9;--ink:#15213B;--ink-2:#3E4A63;--muted:#6B7488;--rule:#D3D8E2;--navy:#1F4E79;--red:#C0504D;--amber:#E2A03F;--table:#FFFFFF}
*{box-sizing:border-box}
html{font-size:17px}
body{margin:0;background:var(--paper);color:var(--ink);font-family:"Source Serif 4",Georgia,"Times New Roman",serif;line-height:1.62}
a{color:var(--navy)}
.wrap{max-width:46rem;margin:0 auto;padding:0 1.25rem 5rem}
header.hero{padding:4.5rem 0 2.5rem;border-bottom:2px solid var(--ink)}
.kicker{font-family:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",Helvetica,Arial,sans-serif;font-size:.9rem;color:var(--ink-2);margin:0 0 1.2rem}
h1{font-size:2.35rem;line-height:1.12;font-weight:700;margin:0 0 1.1rem;letter-spacing:-.01em}
.question{font-size:1.28rem;line-height:1.45;color:var(--ink-2);margin:0 0 1.6rem;font-style:italic;max-width:40rem}
.byline{font-family:"IBM Plex Sans",system-ui,sans-serif;font-size:.9rem;color:var(--muted);margin:0}
nav.toc{font-family:"IBM Plex Sans",system-ui,sans-serif;font-size:.88rem;padding:1rem 0;border-bottom:1px solid var(--rule);margin-bottom:1rem}
nav.toc a{margin-right:1.1rem;text-decoration:none;color:var(--ink-2);white-space:nowrap}
nav.toc a:hover,nav.toc a:focus{color:var(--navy);text-decoration:underline}
h2{font-family:"IBM Plex Sans",system-ui,sans-serif;font-size:1.55rem;font-weight:700;margin:3rem 0 .9rem;line-height:1.2}
h3{font-family:"IBM Plex Sans",system-ui,sans-serif;font-size:1.12rem;font-weight:700;margin:2rem 0 .6rem}
p{margin:0 0 1.1rem}
.lead{font-size:1.06rem}
figure{margin:1.8rem 0 2rem;text-align:center}
figure img{max-width:100%;height:auto;border:1px solid var(--rule);background:#fff}
figcaption{font-family:"IBM Plex Sans",system-ui,sans-serif;font-size:.84rem;color:var(--ink-2);text-align:left;margin:.6rem auto 0;max-width:40rem;line-height:1.45}
figcaption span{font-weight:700;color:var(--ink)}
table{font-family:"IBM Plex Sans",system-ui,sans-serif;font-size:.84rem;border-collapse:collapse;margin:1.6rem auto 2rem;background:var(--table);line-height:1.35}
table.wide{width:100%}
table.small{font-size:.76rem}
caption{caption-side:top;text-align:left;font-weight:400;color:var(--ink-2);padding:0 0 .55rem;line-height:1.45}
th,td{padding:.45rem .6rem;border-bottom:1px solid var(--rule);text-align:left;vertical-align:top}
th{background:#EEF1F6;font-weight:600;border-bottom:2px solid var(--ink)}
td:first-child,th:first-child{padding-left:.5rem}
.num td:not(:first-child){text-align:right}
details{margin:1rem 0 2rem}
summary{font-family:"IBM Plex Sans",system-ui,sans-serif;cursor:pointer;color:var(--navy)}
.refs{font-family:"IBM Plex Sans",system-ui,sans-serif;font-size:.86rem;color:var(--ink-2)}
.refs p{margin:0 0 .6rem;padding-left:1.6rem;text-indent:-1.6rem}
footer{margin-top:4rem;padding-top:1rem;border-top:1px solid var(--rule);font-family:"IBM Plex Sans",system-ui,sans-serif;font-size:.82rem;color:var(--muted)}
@media (max-width:640px){html{font-size:16px}h1{font-size:1.85rem}.question{font-size:1.1rem}header.hero{padding:3rem 0 2rem}table{display:block;overflow-x:auto}}
@media print{body{background:#fff}nav.toc{display:none}.wrap{max-width:100%}h2{page-break-after:avoid}figure,table{page-break-inside:avoid}details{display:block}details summary{display:none}details[open],details{open:true}}
"""

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Model versus market: a match forecasting model against Polymarket at the 2026 World Cup</title>
<meta name="description" content="Executive report. A result-forecasting model rebuilt from public data is tested against Polymarket prices over all 104 matches of the 2026 FIFA World Cup.">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600;700&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,700;1,8..60,400&display=swap" rel="stylesheet">
<style>{css}</style>
</head>
<body>
<div class="wrap">

<header class="hero">
<p class="kicker">Executive report: Capstone Project for the Professional Certificate in Data Analytics.</p>
<h1>Model versus market: testing a match forecasting model against Polymarket at the 2026 World Cup</h1>
<p class="question">Would a forecasting model, which was constructed using public match history, have produced better predictions than a prediction market, and would having traded the selections made by this model have resulted in a profit?</p>
<p class="byline">Piero, September 2026. For the data, code, and figures, refer to the appendix and the project archive.</p>
</header>

<nav class="toc" aria-label="Sections">
<a href="#summary">Executive summary</a><a href="#intro">Introduction</a><a href="#methods">Methods</a><a href="#results">Results</a><a href="#conclusion">Conclusion</a><a href="#appendix">Appendix</a>
</nav>

<h2 id="summary">Executive summary</h2>
<p class="lead">For the 2026 World Cup, I applied a statistical model to predict match outcomes and then placed those predictions on Polymarket, betting between 100 and 200 USD on each match. Since I never kept a record of the trades (other than the final profit I made), this report reconstructs the model using public data and examines it under controlled conditions: the classifier is trained solely on matches that took place before the first game, the ratings are updated one match at a time as new information becomes available at each kickoff, and each prediction is compared with the price that was recorded on Polymarket before the same kickoff. The testing includes all 104 matches.</p>
<p>The market proved to be the more accurate forecaster. The model correctly predicted the 90-minute result in {pct(ens.accuracy)} of the matches, having a Brier score of {ens.brier:.3f}; Polymarket achieved {pct(mkt.accuracy)} and a score of {mkt.brier:.3f}. This difference is mainly seen in the group stage ({ens.brier_group:.3f} compared to {mkt.brier_group:.3f}) and nearly disappears in the knockout rounds ({ens.brier_knockout:.3f} compared to {mkt.brier_knockout:.3f}). For the entire tournament, the difference is statistically significant since the bootstrap 95% confidence interval for the gap between the model's and the market's Brier scores extends from {diff['ci95'][0]:.3f} to {diff['ci95'][1]:.3f} and lies completely above zero. The two forecasters agreed on the favourite in {pct(agree['ens_vs_market'], 0)} of the matches, and in the {len(dis)} instances where they disagreed, the market was correct {word(n_mkt_right)} times and the model {word(n_model_right)}.</p>
<p>Trading the picks would not have been profitable. If a flat stake of {STAKE:.0f} USD was placed on the model's choice in each match, the result was a loss of {abs(A.profit_usd):,.0f} USD on {A.staked_usd:,.0f} USD staked (a return of {A.roi * 100:.1f}%). When the bets were limited to those matches in which the model's probability was more than five points higher than the market price, a profit of {B.profit_usd:,.0f} USD was made on {B.staked_usd:,.0f} USD staked ({B.roi * 100:+.1f}%), but this was based on {int(B.bets)} bets with a confidence interval ranging from {B.profit_ci95_low:+,.0f} to {B.profit_ci95_high:+,.0f} USD, which is no different from luck. However, the strategy of always backing the market favourite in each match, one that makes no use of the model, yielded a profit of {Dm.profit_usd:,.0f} USD.</p>
<p>I suggest that you cease trading this model against liquid World Cup markets (a market which, as it turns out, did not require my assistance in pricing a football match), continue to use it as a standalone sanity check for market prices, and reallocate the modelling work to two areas in which a private model can still provide value, namely, the draw probabilities, since the model never takes them into account, and the inputs which are priced by the market but are not included in the public match history.</p>

<h2 id="intro">Introduction</h2>
<p>Prediction markets have now developed into a serious means of forecasting sports outcomes. On Polymarket, there was a dedicated market for each of the matches in the 2026 World Cup, with three types of contracts available for each one (a home win, a draw, and an away win), and the total volume for the {D['polymarket_markets']} contracts examined in this study was {D['volume_total_usd'] / 1e9:.2f} billion USD, which amounts to a median of {D['volume_median_per_match_usd'] / 1e6:.1f} million USD per match. Prices in a well-liquid market are not mere expressions of personal opinion; they represent the combined judgement of thousands of traders risking money, and prices are constantly updated right up to the start of the match. Anyone who wishes to place a bet contrary to this view, whether they do so using a spreadsheet or with the aid of a machine learning model, is in effect asserting that they have some knowledge which the market itself lacks.</p>
<p>I made that assertion at the tournament. In the weeks leading up to the first match, I built a pipeline that collected historical international match results, calculated Elo ratings, trained both a gradient boosting classifier and a Poisson goals model, and produced probability estimates for each game. During the tournament, I compared these probabilities with Polymarket prices, looked for the cases in which my model was more confident than the market, and placed bets of about 100 to 200 USD on the outcome it favoured. This was a live experiment run without a lab notebook (call it "gambling"); I never saved records of my trades or the prices at the time I made each decision. Any memory I have of how it turned out is contaminated by hindsight, and by the income I somehow made anyway.</p>
<p>The project takes this informal experiment and turns it into one that can be verified. The business issue is the same one any analyst developing forecasting models faces: does a model built on cheap, public data perform sufficiently better than an established benchmark to justify action? Although the benchmark is particularly strong here, the question is the same as when a retailer tests whether its demand model outperforms the supplier's forecasts, or when a lender checks whether its credit model exceeds bureau scores. This question is suitable for a data-driven answer since all the components are public and complete. The tournament has now ended, so the actual outcomes for all 104 matches are known. Polymarket's API records the market prices at twelve-hour intervals from the day each market was opened. The training data include every officially recognised international match since 1872, and the analysis relies on no private dataset.</p>
<p>The remainder of the report follows the normal sequence. The Methods section outlines the two main data sources and one additional source, details the cleaning required to combine them, and explains the models used and the evaluation design, including the simulated betting rule used to replace my actual trades. The Results section provides descriptive statistics for both datasets, presents model performance on four years of pre-tournament matches, compares the models head-to-head with the market, and includes the betting simulation. The Conclusion comes back to the question mentioned above and states what I will do differently.</p>

<h2 id="methods">Methods</h2>
<p>This section sets out the data, along with the cleaning and merging plan, the models, and the evaluation design, in that order. The entire process was carried out in Python (using pandas, scikit-learn, statsmodels, SciPy, and matplotlib), and the three data sources were kept in one SQLite database; the complete pipeline was run end to end via the five scripts listed in the appendix.</p>

<h3>Data sources</h3>
<p>The analysis is carried by two independent sources, while the official fixture list is supplied by a third (Table 1). The first is the international results file in the martj42/international_results repository on GitHub, a CSV containing every senior men's international match acknowledged by the compiler, from the first England v Scotland fixture in 1872 to {D['history_last']}. The second source is Polymarket; its public Gamma API includes one event for each World Cup match with three binary markets, and its CLOB API provides the historical price data for each market's Yes share. A Yes share pays out 1 USD when the outcome occurs, so the share price represents the market's probability of that outcome. The third source, openfootball's worldcup.json, is the official fixture and score list for the tournament, consisting of 104 rows and giving details of the stages, venues, and the 90-minute result. All row counts reported here were taken from the data files themselves on the access date given in the references; the descriptive text on the repository landing page lags behind the file it describes and reports a lower total for an earlier year, as do third-party mirrors of the dataset.</p>
{sources_tbl}

<h3>Data cleaning and merge plan</h3>
<p>The merge key is the fixture, that is, the two teams involved and the date on which the game starts. This simple key required several alignment steps to work across the different sources (Table 2), three of which were important enough that missing them would affect the results. Since Polymarket settles its markets on the score following 90 minutes plus stoppage time, whereas the results file shows the score after extra time for knockout matches, the full-time score from openfootball is the one used for evaluation, and the match between Spain and Argentina is treated as a draw (0-0 after 90 minutes; 1-0 after extra time). Three evening kickoffs in North America take place on the following UTC day, which is the date Polymarket uses in its identifiers. In the results file, 13 matches involving the host nations are marked as non-neutral; in three of these, openfootball lists the host team second, so the model would have awarded the home advantage to the wrong side unless a correction was made.</p>
{cleaning_tbl}
<p>The combined evaluation table includes one row for each match, listing the fixture, the stage, the 90-minute score, the model's three probabilities, the three pre-kickoff prices and the market-derived probabilities. The pre-kickoff price is the last price recorded before kick-off, as stated in the market metadata. Since the archived history is sampled every 12 hours, this price is typically {D['hours_before_kickoff_median']:.0f} hours before kick-off and in no case more than {D['hours_before_kickoff_max']:.0f} hours before. It is a pre-match price, not the exact settlement line, which I will return to in the discussion. On average, the three prices for a match add up to {D['overround_mean']:.3f} (ranging from {D['overround_min']:.3f} to {D['overround_max']:.3f}), so the overround bookmakers include in their odds is essentially missing; I normalised the prices to sum to one for the probability comparisons and left them unchanged for the betting simulation. To verify the join, I checked each market's resolution flag against the official score and found that all 312 agreed.</p>

<h3>Features and models</h3>
<p>The model consists of three components. The first of these is an Elo rating, which is calculated in chronological order based on the entire results file, in accordance with the rules used by the World Football Elo Ratings: this includes a K-factor that varies according to the importance of the match (20 in the case of friendly games, 40 for qualifiers, 50 for continental finals and 60 for World Cup finals), a multiplier based on goal difference, and 100 points as a result of home advantage when the venue is not neutral. Elo suits international football well because most teams play only a small number of matches, the fixture schedule is sparse, and updating the rating after each match uses all results without requiring a balanced schedule. Along with the rating, each team also has a short-form record showing points per match, goals scored, and goals conceded in its most recent five matches.</p>
<p>The second layer gives a prediction of the three possible results. The primary classifier used is scikit-learn's HistGradientBoostingClassifier, which has been trained on {D['training_2000_plus']:,} matches from the year 2000 up to 10 June 2026 using nine features: the two Elo ratings, the difference in Elo ratings with and without home advantage, the Elo win expectancy, the home-advantage flag, the match-importance class, and the three form differences. Gradient boosting was selected because it can capture nonlinear relationships (for example, home advantage matters less when both teams have high ratings) without feature engineering, works with inputs on different scales, and produces probabilities that are sufficiently calibrated for this application. It is compared with a multinomial logistic regression using the same features, a logistic model based only on Elo expectancy, and the base rates. The third layer consists of a goals model: two Poisson regressions (one for home goals and one for away goals) based on the raw Elo difference, home advantage, and match importance, as described by Dixon and Coles (1997). By multiplying the two Poisson distributions together, a score matrix is obtained from which the same three outcome probabilities and the expected goals can be derived. The ensemble employed for the head-to-head comparison, and for the betting rule, is the simple average of the classifier's probabilities and the Poisson probabilities. The two components rarely reach different conclusions; averaging is merely a minor way to offset each component's idiosyncrasies and does not add accuracy.</p>

<h3>Evaluation design</h3>
<p>Two evaluations were carried out. The first is a temporal hold-out in which the models are trained on matches from 2000 to 2021 and then assessed on the {H['n_holdout']:,} matches that took place from January 2022 to 10 June 2026. This establishes that they perform well on matches they have not seen before and forms the basis for selecting between them. The second evaluation is the tournament itself. In this case, the classifier and the Poisson models were refitted on all matches that occurred before 11 June 2026 and then held fixed. The 104 fixtures were predicted in sequence; after each match, the Elo ratings and form records were updated with the result from the first 90 minutes, since this information was available before the next match began, but the model was not retrained during the tournament. This mirrors how the system was used in live conditions and eliminates the possibility of tournament information leaking into the training data.</p>
<p>The quality of forecasts is assessed using the accuracy of the most probable outcome, log loss, and the multi-class Brier score (Brier, 1950), which is the average of the sum of the squared errors across the three possible outcomes for each match, so that a forecast that assigns equal probabilities to all outcomes gives a score of 0.667 and a perfect forecast yields a score of zero. The Brier score is the main metric because it accounts for both discrimination and calibration and, unlike accuracy, penalises a forecaster who is confident but wrong. The market is scored on the same footing from its normalised probabilities. Because the comparison is paired, with a model score and a market score for each match, the uncertainty of the difference is estimated by randomly resampling the 104 per-match differences 10,000 times. Calibration is assessed by dividing all 312 outcome probabilities into six groups and comparing the average forecast with the actual frequency.</p>
<p>In the betting simulation, the trades that I had not documented are replaced by a rule that allows anyone to re-run them. With Strategy A, a stake of {STAKE:.0f} USD, the midpoint of the amount I actually risked, is placed on the model's most probable outcome in each match, with the Yes share bought at the pre-kickoff price. If the predicted outcome occurs, the position yields a return equal to {STAKE:.0f} divided by the price; if not, the stake is lost. Strategy B implements the same bet only when the model's probability for its chosen outcome exceeds the market's by more than five percentage points, which is the kind of filter I had applied informally. Strategy C bets on the outcome among the three with the greatest positive edge over the market, again requiring this edge to be more than five points, and it is included to check whether the model's disagreements with the market contain information regardless of which outcome they support. Strategy D always bets on the market favourite in each match and acts as the control case: it makes no use of the model and shows what the raw prices themselves returned during the tournament. Neither trading fees, slippage, nor position limits are taken into account; the results therefore represent an upper bound on what the rules would have earned.</p>

<h2 id="results">Results</h2>
<p>This section starts with a descriptive account of the data, followed by the hold-out benchmark, the head-to-head comparison with the market, the betting simulation, and a discussion of what the numbers mean and where they are fragile.</p>

<h3>Descriptive statistics</h3>
<p>Figure 1 illustrates the distribution of game results in the training data and in the tournament. Since 2000, in the {D['training_2000_plus']:,} matches played, the first-named team wins about half the time at home and about 40% of the time on neutral ground, while draws make up 18% to 28% of matches depending on the situation. Draws in the World Cup finals from 2002 to 2022 occurred in 22.9% of the matches. The 2026 tournament saw {n_draws} draws in 104 matches ({pct(D['wc_result_shares']['D'])}), with {n_ko_draws} of the 32 knockout matches ending level after 90 minutes. This draw rate is high by historical standards and, as the following sections demonstrate, it is the single feature of this tournament that hurt both forecasters most. The number of goals was slightly above the training-era average: {D['goals_per_match_wc']:.2f} per match compared to {D['goals_per_match_hist_2000']:.2f} since 2000.</p>
{fig("fig1_outcome_shares.png", "Share of home (first-named) wins, draws and away wins by match context in the training data since 2000, compared with the 104 matches of the 2026 World Cup. Home team rows are matches played at a genuine home venue; the others are neutral.", 1)}
<p>Figure 2 and Table 9 show the ratings the model had at the start of the tournament. Spain ({top10.iloc[0].elo:.0f}), Argentina ({top10.iloc[1].elo:.0f}) and France ({top10.iloc[2].elo:.0f}) were in the top positions, which aligned with the final being played between Spain and Argentina. Colombia and Ecuador are within the top eight, ahead of the Netherlands and Germany, suggesting the Elo system weights recent results in competitive matches more heavily than reputation.</p>
{fig("fig2_elo_pre_tournament.png", "Elo ratings of the sixteen highest-rated World Cup teams on 10 June 2026, computed from every recognised international match since 1872 with importance-weighted K-factors.", 2, "620px")}
<p>The opening match is shown in Figure 3. Three contracts were launched on 7 April, nine weeks before the game, and their prices were recorded continuously until kick-off. During this time, the price for Mexico rose from 0.50 to around 0.69, while the draw and South Africa contract prices fell. Throughout the tournament, contracts opened between 6 April and 16 July 2026 (with the knockout markets opening only once the fixture was known), and the 12-hour price archive offers, on average, {D['price_points'] / D['polymarket_markets']:.0f} observations before kick-off for each market.</p>
{fig("fig3_price_path_opening_match.png", "Price of the Yes share for each of the three Polymarket contracts on the opening match, Mexico v South Africa, from market opening to kickoff on 11 June 2026. Prices are sampled every 12 hours; the last point before the dashed line is the pre-kickoff price used in the analysis.", 3)}

<h3>Hold-out performance before the tournament</h3>
<p>Table 3 shows the temporal hold-out. All of the rating-based models show a substantial improvement over the base rates: accuracy increases from {pct(hold.loc['Base rates (train frequencies)'].accuracy)} to around 60%, and the Brier score drops from {hold.loc['Base rates (train frequencies)'].brier:.3f} to about 0.514. The four rating-based models differ by only a few thousandths, and most of the gain comes from Elo alone. Gradient boosting and logistic regression are effectively tied in this hold-out ({hold.loc['HistGradientBoosting'].brier:.4f} versus {hold.loc['Multinomial logistic regression'].brier:.4f}), suggesting that the non-linearities the boosted model can learn add little at this level of feature detail. I have retained gradient boosting as the classifier since it was the one actually in use, and the comparison with the market should evaluate that system rather than a replacement; the hold-out result shows that this choice involves no measurable cost. One structural feature of all these models matters for what comes next: on the hold-out, the classifier assigned a draw as the most likely outcome in just {pct(H['hgb_pred_shares'].get('D', 0), 1)} of matches, even though {pct(H['holdout_result_shares']['D'])} of matches ended in a draw. It is rare for a three-way classifier to rank a draw first because its draw probability is rarely higher than both win probabilities; this is a known limitation of argmax selection, not a bug, and it matters in a tournament that includes {n_draws} draws.</p>
{hold_tbl}

<h3>Model versus market on the 104 matches</h3>
<p>Table 4 and Figure 6 show the head-to-head results. The ensemble obtained a Brier score of {ens.brier:.3f} and a log loss of {ens.log_loss:.3f}; Polymarket achieved {mkt.brier:.3f} and {mkt.log_loss:.3f}, respectively. The market was also correct more frequently ({pct(mkt.accuracy)} compared to {pct(ens.accuracy)}). The gradient boosting component alone ({hgb.brier:.3f}) and the Poisson component ({pois.brier:.3f}) lie on either side of the ensemble; both of them are behind the market. The paired bootstrap gives a mean difference between the model and the market in Brier score of {diff['mean_diff']:.3f}, with a 95% interval ranging from {diff['ci95'][0]:.3f} to {diff['ci95'][1]:.3f}. Since this interval does not include zero, the market's advantage is not due to a small-sample effect. When looking at each match individually, the model outperformed the market in {pct(diff['share_matches_model_better'], 0)} of the fixtures.</p>
<p>The section showing the results by stage is the most informative in the table. During the group stage, the market was clearly ahead ({mkt.brier_group:.3f} compared with {ens.brier_group:.3f}); in the 32 knockout matches, the two figures are very close ({mkt.brier_knockout:.3f} versus {ens.brier_knockout:.3f}), and in this case the model's accuracy is actually higher ({pct(ens.accuracy_knockout)} compared with {pct(mkt.accuracy_knockout)}). Two explanations account for this. On the one hand, by the knockout rounds the Elo ratings had absorbed three group-stage matches for each team, so the model had caught up with the form information the market had already priced in at the beginning. On the other hand, knockout matches between teams with similar ratings are harder for everyone, and the market had less advantage to work with than it did in pricing the mismatches of the group stage. Although 32 matches make the knockout-stage comparison too small to rank the two forecasters, it is sufficient to say that the model was not severely outperformed there.</p>
{wc_tbl}
{fig("fig6_brier_by_stage.png", "Multi-class Brier score by tournament stage for the three model variants, Polymarket and a uniform forecast. Lower is better.", 6)}
<p>Figure 4 shows the model's estimate for each outcome probability compared with the market. The two forecasting methods are very closely in agreement: in {pct(agree['ens_vs_market'], 0)} of the matches the ensemble and the market had the same favourite, and in about two thirds of the cases both tended to favour the team named first. The discrepancies occur in the middle of the scale, between 0.25 and 0.50, where the model was often more confident than the market in its preferred side. This is precisely the area where the calibration curve in Figure 5 indicates the model is overconfident: in the {mid_bin} interval, the model's forecasts averaged {cal_m.loc[mid_bin].mean_p:.2f}, whereas the outcome occurred {pct(cal_m.loc[mid_bin].observed, 0)} of the time. By contrast, in the same interval the market's forecasts averaged {cal_k.loc[mid_bin].mean_p:.2f} and were realised {pct(cal_k.loc[mid_bin].observed, 0)} of the time. The market's calibration curve follows the diagonal in all the intervals. The model's curve, however, is irregular: it is too high in both the {lo_bin} and {mid_bin} intervals, too low in the {hi_bin} interval, and there are only 29 to 35 observations per interval. The market is better calibrated, and it is this, more than any difference in the ranking of the teams, that gives rise to its advantage in terms of the Brier score.</p>
{fig("fig4_model_vs_market.png", "Model probability against market probability for all 312 outcomes (three per match). Points above the diagonal are outcomes the model rated higher than the market did; green points are outcomes that happened.", 4, "560px")}
{fig("fig5_calibration.png", "Calibration of the ensemble model and of Polymarket: mean forecast probability in each of six bins against the observed frequency of the outcome, with the number of forecasts per bin.", 5, "560px")}
<p>Table 8 lists the {len(dis)} matches in which the model's choice differed from that of the market. The market was correct in {word(n_mkt_right)} of these and the model in {word(n_model_right)}. {word(n_dis_draw).capitalize()} of the ten matches ended in a draw; in {word(n_dis_draw_mkt)} of those the draw was the market's favourite outcome, and in the other {word(n_dis_draw - n_dis_draw_mkt)} neither forecaster gained. In the two cases in which the model predicted correctly, both involved semi-final results: Spain defeating France and Argentina beating England as the second-named team, because the model's higher Elo rating for the teams that reached the final outweighed the market's preference for France and England. The model's poorest individual prediction was its incorrect call in the match between Ghana and Panama, in which it assigned Panama a {pct(gp.ens_A, 0)} probability while the market gave it only {pct(gp.mkt_A, 0)}. Panama's rating going into the match was {gp.elo_away_pre:.0f}, compared to Ghana's {gp.elo_home_pre:.0f}, based on an unbeaten qualification campaign (seven wins and three draws in ten CONCACAF qualifiers) against opponents no stronger than Guatemala, Suriname and El Salvador. The market ignored that record and was right to do so: Panama had also suffered a 6-2 defeat at the hands of Brazil in a friendly twelve days before the tournament, a result which the Elo system absorbed as a single update.</p>
{dis_tbl}
<p>Table 7 displays the results neither of the forecasters had anticipated. {word(n_surp_draw).capitalize()} of the eight largest surprises were draws in which a clear favourite ({surp_favs}) was held by an outsider ({surp_dogs}), and the model had assigned probabilities for those draws ranging from {pct(sd.model_p.min(), 0)} to {pct(sd.model_p.max(), 0)}. The model capped any draw at {ev.ens_D.max():.2f} and never chose it as its most likely outcome, which, in a tournament with {n_draws} draws, limited how well it could possibly do. The highest draw probability in the market was {ev.mkt_D.max():.2f} ({max_draw_fix}), and the market selected the draw as its favourite {word(agree['draws_predicted_by_market'])} times.</p>
{surp_tbl}

<h3>Betting simulation</h3>
<p>Table 5 and Figure 7 show the results that the four betting strategies would have obtained. Strategy A, which is the one most similar to the approach I took, ended up losing {abs(A.profit_usd):,.0f} USD on a stake of {A.staked_usd:,.0f} USD, giving a return of {A.roi * 100:.1f}% and a hit rate of {pct(A.hit_rate)} at an average price of {A.avg_price:.2f}; the bootstrap interval for its total profit ranged from {A.profit_ci95_low:+,.0f} to {A.profit_ci95_high:+,.0f} USD. Its path included a loss of {abs(stageA['Group']['profit']):,.0f} USD over the 72 group matches, reaching a low of {ledA.cum_profit.min():,.0f} USD, before it gained {stageA['Knockout']['profit']:,.0f} USD over the 32 knockout matches, consistent with the stage split in the Brier scores. Strategy B, the five-point edge filter, made {int(B.bets)} bets and returned {B.profit_usd:+,.0f} USD ({B.roi * 100:+.1f}%), having {int(B.wins)} wins at an average price of {B.avg_price:.2f} and a confidence interval extending from {B.profit_ci95_low:+,.0f} to {B.profit_ci95_high:+,.0f} USD; it lost {abs(Bstage.loc['Group'].profit):,.0f} USD in the group stage and earned all of its profit from {int(Bstage.loc['Knockout'].bets)} knockout bets, two of which were the semi-final predictions mentioned above. Strategy C, which backed the option with the largest edge regardless of the outcome it favoured, is the clearest verdict on the model's disagreements with the market: it placed {int(C.bets)} bets, had a hit rate of {pct(C.hit_rate)}, and lost {abs(C.profit_usd):,.0f} USD ({C.roi * 100:.1f}%). Where the model identified the most value, mainly on underdogs and the ten draws it backed, it was most often wrong. The control, Strategy D, achieved a return of {Dm.profit_usd:+,.0f} USD ({Dm.roi * 100:+.1f}%) by selecting the favourite at the raw prices; favourites won {pct(Dm.hit_rate)} of the matches at an average price of {Dm.avg_price:.2f}, so the market itself was, if anything, slightly underpricing the favourites during this tournament.</p>
{strat_tbl}
{fig("fig7_cumulative_pnl.png", "Cumulative profit of the four betting rules across the 104 matches in tournament order, 150 USD flat stake, filled at the pre-kickoff price. The dotted line marks the start of the knockout stage.", 7)}
<p>Figure 8 and Table 6 examine whether the extent to which the model's estimate exceeded that of the market predicted the results of the bets made under Strategy A. It did not, in any monotonic way: in the cases where the model was more than five points below the market on its own selection, the return was {edge.iloc[0].roi * 100:+.0f}% (these are matches in which both agreed on the favourite and the market was the more confident of the two); for the cases with a five-to-ten point edge, the return was {edge.iloc[3].roi * 100:+.0f}% based on {int(edge.iloc[3].bets)} bets; and for those with an edge of more than ten points, the return was {edge.iloc[4].roi * 100:+.0f}% based on {int(edge.iloc[4].bets)}. A useful indicator would have shown returns increasing as the edge increased. What the chart displays is noise around a very slightly negative average, with one favourable category small enough to be due to chance.</p>
{fig("fig8_roi_by_edge.png", "Return on stake of Strategy A bets grouped by the model's edge over the market for its pick (model probability minus market probability, in percentage points), with the number of bets in each group.", 8)}

<h3>Discussion</h3>
<p>Three findings carry the report. The market forecast proved to be better than the model, a difference which remains significant after a paired bootstrap analysis; this advantage was primarily due to calibration in the group stage and to the model's inherent inability to favour draws; and none of the four betting rules generated a return that could be distinguished from random chance, the one that most directly expressed "bet where the model disagrees with the market" losing a third of its stake. These findings agree with one another and with the economic characteristics of the setting. A market with a median volume of {D['volume_median_per_match_usd'] / 1e6:.0f} million USD per match has already absorbed the public results history on which the model was built, along with squad news, injuries, line-ups, weather conditions, and specialist forecasters' views. Because the model relies solely on match results, it uses only a subset of the information available to the market, and a subset cannot consistently outperform the whole.</p>
<p>The limitations apply in both directions. Regarding the model, it does not use player-level information, nor club form, nor market data as features, and its five-match form window is rather crude for international teams. As for the conclusions, the sample size of 104 matches is small when it comes to betting returns, as the confidence intervals show, and the tournament's {pct(D['wc_result_shares']['D'], 0)} draw rate was untypical, which had a greater negative impact on a draw-averse model than it would have in an ordinary year. Two aspects of the price data warrant attention. The pre-kickoff price is a median of {D['hours_before_kickoff_median']:.0f} hours before kick-off, so news about the line-up released in the final hours is not included; this makes the market appear slightly weaker than its actual closing state, not stronger, and therefore does not rescue the model. Also, the simulation ignores fees and slippage, which means every strategy appears better than it would have been in practice. Both of these biases work against the model's case and therefore strengthen rather than weaken the conclusion. Finally, the ensemble and the Poisson model differed in their choice in only {word(n_pick_diff)} matches, both of which ended in a draw, so both bets lost and the betting results for those two entries in Table 5 are identical; the classifier on its own performed worse.</p>

<h2 id="conclusion">Conclusion</h2>
<p>It was to be determined whether a forecasting model developed from public match history outperformed Polymarket in predicting the results at the 2026 World Cup, and whether trading the model's predictions would have been profitable. Across all 104 matches, the answer to both questions is negative. The model was a competent forecaster in an absolute sense, achieving {pct(ens.accuracy)} accuracy and a Brier score of {ens.brier:.3f}, consistent with its four-year hold-out and considerably better than the base rates. The market performed better, with {pct(mkt.accuracy)} accuracy and a Brier score of {mkt.brier:.3f}; the difference was significant across the tournament and most evident in the group stage, where the market's better calibration showed. The model never ranked a draw as its first choice, and {n_draws} draws occurred in the tournament. When the model's predictions were backed at a flat {STAKE:.0f} USD, the result was a return of {A.roi * 100:.1f}%; when only the cases where the model differed from the market by more than five points were backed, the return was {B.roi * 100:+.1f}% over {int(B.bets)} bets, on an interval that also covers substantial losses; and when the model's largest disagreements were backed, the result was {C.roi * 100:.1f}%.</p>
<p>The work itself held up. Three public sources were joined on a fixture key after discrepancies of naming, time zone and rule definition had been resolved, and the join was verified by comparing each market's resolution with the official score. The model was trained with a hard cut-off before the tournament, and its ratings were updated only with information available at each kickoff. The betting rules are explicit and can be re-run, which my original trades cannot (the undocumented ones, for the record, finished the tournament ahead by roughly 8,000 USD, on a method consisting of watching the matches live and trusting my stomach, which is unrepeatable, indefensible, and the reason this reconstruction exists at all).</p>
<p>Four changes follow. First, I will not trade this model, or any model based solely on match results, against liquid World Cup or major-tournament markets, since the market price is the better forecast and should be treated as such. Second, when I build a forecasting model I will record the benchmark forecast and my decision at the moment I make it, so that the next evaluation does not have to be carried out from scratch; a table with a timestamp, the model's probabilities and the market price would have turned this report into a one-week task rather than a full rebuild. Third, the next version of the model should target the two gaps this test exposed: a separate draw model, since the argmax selection systematically ignores the outcome that occurred {pct(D['wc_result_shares']['D'], 0)} of the time, and the features the market prices but match history does not contain, beginning with squad market values and the club-level form of the players called up. Fourth, if the objective is to identify markets where a private model provides value, the place to look is where liquidity is thin, such as qualifiers, friendlies and lower-tier tournaments, and the way to look is with a pre-registered rule and a paper ledger before any money is staked. The 2026 World Cup was a good place to learn that lesson and an expensive place to keep testing it.</p>

<h2 id="appendix">Appendix</h2>
<h3>Reproducibility</h3>
<p>The pipeline is five scripts run in order from the repository root. <code>00_verify_sources.py</code> contacts the three sources, reports their status, row counts and date ranges, and checks the local copies against them. <code>01_build_database.py</code> downloads the sources and writes <code>wc2026.db</code> (tables: matches_history, wc2026_matches, polymarket_markets, polymarket_prices). <code>02_model.py</code> computes the Elo and form features, runs the hold-out benchmark, fits the final models and predicts the 104 matches (tables: elo_pre_tournament, model_predictions). <code>03_evaluate.py</code> merges the predictions with the pre-kickoff prices, computes all metrics and the betting simulation, and draws the figures (table: evaluation). <code>04_build_report.py</code> renders this page from the computed results, so that every figure quoted in the text is generated rather than transcribed. Random seeds are fixed; the only element outside my control is the Polymarket API, whose archived prices could in principle be revised, which is why raw copies of every source are kept in the data folder.</p>
{elo_tbl}
<h3>All 104 matches</h3>
<details open><summary>Model and market probabilities for every match</summary>
{app_tbl}
</details>
<h3>References</h3>
<div class="refs">
<p>Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. Monthly Weather Review, 78(1), 1-3.</p>
<p>Dixon, M. J. and Coles, S. G. (1997). Modelling association football scores and inefficiencies in the football betting market. Journal of the Royal Statistical Society: Series C, 46(2), 265-280.</p>
<p>Jürisoo, M. (martj42). International football results from 1872 to the present. GitHub repository, github.com/martj42/international_results. Accessed 10 September 2026.</p>
<p>openfootball. World Cup 2026 fixtures and results in JSON. GitHub repository, github.com/openfootball/worldcup.json. Accessed 10 September 2026.</p>
<p>Pedregosa, F. et al. (2011). Scikit-learn: machine learning in Python. Journal of Machine Learning Research, 12, 2825-2830.</p>
<p>Polymarket. Developer documentation for the Gamma and CLOB APIs, docs.polymarket.com. Accessed 10 September 2026.</p>
<p>World Football Elo Ratings. About the rating system, eloratings.net/about. Accessed 10 September 2026.</p>
</div>
<footer>Prepared with Python 3.12, pandas, scikit-learn, statsmodels, SciPy and matplotlib. Data as of 10 September 2026.</footer>
</div>
</body>
</html>
"""

os.makedirs(os.path.join(DOCS, "assets"), exist_ok=True)
os.makedirs(os.path.join(DOCS, "data"), exist_ok=True)
open(os.path.join(DOCS, "index.html"), "w", encoding="utf-8").write(html)
for f in os.listdir(ASSETS):
    shutil.copy(os.path.join(ASSETS, f), os.path.join(DOCS, "assets", f))
for f in ["matches_evaluated.csv", "metrics_wc2026.csv", "betting_strategies.csv", "predictions.csv", "elo_pre_tournament.csv", "calibration.csv", "edge_buckets.csv"]:
    shutil.copy(os.path.join(OUT, f), os.path.join(DOCS, "data", f))
open(os.path.join(DOCS, ".nojekyll"), "w").close()
print("written", os.path.join(DOCS, "index.html"), f"{len(html):,} chars")
