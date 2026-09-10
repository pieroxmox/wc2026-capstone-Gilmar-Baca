"""
03_evaluate.py
Merges model predictions with Polymarket pre-match prices and evaluates both.

1. Pre-match price = last recorded price before the kickoff timestamp (12-hour resolution).
2. Market probabilities = prices normalised to sum to one (removes the overround).
3. Accuracy, log loss and multi-class Brier for each model and for the market, overall and by stage,
   with bootstrap confidence intervals for the model-minus-market Brier difference.
4. Calibration table for model and market.
5. Betting simulation: flat 150 USD stake, filled at the pre-match price, no fees or slippage.
6. Figures for the report (matplotlib, PNG, 150 dpi).

Run:  python 03_evaluate.py
Output: results/*.csv, results/summary.json, assets/fig*.png
"""

import json
import os
import sqlite3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DB_PATH = "wc2026.db"
OUT, ASSETS = "results", "assets"
CLASSES = ["H", "D", "A"]
STAKE = 150.0
EDGE = 0.05
SEED = 42
MODELS = {"hgb": "Gradient boosting", "pois": "Poisson goals", "ens": "Ensemble", "mkt": "Polymarket"}

# palette
C_MODEL, C_MARKET, C_GREY, C_ACCENT, C_GREEN, C_RED = "#1F4E79", "#C0504D", "#8C8C8C", "#E2A03F", "#3C8D5A", "#B03A2E"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 150, "savefig.bbox": "tight"})


def onehot(y):
    return np.array([[1.0 if c == t else 0.0 for c in CLASSES] for t in y])


def brier(y, P):
    return float(np.mean(np.sum((P - onehot(y)) ** 2, axis=1)))


def brier_rows(y, P):
    return np.sum((P - onehot(y)) ** 2, axis=1)


def logloss(y, P):
    P = np.clip(P, 1e-12, 1)
    return float(-np.mean(np.log(np.sum(P * onehot(y), axis=1))))


def accuracy(y, P):
    return float(np.mean(np.array(CLASSES)[P.argmax(1)] == np.asarray(y)))


def probs(df, m):
    return df[[f"{m}_H", f"{m}_D", f"{m}_A"]].values


def bootstrap_ci(x, n=10000, seed=SEED):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    means = np.array([rng.choice(x, len(x), replace=True).mean() for _ in range(n)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def savefig(name):
    plt.savefig(os.path.join(ASSETS, name))
    plt.close()


# ----------------------------------------------------------------------
# 1. Market prices
# ----------------------------------------------------------------------
def market_frame(con):
    mk = pd.read_sql("SELECT * FROM polymarket_markets", con)
    px = pd.read_sql("SELECT * FROM polymarket_prices", con)
    # epoch seconds, independent of the datetime resolution pandas picks
    mk["kick_ts"] = ((pd.to_datetime(mk.kickoff_utc, utc=True) - pd.Timestamp(0, tz="UTC")).dt.total_seconds()).astype("int64")
    j = px.merge(mk[["wc_match_id", "market_type", "kick_ts"]], on=["wc_match_id", "market_type"])
    pre = j[j.ts < j.kick_ts].sort_values("ts")
    close = pre.groupby(["wc_match_id", "market_type"]).agg(price_close=("price", "last"), ts_close=("ts", "last"),
                                                            price_open=("price", "first"), n_points=("price", "size")).reset_index()
    close = close.merge(mk[["wc_match_id", "market_type", "kick_ts", "volume_usd", "resolved_yes"]], on=["wc_match_id", "market_type"])
    close["hours_before_kickoff"] = (close.kick_ts - close.ts_close) / 3600
    wide = close.pivot(index="wc_match_id", columns="market_type", values="price_close").rename(
        columns={"home": "px_H", "draw": "px_D", "away": "px_A"})
    wide["overround"] = wide[["px_H", "px_D", "px_A"]].sum(1)
    for c in CLASSES:
        wide[f"mkt_{c}"] = wide[f"px_{c}"] / wide["overround"]
    wide["mkt_pick"] = wide[["mkt_H", "mkt_D", "mkt_A"]].values.argmax(1)
    wide["mkt_pick"] = wide["mkt_pick"].map(dict(enumerate(CLASSES)))
    wide["hours_before_kickoff"] = close.groupby("wc_match_id").hours_before_kickoff.mean()
    wide["volume_usd"] = close.groupby("wc_match_id").volume_usd.sum()
    wide["n_price_points"] = close.groupby("wc_match_id").n_points.sum()
    wide["kick_ts"] = close.groupby("wc_match_id").kick_ts.first()
    return wide.reset_index(), close


# ----------------------------------------------------------------------
# 2. Betting simulation
# ----------------------------------------------------------------------
def simulate(df, pick_col, filter_mask=None, label=""):
    d = df.copy()
    if filter_mask is not None:
        d = d[filter_mask].copy()
    d["bet"] = d[pick_col]
    d["price"] = [r[f"px_{r['bet']}"] for _, r in d.iterrows()]
    d["won"] = (d["bet"] == d["result_90"]).astype(int)
    d["profit"] = np.where(d.won == 1, STAKE * (1 / d.price - 1), -STAKE)
    d["cum_profit"] = d.profit.cumsum()
    staked = STAKE * len(d)
    lo, hi = bootstrap_ci(d.profit.values) if len(d) > 1 else (np.nan, np.nan)
    peak = d.cum_profit.cummax()
    summary = {"strategy": label, "bets": int(len(d)), "wins": int(d.won.sum()),
               "hit_rate": float(d.won.mean()) if len(d) else np.nan,
               "avg_price": float(d.price.mean()) if len(d) else np.nan,
               "staked_usd": staked, "profit_usd": float(d.profit.sum()),
               "roi": float(d.profit.sum() / staked) if staked else np.nan,
               "profit_ci95_low": lo * len(d), "profit_ci95_high": hi * len(d),
               "max_drawdown_usd": float((d.cum_profit - peak).min()) if len(d) else 0.0}
    return summary, d[["wc_match_id", "date", "stage", "home_team", "away_team", "bet", "price", "result_90", "won", "profit", "cum_profit"]]


def main():
    os.makedirs(OUT, exist_ok=True); os.makedirs(ASSETS, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    pred = pd.read_sql("SELECT * FROM model_predictions ORDER BY date, wc_match_id", con)
    hist = pd.read_sql("SELECT * FROM matches_history", con)
    elo = pd.read_sql("SELECT * FROM elo_pre_tournament", con)
    mk, close = market_frame(con)
    df = pred.merge(mk, on="wc_match_id").sort_values(["date", "wc_match_id"]).reset_index(drop=True)
    df["match_no"] = np.arange(1, len(df) + 1)
    for c in CLASSES:
        df[f"edge_{c}"] = df[f"ens_{c}"] - df[f"mkt_{c}"]
    df["edge_pick"] = [r[f"edge_{r['ens_pick']}"] for _, r in df.iterrows()]
    df["max_edge_outcome"] = df[["edge_H", "edge_D", "edge_A"]].values.argmax(1)
    df["max_edge_outcome"] = df["max_edge_outcome"].map(dict(enumerate(CLASSES)))
    df["max_edge"] = df[["edge_H", "edge_D", "edge_A"]].max(1)
    df.to_csv(os.path.join(OUT, "matches_evaluated.csv"), index=False)
    df.to_sql("evaluation", con, if_exists="replace", index=False)
    summary = {}

    # --- data description ------------------------------------------------------------
    summary["data"] = {
        "history_rows": int(len(hist)), "history_first": str(hist.date.min()), "history_last": str(hist.date.max()),
        "history_pre_tournament": int((hist.date < "2026-06-11").sum()),
        "training_2000_plus": int(((hist.date >= "2000-01-01") & (hist.date < "2026-06-11")).sum()),
        "wc_matches": int(len(df)), "group_matches": int((df.stage == "Group").sum()), "knockout_matches": int((df.stage == "Knockout").sum()),
        "wc_result_shares": df.result_90.value_counts(normalize=True).round(3).to_dict(),
        "polymarket_markets": int(len(close)), "price_points": int(close.n_points.sum()),
        "hours_before_kickoff_median": float(df.hours_before_kickoff.median()),
        "hours_before_kickoff_p90": float(df.hours_before_kickoff.quantile(0.9)),
        "hours_before_kickoff_max": float(df.hours_before_kickoff.max()),
        "overround_mean": float(df.overround.mean()), "overround_min": float(df.overround.min()), "overround_max": float(df.overround.max()),
        "volume_total_usd": float(df.volume_usd.sum()), "volume_median_per_match_usd": float(df.volume_usd.median()),
        "goals_per_match_wc": float((df.ft_home + df.ft_away).mean()),
        "goals_per_match_hist_2000": float((hist[(hist.date >= "2000-01-01") & (hist.date < "2026-06-11")].eval("home_score+away_score")).mean()),
    }

    # --- accuracy / log loss / brier -------------------------------------------------
    rows = []
    for m, name in MODELS.items():
        P = probs(df, m)
        row = {"model": name, "accuracy": accuracy(df.result_90, P), "log_loss": logloss(df.result_90, P), "brier": brier(df.result_90, P)}
        for stage in ["Group", "Knockout"]:
            s = df.stage == stage
            row[f"brier_{stage.lower()}"] = brier(df.result_90[s], P[s.values])
            row[f"accuracy_{stage.lower()}"] = accuracy(df.result_90[s], P[s.values])
        rows.append(row)
    base = np.tile([1 / 3] * 3, (len(df), 1))
    rows.append({"model": "Uniform (1/3 each)", "accuracy": accuracy(df.result_90, base), "log_loss": logloss(df.result_90, base), "brier": brier(df.result_90, base),
                 "brier_group": brier(df.result_90[df.stage == "Group"], base[(df.stage == "Group").values]),
                 "brier_knockout": brier(df.result_90[df.stage == "Knockout"], base[(df.stage == "Knockout").values]),
                 "accuracy_group": np.nan, "accuracy_knockout": np.nan})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(os.path.join(OUT, "metrics_wc2026.csv"), index=False)
    print(metrics.round(4).to_string(index=False))
    # paired difference model - market (positive = market better)
    diffs = {}
    for m in ["hgb", "pois", "ens"]:
        d = brier_rows(df.result_90, probs(df, m)) - brier_rows(df.result_90, probs(df, "mkt"))
        lo, hi = bootstrap_ci(d)
        diffs[m] = {"mean_diff": float(d.mean()), "ci95": [lo, hi], "share_matches_model_better": float((d < 0).mean())}
    summary["brier_diff_vs_market"] = diffs
    summary["metrics"] = metrics.round(4).to_dict(orient="records")
    # agreement between model and market picks
    summary["pick_agreement"] = {
        "ens_vs_market": float((df.ens_pick == df.mkt_pick).mean()),
        "ens_pick_shares": df.ens_pick.value_counts(normalize=True).round(3).to_dict(),
        "mkt_pick_shares": df.mkt_pick.value_counts(normalize=True).round(3).to_dict(),
        "draws_predicted_by_ens": int((df.ens_pick == "D").sum()), "draws_predicted_by_market": int((df.mkt_pick == "D").sum()),
        "favourite_hit_rate_market": float((df.mkt_pick == df.result_90).mean()),
    }

    # --- calibration -----------------------------------------------------------------
    long = []
    for c in CLASSES:
        long.append(pd.DataFrame({"p_model": df[f"ens_{c}"], "p_market": df[f"mkt_{c}"], "hit": (df.result_90 == c).astype(int)}))
    long = pd.concat(long, ignore_index=True)
    bins = [0, 0.15, 0.25, 0.35, 0.45, 0.6, 1.0]
    cal = []
    for src in ["model", "market"]:
        b = pd.cut(long[f"p_{src}"], bins, include_lowest=True)
        g = long.groupby(b, observed=True).agg(n=("hit", "size"), mean_p=(f"p_{src}", "mean"), observed=("hit", "mean")).reset_index()
        g["source"] = src; g["bin"] = g.iloc[:, 0].astype(str)
        cal.append(g[["source", "bin", "n", "mean_p", "observed"]])
    cal = pd.concat(cal, ignore_index=True)
    cal.to_csv(os.path.join(OUT, "calibration.csv"), index=False)

    # --- betting simulation ----------------------------------------------------------
    strategies, ledgers = [], {}
    for m in ["ens", "hgb", "pois"]:
        s, led = simulate(df, f"{m}_pick", None, f"A. {MODELS[m]} pick, every match")
        strategies.append(s); ledgers[f"A_{m}"] = led
    s, led = simulate(df, "ens_pick", df.edge_pick > EDGE, f"B. Ensemble pick when edge > {EDGE:.0%}")
    strategies.append(s); ledgers["B"] = led
    s, led = simulate(df, "max_edge_outcome", df.max_edge > EDGE, f"C. Largest-edge outcome when edge > {EDGE:.0%}")
    strategies.append(s); ledgers["C"] = led
    s, led = simulate(df, "mkt_pick", None, "D. Market favourite, every match (control)")
    strategies.append(s); ledgers["D"] = led
    strat = pd.DataFrame(strategies)
    strat.to_csv(os.path.join(OUT, "betting_strategies.csv"), index=False)
    for k, led in ledgers.items():
        led.to_csv(os.path.join(OUT, f"ledger_{k}.csv"), index=False)
    print(strat.round(3).to_string(index=False))
    summary["strategies"] = strat.round(4).to_dict(orient="records")
    # profit by stage for the headline strategy
    ledA = ledgers["A_ens"]
    summary["strategy_A_ens_by_stage"] = ledA.groupby("stage").agg(bets=("won", "size"), wins=("won", "sum"), profit=("profit", "sum")).round(2).to_dict(orient="index")
    # edge buckets for strategy A
    ledA = ledA.merge(df[["wc_match_id", "edge_pick"]], on="wc_match_id")
    eb = pd.cut(ledA.edge_pick, [-1, -0.05, 0, 0.05, 0.10, 1], labels=["< -5 pts", "-5 to 0", "0 to 5", "5 to 10", "> 10 pts"])
    edge_tbl = ledA.groupby(eb, observed=True).agg(bets=("won", "size"), hit_rate=("won", "mean"), profit=("profit", "sum")).reset_index()
    edge_tbl["roi"] = edge_tbl.profit / (STAKE * edge_tbl.bets)
    edge_tbl.rename(columns={"edge_pick": "edge_bucket"}, inplace=True)
    edge_tbl.to_csv(os.path.join(OUT, "edge_buckets.csv"), index=False)
    summary["edge_buckets"] = edge_tbl.round(4).astype({"edge_bucket": str}).to_dict(orient="records")
    # biggest surprises: lowest market price for the actual result
    df["price_of_result"] = [r[f"px_{r['result_90']}"] for _, r in df.iterrows()]
    surprises = df.nsmallest(8, "price_of_result")[["wc_match_id", "date", "home_team", "away_team", "ft_home", "ft_away", "result_90", "price_of_result", "ens_H", "ens_D", "ens_A"]]
    surprises.to_csv(os.path.join(OUT, "surprises.csv"), index=False)
    summary["surprises"] = surprises.round(3).to_dict(orient="records")
    summary["elo_top"] = elo[elo.in_wc2026 == 1].head(16).round(0).to_dict(orient="records")

    # ==================================================================================
    # Figures
    # ==================================================================================
    h = hist[(hist.date >= "2000-01-01") & (hist.date < "2026-06-11")].copy()
    h["result"] = np.where(h.home_score > h.away_score, "H", np.where(h.home_score < h.away_score, "A", "D"))
    h["ctx"] = np.where(h.tournament == "FIFA World Cup", "World Cup finals",
                        np.where(h.tournament.str.contains("qualification"), "Qualifiers",
                                 np.where(h.tournament == "Friendly", "Friendlies", "Other tournaments")))
    h["ctx"] = np.where((h.ctx != "World Cup finals") & (~h.neutral.astype(bool)), h.ctx + " (home team)", h.ctx)
    # Fig 1: outcome shares by context
    tab = pd.crosstab(h.ctx, h.result, normalize="index")[CLASSES]
    tab.loc["World Cup 2026 (this study)"] = df.result_90.value_counts(normalize=True).reindex(CLASSES).values
    fig, ax = plt.subplots(figsize=(8, 4))
    left = np.zeros(len(tab))
    for c, col, lab in zip(CLASSES, [C_MODEL, C_GREY, C_MARKET], ["Home / first-named team wins", "Draw", "Away / second-named team wins"]):
        ax.barh(tab.index, tab[c], left=left, color=col, label=lab)
        for i, (v, l) in enumerate(zip(tab[c], left)):
            ax.text(l + v / 2, i, f"{v:.0%}", ha="center", va="center", color="white", fontsize=8)
        left += tab[c].values
    ax.set_xlim(0, 1); ax.set_xlabel("Share of matches"); ax.set_ylabel("")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False, fontsize=8)
    ax.grid(False)
    savefig("fig1_outcome_shares.png")

    # Fig 2: pre-tournament Elo
    top = elo[elo.in_wc2026 == 1].head(16).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top.team, top.elo, color=C_MODEL)
    for i, v in enumerate(top.elo):
        ax.text(v + 5, i, f"{v:.0f}", va="center", fontsize=8)
    ax.set_xlim(top.elo.min() - 80, top.elo.max() + 60)
    ax.set_xlabel("Elo rating on 10 June 2026 (computed from results since 1872)"); ax.set_ylabel("")
    savefig("fig2_elo_pre_tournament.png")

    # Fig 3: price paths for the final
    px = pd.read_sql("SELECT * FROM polymarket_prices", con)
    ex = df[df.wc_match_id == 1].iloc[0]
    p = px[px.wc_match_id == 1].copy(); p["t"] = pd.to_datetime(p.ts, unit="s", utc=True)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for mt, col, lab in [("home", C_MODEL, f"{ex.home_team} win"), ("draw", C_GREY, "Draw"), ("away", C_MARKET, f"{ex.away_team} win")]:
        q = p[p.market_type == mt].sort_values("t")
        ax.plot(q.t, q.price, marker="o", ms=3, color=col, label=lab)
    ax.axvline(pd.to_datetime(ex.kick_ts, unit="s", utc=True), color="black", ls="--", lw=0.8)
    ax.text(pd.to_datetime(ex.kick_ts, unit="s", utc=True), 0.68, " kickoff", fontsize=8, va="top")
    ax.set_ylabel("Price of the Yes share (USD)"); ax.set_xlabel("Date (UTC), 12-hour samples"); ax.set_ylim(0, 0.7)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.autofmt_xdate()
    savefig("fig3_price_path_opening_match.png")

    # Fig 4: model vs market scatter
    fig, ax = plt.subplots(figsize=(6, 6))
    for hit, col, lab in [(0, C_GREY, "Outcome did not happen"), (1, C_GREEN, "Outcome happened")]:
        q = long[long.hit == hit]
        ax.scatter(q.p_market, q.p_model, s=18, color=col, alpha=0.75, label=lab, edgecolor="none")
    ax.plot([0, 0.9], [0, 0.9], color="black", lw=0.8, ls="--")
    ax.set_xlabel("Polymarket pre-match probability (normalised)"); ax.set_ylabel("Ensemble model probability")
    ax.set_xlim(0, 0.9); ax.set_ylim(0, 0.9); ax.set_aspect("equal")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    savefig("fig4_model_vs_market.png")

    # Fig 5: calibration
    fig, ax = plt.subplots(figsize=(6, 5))
    for src, col, lab in [("model", C_MODEL, "Ensemble model"), ("market", C_MARKET, "Polymarket")]:
        g = cal[cal.source == src]
        ax.plot(g.mean_p, g.observed, marker="o", color=col, label=lab)
        for _, r in g.iterrows():
            ax.annotate(f"n={r.n}", (r.mean_p, r.observed), textcoords="offset points", xytext=(4, -10 if src == "model" else 6), fontsize=7, color=col)
    ax.plot([0, 0.8], [0, 0.8], color="black", lw=0.8, ls="--", label="Perfect calibration")
    ax.set_xlabel("Average predicted probability in bin"); ax.set_ylabel("Observed frequency of the outcome")
    ax.set_xlim(0, 0.8); ax.set_ylim(0, 0.8)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    savefig("fig5_calibration.png")

    # Fig 6: Brier by stage
    mm = metrics[metrics.model.isin(["Gradient boosting", "Poisson goals", "Ensemble", "Polymarket", "Uniform (1/3 each)"])]
    x = np.arange(len(mm)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w / 2, mm.brier_group, w, color=C_MODEL, label=f"Group stage (n={int((df.stage == 'Group').sum())})")
    ax.bar(x + w / 2, mm.brier_knockout, w, color=C_ACCENT, label=f"Knockout stage (n={int((df.stage == 'Knockout').sum())})")
    for xi, (a, b) in enumerate(zip(mm.brier_group, mm.brier_knockout)):
        ax.text(xi - w / 2, a + 0.005, f"{a:.3f}", ha="center", fontsize=7); ax.text(xi + w / 2, b + 0.005, f"{b:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(mm.model, fontsize=8)
    ax.set_ylabel("Multi-class Brier score (lower is better)"); ax.set_ylim(0.4, 0.75)
    ax.legend(frameon=False, fontsize=8)
    savefig("fig6_brier_by_stage.png")

    # Fig 7: cumulative P&L
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for k, col, lab in [("A_ens", C_MODEL, strat.strategy[0]), ("B", C_ACCENT, strat.strategy[3]), ("C", C_GREEN, strat.strategy[4]), ("D", C_MARKET, strat.strategy[5])]:
        led = ledgers[k]
        full = df[["match_no", "wc_match_id"]].merge(led[["wc_match_id", "profit"]], on="wc_match_id", how="left").fillna({"profit": 0})
        ax.plot(full.match_no, full.profit.cumsum(), color=col, label=lab, lw=1.6)
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(72.5, color="black", lw=0.8, ls=":"); ax.text(73, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 50, "knockouts", fontsize=8)
    ax.set_xlabel("Match number in tournament order (1 to 104)"); ax.set_ylabel(f"Cumulative profit (USD, {STAKE:.0f} USD flat stake)")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    savefig("fig7_cumulative_pnl.png")

    # Fig 8: hit rate and ROI by edge bucket (strategy A)
    fig, ax1 = plt.subplots(figsize=(8, 4))
    x = np.arange(len(edge_tbl))
    ax1.bar(x, edge_tbl.roi, color=[C_GREEN if v >= 0 else C_RED for v in edge_tbl.roi])
    for xi, (v, n) in enumerate(zip(edge_tbl.roi, edge_tbl.bets)):
        ax1.text(xi, v + (0.02 if v >= 0 else -0.06), f"{v:+.0%}\n({n} bets)", ha="center", fontsize=8)
    ax1.axhline(0, color="black", lw=0.8)
    ax1.set_xticks(x); ax1.set_xticklabels(edge_tbl.edge_bucket.astype(str))
    ax1.set_xlabel("Edge of the model's pick over the market (model probability minus market probability)")
    ax1.set_ylabel("Return on stake"); ax1.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax1.set_ylim(min(edge_tbl.roi.min() - 0.15, -0.3), max(edge_tbl.roi.max() + 0.2, 0.3))
    savefig("fig8_roi_by_edge.png")

    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=float)
    con.close()
    print("\nsummary:", json.dumps({k: summary[k] for k in ["data", "brier_diff_vs_market", "pick_agreement"]}, indent=1, default=float))


if __name__ == "__main__":
    main()
