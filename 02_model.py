"""
02_model.py
Feature engineering and modelling.

1. Elo ratings computed chronologically over the full martj42 history (1872 to 10 June 2026).
2. Rolling form features from each team's previous five matches.
3. Result classifier: HistGradientBoostingClassifier (H / D / A), benchmarked against
   multinomial logistic regression and naive baselines on a temporal hold-out (2022 to June 2026).
4. Goals model: two Poisson GLMs (home goals, away goals) giving expected goals and a
   score-matrix view of the same three outcomes.
5. Sequential prediction of the 104 World Cup 2026 matches. The classifier is trained only on
   matches before 11 June 2026; Elo and form update match by match with information available
   at each kickoff, but the classifier is never refitted during the tournament.

Run:  python 02_model.py
Output: results/predictions.csv, results/holdout_metrics.json, results/elo_pre_tournament.csv
        and the same tables written to wc2026.db
"""

import json
import os
import sqlite3

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import poisson
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

DB_PATH = "wc2026.db"
OUT = "results"
TOURNAMENT_START = "2026-06-11"
TRAIN_FROM = "2000-01-01"
HOLDOUT_FROM = "2022-01-01"
HOSTS = {"United States", "Mexico", "Canada"}
CLASSES = ["H", "D", "A"]
SEED = 42

CONTINENTAL = ("UEFA Euro", "Copa América", "African Cup of Nations", "AFC Asian Cup",
               "Gold Cup", "Confederations Cup", "UEFA Nations League")


# ----------------------------------------------------------------------
# Elo
# ----------------------------------------------------------------------
def k_class(tournament):
    t = str(tournament)
    if t == "FIFA World Cup":
        return 4
    if "qualification" in t:
        return 2
    if t == "Friendly":
        return 0
    if any(c in t for c in CONTINENTAL):
        return 3
    return 1


K_BY_CLASS = {0: 20, 1: 30, 2: 40, 3: 50, 4: 60}
HOME_ADV_ELO = 100


def gd_multiplier(gd):
    gd = abs(gd)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11 + gd) / 8


class RatingState:
    """Elo ratings plus a short result history per team, updated chronologically."""

    def __init__(self):
        self.elo = {}
        self.recent = {}  # team -> list of (points, gf, ga), most recent last

    def get(self, team):
        return self.elo.get(team, 1500.0)

    def form(self, team, n=5):
        h = self.recent.get(team, [])[-n:]
        if not h:
            return 0.5, 1.3, 1.3, 0
        pts = np.mean([x[0] for x in h]); gf = np.mean([x[1] for x in h]); ga = np.mean([x[2] for x in h])
        return pts, gf, ga, len(h)

    def features(self, home, away, neutral, kc):
        eh, ea = self.get(home), self.get(away)
        adv = 0 if neutral else HOME_ADV_ELO
        diff = eh + adv - ea
        fh, fa = self.form(home), self.form(away)
        return {
            "elo_home": eh, "elo_away": ea, "elo_diff": diff, "elo_diff_raw": eh - ea,
            "elo_expect": 1 / (1 + 10 ** (-diff / 400)),
            "home_adv": 0 if neutral else 1, "k_class": kc,
            "form_pts_diff": fh[0] - fa[0], "form_gf_diff": fh[1] - fa[1], "form_ga_diff": fh[2] - fa[2],
            "n_recent_home": fh[3], "n_recent_away": fa[3],
        }

    def update(self, home, away, hs, as_, neutral, kc):
        eh, ea = self.get(home), self.get(away)
        adv = 0 if neutral else HOME_ADV_ELO
        exp_h = 1 / (1 + 10 ** ((ea - eh - adv) / 400))
        s_h = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
        delta = K_BY_CLASS[kc] * gd_multiplier(hs - as_) * (s_h - exp_h)
        self.elo[home] = eh + delta
        self.elo[away] = ea - delta
        self.recent.setdefault(home, []).append((s_h, hs, as_))
        self.recent.setdefault(away, []).append((1 - s_h, as_, hs))
        for t in (home, away):
            self.recent[t] = self.recent[t][-10:]


FEATURES = ["elo_home", "elo_away", "elo_diff", "elo_diff_raw", "elo_expect", "home_adv", "k_class",
            "form_pts_diff", "form_gf_diff", "form_ga_diff"]


def build_training_frame(hist):
    """Walk through history once, emitting pre-match features and updating ratings after each match."""
    state = RatingState()
    rows = []
    for r in hist.itertuples(index=False):
        kc = k_class(r.tournament)
        f = state.features(r.home_team, r.away_team, r.neutral, kc)
        f.update({"match_id": r.match_id, "date": r.date, "home_team": r.home_team, "away_team": r.away_team,
                  "home_score": r.home_score, "away_score": r.away_score, "tournament": r.tournament,
                  "result": "H" if r.home_score > r.away_score else ("A" if r.home_score < r.away_score else "D")})
        rows.append(f)
        state.update(r.home_team, r.away_team, r.home_score, r.away_score, r.neutral, kc)
    return pd.DataFrame(rows), state


# ----------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------
def onehot(y_true, classes=CLASSES):
    return np.array([[1.0 if c == t else 0.0 for c in classes] for t in y_true])


def brier_multi(y_true, proba, classes=CLASSES):
    """Multi-class Brier score: mean over matches of the squared error summed over the three outcomes."""
    return float(np.mean(np.sum((proba - onehot(y_true, classes)) ** 2, axis=1)))


def log_loss(y_true, proba, labels=CLASSES):
    """Mean negative log probability assigned to the observed outcome (columns follow `labels`)."""
    p = np.clip(np.asarray(proba, dtype=float), 1e-12, 1)
    return float(-np.mean(np.log(np.sum(p * onehot(y_true, labels), axis=1))))


def make_hgb():
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.04, max_depth=4, min_samples_leaf=60,
                                          l2_regularization=1.0, random_state=SEED)


def evaluate(name, model, Xtr, ytr, Xte, yte):
    model.fit(Xtr, ytr)
    p = model.predict_proba(Xte)
    order = [list(model.classes_).index(c) for c in CLASSES]
    p = p[:, order]
    pred = np.array(CLASSES)[p.argmax(axis=1)]
    return {"model": name, "accuracy": float(accuracy_score(yte, pred)),
            "log_loss": float(log_loss(yte, p, labels=CLASSES)), "brier": brier_multi(yte, p)}, model


POISSON_X = ["elo_diff_raw", "home_adv", "k_class"]


def fit_poisson(train):
    X = sm.add_constant(train[POISSON_X].astype(float))
    m_home = sm.GLM(train["home_score"], X, family=sm.families.Poisson()).fit()
    m_away = sm.GLM(train["away_score"], X, family=sm.families.Poisson()).fit()
    return m_home, m_away


def poisson_probs(m_home, m_away, feats, max_goals=10):
    X = sm.add_constant(feats[POISSON_X].astype(float), has_constant="add")
    lam_h = np.asarray(m_home.predict(X)); lam_a = np.asarray(m_away.predict(X))
    out = []
    g = np.arange(max_goals + 1)
    for lh, la in zip(lam_h, lam_a):
        ph = poisson.pmf(g, lh); pa = poisson.pmf(g, la)
        M = np.outer(ph, pa)
        M = M / M.sum()  # renormalise after truncating at max_goals
        out.append([np.tril(M, -1).sum(), np.trace(M), np.triu(M, 1).sum()])
    return np.array(out), lam_h, lam_a


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    os.makedirs(OUT, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    hist = pd.read_sql("SELECT * FROM matches_history ORDER BY date, match_id", con)
    hist["neutral"] = hist["neutral"].astype(bool)
    wc = pd.read_sql("SELECT * FROM wc2026_matches ORDER BY date, wc_match_id", con)

    pre = hist[hist.date < TOURNAMENT_START].copy()
    print(f"history before tournament: {len(pre):,} matches")
    frame, state = build_training_frame(pre)

    train_all = frame[frame.date >= TRAIN_FROM].copy()
    tr = train_all[train_all.date < HOLDOUT_FROM]
    te = train_all[train_all.date >= HOLDOUT_FROM]
    print(f"train {len(tr):,} (2000-2021)  holdout {len(te):,} (2022 to June 2026)")

    # --- temporal hold-out benchmark -------------------------------------------------
    results = []
    base = te.result.value_counts(normalize=True).reindex(CLASSES).values
    p_base = np.tile(base, (len(te), 1))
    results.append({"model": "Base rates (train frequencies)", "accuracy": float((te.result == CLASSES[base.argmax()]).mean()),
                    "log_loss": float(log_loss(te.result, p_base, labels=CLASSES)), "brier": brier_multi(te.result, p_base)})
    # Elo-only: map Elo expectancy to three outcomes with a fitted logistic model on elo_expect alone
    m, _ = evaluate("Elo only (logistic on Elo expectancy)", make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)),
                    tr[["elo_expect"]], tr.result, te[["elo_expect"]], te.result)
    results.append(m)
    m, _ = evaluate("Multinomial logistic regression", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
                    tr[FEATURES], tr.result, te[FEATURES], te.result)
    results.append(m)
    m, hgb_holdout = evaluate("HistGradientBoosting", make_hgb(), tr[FEATURES], tr.result, te[FEATURES], te.result)
    results.append(m)
    mh, ma = fit_poisson(tr)
    pp, _, _ = poisson_probs(mh, ma, te)
    results.append({"model": "Poisson goals model", "accuracy": float((np.array(CLASSES)[pp.argmax(1)] == te.result).mean()),
                    "log_loss": float(log_loss(te.result, pp, labels=CLASSES)), "brier": brier_multi(te.result, pp)})
    holdout = pd.DataFrame(results)
    print(holdout.round(4).to_string(index=False))
    # per-class breakdown of the hold-out for the report
    p_hgb = hgb_holdout.predict_proba(te[FEATURES])[:, [list(hgb_holdout.classes_).index(c) for c in CLASSES]]
    holdout_detail = {
        "n_train": int(len(tr)), "n_holdout": int(len(te)),
        "holdout_result_shares": te.result.value_counts(normalize=True).round(4).to_dict(),
        "hgb_pred_shares": pd.Series(np.array(CLASSES)[p_hgb.argmax(1)]).value_counts(normalize=True).round(4).to_dict(),
        "metrics": holdout.round(4).to_dict(orient="records"),
    }
    json.dump(holdout_detail, open(os.path.join(OUT, "holdout_metrics.json"), "w"), indent=1)

    # --- refit on everything before the tournament -----------------------------------
    hgb = make_hgb().fit(train_all[FEATURES], train_all.result)
    mh, ma = fit_poisson(train_all)
    print("Poisson home-goals coefficients:", mh.params.round(4).to_dict())

    # pre-tournament Elo table
    elo = pd.DataFrame({"team": list(state.elo), "elo": list(state.elo.values())}).sort_values("elo", ascending=False)
    wc_teams = set(wc.home_team) | set(wc.away_team)
    elo["in_wc2026"] = elo.team.isin(wc_teams)
    elo["rank_overall"] = np.arange(1, len(elo) + 1)
    elo.to_csv(os.path.join(OUT, "elo_pre_tournament.csv"), index=False)
    elo.to_sql("elo_pre_tournament", con, if_exists="replace", index=False)

    # --- sequential World Cup prediction ---------------------------------------------
    # host-nation matches: martj42 flags them non-neutral with the host as home_team. openfootball keeps the
    # draw order, so in three group games the host is listed second; those are predicted with the teams
    # swapped (host as home) and the probabilities flipped back to the openfootball order.
    non_neutral = set(hist[(hist.date >= TOURNAMENT_START) & (~hist.neutral)][["home_team", "away_team"]].itertuples(index=False, name=None))
    rows = []
    for r in wc.itertuples(index=False):
        if (r.home_team, r.away_team) in non_neutral:
            host, neutral, swap = "home", False, False
        elif (r.away_team, r.home_team) in non_neutral:
            host, neutral, swap = "away", False, True
        else:
            host, neutral, swap = None, True, False
        h_t, a_t = (r.away_team, r.home_team) if swap else (r.home_team, r.away_team)
        f = state.features(h_t, a_t, neutral, 4)
        X = pd.DataFrame([f])[FEATURES]
        p = hgb.predict_proba(X)[0][[list(hgb.classes_).index(c) for c in CLASSES]]
        pp, lh, la = poisson_probs(mh, ma, X)
        pp = pp[0]
        if swap:  # back to openfootball order: H <-> A, expected goals swapped
            p = p[[2, 1, 0]]; pp = pp[[2, 1, 0]]; lh, la = la, lh
        ens = (p + pp) / 2
        rows.append({
            "wc_match_id": r.wc_match_id, "date": r.date, "stage": r.stage, "round": r.round,
            "home_team": r.home_team, "away_team": r.away_team, "host_side": host or "none",
            "elo_home_pre": round(state.get(r.home_team), 1), "elo_away_pre": round(state.get(r.away_team), 1),
            "hgb_H": p[0], "hgb_D": p[1], "hgb_A": p[2],
            "pois_H": pp[0], "pois_D": pp[1], "pois_A": pp[2],
            "ens_H": ens[0], "ens_D": ens[1], "ens_A": ens[2],
            "xg_home": float(lh[0]), "xg_away": float(la[0]),
            "ft_home": r.ft_home, "ft_away": r.ft_away, "result_90": r.result_90,
        })
        if swap:
            state.update(r.away_team, r.home_team, int(r.ft_away), int(r.ft_home), neutral, 4)
        else:
            state.update(r.home_team, r.away_team, int(r.ft_home), int(r.ft_away), neutral, 4)
    pred = pd.DataFrame(rows)
    for m in ["hgb", "pois", "ens"]:
        pred[f"{m}_pick"] = pred[[f"{m}_H", f"{m}_D", f"{m}_A"]].values.argmax(1)
        pred[f"{m}_pick"] = pred[f"{m}_pick"].map(dict(enumerate(CLASSES)))
    pred.to_csv(os.path.join(OUT, "predictions.csv"), index=False)
    pred.to_sql("model_predictions", con, if_exists="replace", index=False)
    con.close()
    print(f"\nWC2026 predictions: {len(pred)} matches")
    for m in ["hgb", "pois", "ens"]:
        P = pred[[f"{m}_H", f"{m}_D", f"{m}_A"]].values
        print(f"  {m}: accuracy {(pred[f'{m}_pick'] == pred.result_90).mean():.3f}  "
              f"log loss {log_loss(pred.result_90, P, labels=CLASSES):.4f}  brier {brier_multi(pred.result_90, P):.4f}")


if __name__ == "__main__":
    main()
