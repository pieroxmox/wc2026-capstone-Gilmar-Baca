"""
01_build_database.py
Builds wc2026.db from three public sources.

Source 1 (primary)   martj42/international_results  CSV on GitHub
                     ~49.5k international matches since 1872.
Source 2 (primary)   Polymarket Gamma API + CLOB API
                     One event per World Cup 2026 match, three markets each
                     (home win / draw / away win) with daily price history.
Source 3 (auxiliary) openfootball/worldcup.json 2026
                     104 matches with half-time, full-time and extra-time scores.

Run:  python 01_build_database.py
Output: wc2026.db (SQLite) and raw copies of every source in ./data
"""

import io
import json
import os
import sqlite3
import time
import datetime as dt

import pandas as pd
import requests

DB_PATH = "wc2026.db"
DATA_DIR = "data"
H = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124 Safari/537.36"}

HIST_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
WC_JSON_URL = "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2026/worldcup.json"
GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"

# openfootball name -> martj42 name (only where they differ)
NAME_FIX = {"Bosnia & Herzegovina": "Bosnia and Herzegovina", "USA": "United States"}

# martj42 name -> code used in Polymarket event slugs (mostly FIFA codes, a few ISO codes)
CODE = {
    "Algeria": "alg", "Argentina": "arg", "Australia": "aus", "Austria": "aut", "Belgium": "bel",
    "Bosnia and Herzegovina": "bih", "Brazil": "bra", "Canada": "can", "Cape Verde": "cvi",
    "Colombia": "col", "Croatia": "hrv", "Curaçao": "cuw", "Czech Republic": "cze", "DR Congo": "cdr",
    "Ecuador": "ecu", "Egypt": "egy", "England": "eng", "France": "fra", "Germany": "ger", "Ghana": "gha",
    "Haiti": "hai", "Iran": "irn", "Iraq": "irq", "Ivory Coast": "civ", "Japan": "jpn", "Jordan": "jor",
    "Mexico": "mex", "Morocco": "mar", "Netherlands": "ned", "New Zealand": "nzl", "Norway": "nor",
    "Panama": "pan", "Paraguay": "par", "Portugal": "prt", "Qatar": "qat", "Saudi Arabia": "ksa",
    "Scotland": "sco", "Senegal": "sen", "South Africa": "rsa", "South Korea": "kr", "Spain": "esp",
    "Sweden": "swe", "Switzerland": "che", "Tunisia": "tun", "Turkey": "tur", "United States": "usa",
    "Uruguay": "uru", "Uzbekistan": "uzb",
}


def get(url, **kw):
    for attempt in range(4):
        try:
            r = requests.get(url, headers=H, timeout=60, **kw)
            if r.status_code == 429:
                time.sleep(3 * (attempt + 1))
                continue
            r.raise_for_status()
            return r
        except requests.RequestException:
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"failed: {url}")


# ----------------------------------------------------------------------
# Source 1: historical international results
# ----------------------------------------------------------------------
def load_history():
    path = os.path.join(DATA_DIR, "martj42_results.csv")
    if not os.path.exists(path):
        open(path, "w", encoding="utf-8").write(get(HIST_URL).text)
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["neutral"] = df["neutral"].astype(bool)
    df = df.sort_values("date").reset_index(drop=True)
    df.insert(0, "match_id", range(1, len(df) + 1))
    return df


# ----------------------------------------------------------------------
# Source 3: World Cup 2026 official fixtures and scores
# ----------------------------------------------------------------------
def load_wc2026():
    path = os.path.join(DATA_DIR, "openfootball_wc2026.json")
    if not os.path.exists(path):
        json.dump(get(WC_JSON_URL).json(), open(path, "w"))
    j = json.load(open(path))
    rows = []
    for i, m in enumerate(j["matches"], start=1):
        t1 = m["team1"]["name"] if isinstance(m["team1"], dict) else m["team1"]
        t2 = m["team2"]["name"] if isinstance(m["team2"], dict) else m["team2"]
        sc = m.get("score", {})
        ht, ft, et = sc.get("ht"), sc.get("ft"), sc.get("et")
        rows.append({
            "wc_match_id": m.get("num", i),
            "date": m["date"],
            "kickoff_local": m.get("time"),
            "stage": "Group" if m.get("group") else "Knockout",
            "round": m.get("group") or m.get("round"),
            "home_team": NAME_FIX.get(t1, t1),
            "away_team": NAME_FIX.get(t2, t2),
            "ht_home": ht[0] if ht else None, "ht_away": ht[1] if ht else None,
            "ft_home": ft[0] if ft else None, "ft_away": ft[1] if ft else None,
            "et_home": et[0] if et else None, "et_away": et[1] if et else None,
            "venue": m.get("ground"),
            "n_goals_home": len(m.get("goals1", [])),
            "n_goals_away": len(m.get("goals2", [])),
        })
    df = pd.DataFrame(rows)
    df["result_90"] = df.apply(lambda r: "H" if r.ft_home > r.ft_away else ("A" if r.ft_home < r.ft_away else "D"), axis=1)
    return df


# ----------------------------------------------------------------------
# Source 2: Polymarket match markets
# ----------------------------------------------------------------------
# alternative spellings Polymarket uses in market titles
ALT = {"South Korea": "Korea Republic", "Czech Republic": "Czechia", "Turkey": "Türkiye",
       "Cape Verde": "Cabo Verde", "United States": "USA", "Iran": "IR Iran", "Ivory Coast": "Côte d'Ivoire"}


def find_event(home, away, date):
    """openfootball dates are local kickoff dates; Polymarket slugs use the UTC date, so evening
    kickoffs in North America fall on the next calendar day. Both dates are tried."""
    d0 = dt.date.fromisoformat(date)
    dates = [str(d0), str(d0 + dt.timedelta(days=1))]
    for d in dates:
        slug = f"fifwc-{CODE[home]}-{CODE[away]}-{d}"
        ev = get(f"{GAMMA}/events", params={"slug": slug}).json()
        if ev:
            return ev[0]
        time.sleep(0.2)
    # fallback: text search, keep fifwc events on either date
    for q in (f"{home} vs. {away}", f"{ALT.get(home, home)} vs. {ALT.get(away, away)}"):
        res = get(f"{GAMMA}/public-search", params={"q": q, "limit_per_type": 10}).json()
        for e in res.get("events", []):
            s = e.get("slug", "")
            if s.startswith("fifwc-") and any(s.endswith(d) for d in dates):
                full = get(f"{GAMMA}/events", params={"slug": s}).json()
                if full:
                    return full[0]
        time.sleep(0.5)
    return None


def classify_markets(markets, home, away):
    """Return [(market, type)] with type in home/draw/away.
    Polymarket titles use some alternative spellings (Türkiye, IR Iran, Côte d'Ivoire, Cabo Verde,
    Korea Republic), so name matching is tried first and event order (home, draw, away) is the fallback."""
    def team_in(q, team):
        cands = {team.lower(), ALT.get(team, team).lower()}
        return any(q.startswith(f"will {c} win") for c in cands)
    out, undecided = [], []
    for m in markets:
        q = m["question"].lower()
        if "draw" in q:
            out.append((m, "draw"))
        elif team_in(q, home):
            out.append((m, "home"))
        elif team_in(q, away):
            out.append((m, "away"))
        else:
            undecided.append(m)
    have = {t for _, t in out}
    for m in undecided:
        t = "home" if "home" not in have else "away"
        have.add(t)
        print(f"    [order fallback] '{m['question']}' -> {t}")
        out.append((m, t))
    return out


def fetch_match_markets(r):
    """Fetch the three markets and their price history for one match. Cached per match on disk
    so the script can be re-run safely after an interruption."""
    raw_dir = os.path.join(DATA_DIR, "polymarket_raw")
    os.makedirs(raw_dir, exist_ok=True)
    cache = os.path.join(raw_dir, f"{int(r.wc_match_id):03d}.json")
    if os.path.exists(cache):
        return json.load(open(cache))
    ev = find_event(r.home_team, r.away_team, r.date)
    if ev is None:
        raise RuntimeError(f"Polymarket event not found for match {r.wc_match_id}: {r.home_team} vs {r.away_team}")
    out = {"event_slug": ev["slug"], "markets": []}
    for m, mtype in classify_markets(ev["markets"], r.home_team, r.away_team):
        toks = json.loads(m.get("clobTokenIds", "[]"))
        res = json.loads(m.get("outcomePrices", "[0,0]"))
        hist = []
        if toks:
            hist = get(f"{CLOB}/prices-history",
                       params={"market": toks[0], "interval": "all", "fidelity": 720}).json().get("history", [])
            time.sleep(0.1)
        out["markets"].append({
            "market_type": mtype, "question": m["question"], "condition_id": m.get("conditionId"),
            "yes_token": toks[0] if toks else None, "market_open": m.get("startDate"),
            "kickoff_utc": m.get("endDate"), "closed_time": m.get("closedTime"),
            "resolved_yes": int(float(res[0]) >= 0.5), "volume_usd": m.get("volumeNum"), "history": hist,
        })
    json.dump(out, open(cache, "w"))
    print(f"  {int(r.wc_match_id):>3} {r.home_team} vs {r.away_team}: {ev['slug']}", flush=True)
    return out


def load_polymarket(wc, limit=None):
    markets, prices, done = [], [], 0
    for _, r in wc.iterrows():
        cached = os.path.exists(os.path.join(DATA_DIR, "polymarket_raw", f"{int(r.wc_match_id):03d}.json"))
        if not cached:
            if limit is not None and done >= limit:
                raise SystemExit(f"batch limit reached ({limit} matches fetched); run again to continue")
            done += 1
        data = fetch_match_markets(r)
        for m in data["markets"]:
            hist = m.pop("history")
            markets.append({"wc_match_id": int(r.wc_match_id), "event_slug": data["event_slug"], **m})
            for p in hist:
                prices.append({"wc_match_id": int(r.wc_match_id), "market_type": m["market_type"],
                               "ts": p["t"], "price": p["p"]})
    mk, px = pd.DataFrame(markets), pd.DataFrame(prices)
    bad = mk.groupby("wc_match_id").market_type.nunique()
    bad = bad[bad != 3]
    if len(bad):
        raise RuntimeError(f"matches without 3 distinct markets: {bad.index.tolist()}")
    px["timestamp_utc"] = pd.to_datetime(px["ts"], unit="s", utc=True).dt.strftime("%Y-%m-%d %H:%M:%S")
    mk.to_csv(os.path.join(DATA_DIR, "polymarket_markets.csv"), index=False)
    px.to_csv(os.path.join(DATA_DIR, "polymarket_prices.csv"), index=False)
    return mk, px


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    print("Source 1: historical results")
    hist = load_history()
    print(f"  {len(hist):,} matches, {hist.date.min().date()} to {hist.date.max().date()}")
    print("Source 3: World Cup 2026 fixtures")
    wc = load_wc2026()
    print(f"  {len(wc)} matches, {wc.result_90.value_counts().to_dict()}")
    print("Source 2: Polymarket markets")
    import sys
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    mk, px = load_polymarket(wc, limit=limit)
    print(f"  {len(mk)} markets, {len(px):,} price points")

    con = sqlite3.connect(DB_PATH)
    hist.assign(date=hist.date.dt.strftime("%Y-%m-%d")).to_sql("matches_history", con, if_exists="replace", index=False)
    wc.to_sql("wc2026_matches", con, if_exists="replace", index=False)
    mk.to_sql("polymarket_markets", con, if_exists="replace", index=False)
    px.to_sql("polymarket_prices", con, if_exists="replace", index=False)
    con.commit()
    for t in ["matches_history", "wc2026_matches", "polymarket_markets", "polymarket_prices"]:
        n = pd.read_sql(f"SELECT COUNT(*) n FROM {t}", con)["n"][0]
        cols = pd.read_sql(f"PRAGMA table_info({t})", con)["name"].tolist()
        print(f"{t}: {n:,} rows, {len(cols)} cols")
    con.close()


if __name__ == "__main__":
    main()
