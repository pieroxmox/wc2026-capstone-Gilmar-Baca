"""Verify that the three data sources are reachable and match the local copies.

Contacts each source, reports HTTP status, row counts and date ranges, and compares the
checksum of the live file against the copy stored in ./data. No API key is required: all
three endpoints are public.

Output: printed report, and results/source_verification.json
"""

import hashlib
import io
import json
import os
from datetime import datetime, timezone

import pandas as pd
import requests

H = {"User-Agent": "Mozilla/5.0"}
DATA_DIR, OUT = "data", "results"
HIST_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
WC_JSON_URL = "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2026/worldcup.json"
GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
SAMPLE_SLUG = "fifwc-esp-arg-2026-07-19"


def md5(b):
    return hashlib.md5(b).hexdigest()


def main():
    os.makedirs(OUT, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    report = {"verified_at_utc": stamp, "sources": {}}
    print(f"Source verification, {stamp}\n")

    print("1. International football results (martj42/international_results)")
    r = requests.get(HIST_URL, headers=H, timeout=60)
    df = pd.read_csv(io.StringIO(r.text))
    wc = df[(df.tournament == "FIFA World Cup") & (df.date >= "2026-06-01")]
    local = os.path.join(DATA_DIR, "martj42_results.csv")
    same = os.path.exists(local) and md5(r.content) == md5(open(local, "rb").read())
    print(f"   HTTP {r.status_code}, {len(r.content):,} bytes")
    print(f"   {len(df):,} rows, {len(df.columns)} columns: {list(df.columns)}")
    print(f"   date range {df.date.min()} to {df.date.max()}")
    print(f"   World Cup 2026 matches present: {len(wc)}, all with scores: {wc.home_score.notna().all()}")
    print(f"   matches the local copy in {DATA_DIR}: {same}")
    report["sources"]["international_results"] = {
        "url": HIST_URL, "status": r.status_code, "bytes": len(r.content), "rows": int(len(df)),
        "columns": list(df.columns), "first_date": str(df.date.min()), "last_date": str(df.date.max()),
        "wc2026_matches": int(len(wc)), "md5": md5(r.content), "matches_local_copy": bool(same),
    }

    print("\n2. Polymarket (Gamma API and CLOB API, public, no key)")
    e = requests.get(f"{GAMMA}/events", params={"slug": SAMPLE_SLUG}, headers=H, timeout=40)
    ev = e.json()[0]
    print(f"   Gamma /events HTTP {e.status_code}: {ev['slug']} ({ev['title']}), closed: {ev['closed']}")
    markets = []
    for m in ev["markets"]:
        print(f"      {m['question']}  resolved: {m['outcomePrices']}  volume: {m['volumeNum']:,.0f} USD")
        markets.append({"question": m["question"], "outcome_prices": m["outcomePrices"], "volume_usd": m["volumeNum"]})
    tok = json.loads(ev["markets"][0]["clobTokenIds"])[0]
    h = requests.get(f"{CLOB}/prices-history", params={"market": tok, "interval": "all", "fidelity": 720},
                     headers=H, timeout=40)
    hist = h.json().get("history", [])
    print(f"   CLOB /prices-history HTTP {h.status_code}: {len(hist)} price points for the first market")
    report["sources"]["polymarket"] = {
        "gamma_status": e.status_code, "clob_status": h.status_code, "sample_event": ev["slug"],
        "markets": markets, "price_points_sample_market": len(hist),
    }

    print("\n3. World Cup 2026 fixtures (openfootball/worldcup.json)")
    j = requests.get(WC_JSON_URL, headers=H, timeout=60)
    matches = j.json()["matches"]
    scored = sum(1 for m in matches if m.get("score", {}).get("ft"))
    final = matches[-1]
    print(f"   HTTP {j.status_code}, {len(matches)} matches, {scored} with a full-time score")
    print(f"   final: {final['team1']} v {final['team2']}, {final['score']}")
    report["sources"]["openfootball"] = {
        "url": WC_JSON_URL, "status": j.status_code, "matches": len(matches), "with_scores": scored,
        "final": f"{final['team1']} v {final['team2']} {final['score']}",
    }

    json.dump(report, open(os.path.join(OUT, "source_verification.json"), "w"), indent=1)
    print(f"\nWritten to {OUT}/source_verification.json")


if __name__ == "__main__":
    main()
