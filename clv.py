"""Closing line value (CLV): did the price we picked beat DraftKings' last pregame price?

Two files on the picks-data branch:
  * picks/candidates/DATE.csv  the pick-time snapshot: tonight's top pick and every Best bets candidate
                               (edge 3+, model 35%+), with price and no-vig chance. Rewritten with the top
                               pick before the first puck drop, frozen after it, like picks/top_picks.csv.
  * picks/clv_log.csv          append-only. Once a date is over, each snapshot row gets the last pregame
                               free-feed price for its market (lines/auto/DATE.csv) and its CLV, and the
                               rows are appended. Rows already in the log are never rewritten.

CLV in points  = last pregame no-vig chance - no-vig chance at pick (positive = the market moved our way).
CLV in price % = (decimal odds we got) x (last pregame no-vig chance) - 1, the return we'd expect if the
                 closing price is fair.
Pure Python, no Streamlit, so it can be tested on its own.
"""
import math
import re
import unicodedata

import pandas as pd

from model_core import implied_3way, implied_no_vig

CAND_COLS = ["date", "saved_at", "is_top", "pick", "market", "game", "ref_mkt", "away", "home", "side", "line",
             "player", "odds", "other_odds", "imp", "p", "edge"]
LOG_COLS = CAND_COLS + ["close_odds", "close_other_odds", "close_imp", "clv_pts", "clv_price_pct", "close_note",
                        "logged_at"]
# Best bets market name -> lines-dict market key used in ref_mkt
REF_MKT = {"ML": "ML", "Total": "TOT", "P1 Total": "P1TOT", "P1 ML": "P1ML", "F10": "F10", "P1 PL": "P1PL",
           "P1 TNB": "P1TNB", "SOG": "SOG", "Pts": "Pts", "Goal": "Goal"}


def _norm(name):
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _num(v):
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)) or str(v).strip() == "":
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _int(v):
    x = _num(v)
    return None if x is None else int(x)


def decimal_odds(american):
    return 1 + (american / 100.0 if american > 0 else 100.0 / abs(american))


def make_ref(market, game, side, line=None, player=None):
    """How to find a pick's price again in a lines dict: market key, teams, side, line, player."""
    away, _, home = str(game).replace(" ", "").partition("@")
    return {"ref_mkt": REF_MKT.get(market, market), "away": away, "home": home, "side": side,
            "line": line, "player": player or ""}


def price_for(dk, ref):
    """(odds, other side's odds, no-vig chance) for a ref in a lines dict (app.lines_from_df shape).
    Two-way markets need the same line as at pick time; a moved line returns (None, None, None)."""
    if not dk:
        return None, None, None
    m, key, side, line = ref["ref_mkt"], (ref["away"], ref["home"]), ref["side"], _num(ref.get("line"))

    def two_way(rec):
        if not rec or rec.get("line") is None or line is None or abs(rec["line"] - line) > 1e-9:
            return None, None, None
        mine, other = (rec.get("over"), rec.get("under")) if side == "Over" else (rec.get("under"), rec.get("over"))
        if mine is None:
            return None, None, None
        return mine, other, implied_no_vig(mine, other)

    if m == "ML":
        ml = dk.get("ml", {}).get(key, {})
        opp = ref["home"] if side == ref["away"] else ref["away"]
        mine, other = ml.get(side), ml.get(opp)
        return (mine, other, implied_no_vig(mine, other)) if mine is not None else (None, None, None)
    if m == "TOT":
        return two_way(dk.get("total", {}).get(key))
    if m == "P1TOT":
        return two_way(dk.get("p1total", {}).get(key))
    if m == "F10":
        return two_way(dk.get("f10", {}).get(key))
    if m == "P1ML":
        prices = dk.get("p1ml", {}).get(key, {})
        return (prices[side], None, implied_3way(prices, side)) if prices.get(side) is not None else (None, None, None)
    if m == "P1PL":
        team, pl_line = (side.split("|") + [None])[:2] if isinstance(side, str) else (None, None)
        pl = dk.get("p1pl", {}).get(key, {})
        rec = pl.get(team)
        if not rec or _num(pl_line) is None or abs(rec["line"] - float(pl_line)) > 1e-9:
            return None, None, None
        opp = ref["home"] if team == ref["away"] else ref["away"]
        other = (pl.get(opp) or {}).get("odds")
        return rec["odds"], other, implied_no_vig(rec["odds"], other)
    if m == "P1TNB":
        tnb = dk.get("p1tnb", {}).get(key, {})
        opp = ref["home"] if side == ref["away"] else ref["away"]
        return (tnb[side], tnb.get(opp), implied_no_vig(tnb[side], tnb.get(opp))) if tnb.get(side) is not None \
            else (None, None, None)
    if m in ("SOG", "Pts", "Goal"):
        players = dk.get("player", {})
        nm = _norm(ref.get("player", ""))
        rec = players.get((ref["away"], ref["home"], nm, m)) or players.get((nm, m))
        return two_way(rec)
    return None, None, None


def candidate_row(v, date_str, saved_at, is_top):
    """Snapshot row for one Best bets value dict (needs v['ref'])."""
    ref = v.get("ref") or {}
    side = ref.get("side")
    if isinstance(side, tuple):  # 1st-period puck line sides are (team, line)
        side = f"{side[0]}|{side[1]:g}"
    return {"date": date_str, "saved_at": saved_at, "is_top": int(bool(is_top)), "pick": v["pick"],
            "market": v["market"], "game": v.get("game", ""), "ref_mkt": ref.get("ref_mkt", ""),
            "away": ref.get("away", ""), "home": ref.get("home", ""), "side": side if side is not None else "",
            "line": ref.get("line") if ref.get("line") is not None else "", "player": ref.get("player", ""),
            "odds": int(v["odds"]) if v.get("odds") is not None else "", "other_odds": v.get("other", "") or "",
            "imp": round(v["imp"], 4) if v.get("imp") is not None else "", "p": round(v["p"], 4),
            "edge": round(v["edge"], 2) if v.get("edge") is not None else ""}


def candidates_snapshot(values, top, date_str, saved_at, min_edge, min_p):
    """Every Best bets candidate clearing the board's default filters, the top pick flagged."""
    rows = [candidate_row(v, date_str, saved_at, top is not None and v["pick"] == top["pick"])
            for v in values if v.get("edge") is not None and v["edge"] >= min_edge and v["p"] >= min_p]
    rows.sort(key=lambda r: (-r["is_top"], -(r["edge"] or 0)))
    return pd.DataFrame(rows, columns=CAND_COLS)


_PL = re.compile(r"^(?P<who>.+) \((?P<team>[A-Z]{2,3})\) (?P<ou>[OU]) (?P<line>[\d.]+) (?P<mkt>SOG|Pts)$")
_ATG = re.compile(r"^(?P<who>.+) \((?P<team>[A-Z]{2,3})\) anytime goal$")
_ML = re.compile(r"^(?P<team>[A-Z]{2,3}) ML \((?P<game>[A-Z]{2,3} @ [A-Z]{2,3})\)$")
_TOT = re.compile(r"^(?P<game>[A-Z]{2,3} @ [A-Z]{2,3}) (?P<ou>[OU]) (?P<line>[\d.]+)$")


def ref_from_label(pick, market, game):
    """Rebuild a ref from a saved pick label, for top picks saved before refs were stored. None if unknown."""
    if market == "ML" and (m := _ML.match(pick)):
        return make_ref("ML", m["game"], m["team"])
    if market == "Total" and (m := _TOT.match(pick)):
        return make_ref("Total", m["game"], "Over" if m["ou"] == "O" else "Under", float(m["line"]))
    if market in ("SOG", "Pts") and (m := _PL.match(pick)):
        return make_ref(market, game, "Over" if m["ou"] == "O" else "Under", float(m["line"]), m["who"])
    if market == "Goal" and (m := _ATG.match(pick)):
        return make_ref("Goal", game, "Over", 0.5, m["who"])
    return None


def top_pick_as_candidate(r):
    """A top_picks.csv row as a snapshot row (no candidates file that day)."""
    ref = ref_from_label(str(r["pick"]), str(r["market"]), str(r["game"])) or make_ref(r["market"], r["game"], "")
    return {"date": r["date"], "saved_at": r.get("saved_at", ""), "is_top": 1, "pick": r["pick"],
            "market": r["market"], "game": r["game"], **ref, "odds": _int(r["odds"]), "other_odds": "",
            "imp": _num(r["imp"]), "p": _num(r["p"]), "edge": _num(r["edge"])}


def with_close(cands, dk_close, logged_at):
    """Add last-pregame price and CLV to snapshot rows."""
    out = []
    for _, r in cands.iterrows():
        row = {c: r.get(c, "") for c in CAND_COLS}
        ref = {k: r.get(k, "") for k in ("ref_mkt", "away", "home", "side", "line", "player")}
        odds_c, other_c, imp_c = price_for(dk_close, ref) if ref["ref_mkt"] and ref["side"] != "" else (None, None, None)
        imp_p, odds_p = _num(row["imp"]), _int(row["odds"])
        note = ""
        if dk_close is None:
            note = "no saved feed lines"
        elif imp_c is None:
            note = "no matching last pregame price (line moved or market gone)"
        clv_pts = round((imp_c - imp_p) * 100, 2) if imp_c is not None and imp_p is not None else None
        clv_px = round((decimal_odds(odds_p) * imp_c - 1) * 100, 2) if imp_c is not None and odds_p else None
        row.update({"close_odds": odds_c if odds_c is not None else "", "close_other_odds": other_c if other_c is not None else "",
                    "close_imp": round(imp_c, 4) if imp_c is not None else "",
                    "clv_pts": clv_pts if clv_pts is not None else "", "clv_price_pct": clv_px if clv_px is not None else "",
                    "close_note": note, "logged_at": logged_at})
        out.append(row)
    return pd.DataFrame(out, columns=LOG_COLS)


def append_log(log, new_rows):
    """Append-only: add rows only for dates the log doesn't have yet. Existing rows come back unchanged."""
    log = log.reindex(columns=LOG_COLS) if len(log) else pd.DataFrame(columns=LOG_COLS)
    have = set(log["date"].astype(str))
    add = new_rows[~new_rows["date"].astype(str).isin(have)]
    if add.empty:
        return log, 0
    return pd.concat([log, add.reindex(columns=LOG_COLS)], ignore_index=True), len(add)


def summary(log, results=None):
    """Sample size, average CLV, share beating the close, and hit rate split by CLV sign.
    results: optional {(date, pick): 'hit'|'miss'|'push'|...}."""
    d = log.copy()
    d["clv_pts"] = pd.to_numeric(d["clv_pts"], errors="coerce")
    d["clv_price_pct"] = pd.to_numeric(d["clv_price_pct"], errors="coerce")
    d = d.dropna(subset=["clv_pts"])
    out = {"n": len(d), "avg_pts": d["clv_pts"].mean() if len(d) else None,
           "avg_price_pct": d["clv_price_pct"].mean() if len(d) else None,
           "beat_close": (d["clv_pts"] > 0).mean() if len(d) else None, "by_sign": []}
    if results is not None and len(d):
        d["result"] = [results.get((str(r["date"]), str(r["pick"]))) for _, r in d.iterrows()]
        for lbl, m in (("Beat the close (CLV > 0)", d["clv_pts"] > 0), ("No better (CLV <= 0)", d["clv_pts"] <= 0)):
            sub = d[m & d["result"].isin(["hit", "miss"])]
            w = int((sub["result"] == "hit").sum())
            out["by_sign"].append({"group": lbl, "bets": int(m.sum()), "graded": len(sub), "wins": w,
                                   "hit_rate": w / len(sub) if len(sub) else None})
    return out


def short_key(name):
    """'Connor McDavid' and the box score's 'C. McDavid' both -> 'cmcdavid'."""
    parts = str(name).replace(".", ". ").split()
    if not parts:
        return ""
    return _norm(parts[0][0] + parts[-1]) if len(parts) > 1 else _norm(parts[0])


def _vs_line(x, line, side):
    if x is None or line is None:
        return None
    if x == line:
        return "push"
    return "hit" if (x > line) == (side == "Over") else "miss"


def grade(ref, fin):
    """Result of a logged pick from the final: 'hit', 'miss', 'push', or None if unknown.
    fin = {"a": away goals, "h": home goals, "p1": {"a1", "h1", "first"} or None,
           "players": {short_key(name): {"g", "pts", "sog"}}}"""
    if not fin:
        return None
    m, side, line = ref.get("ref_mkt"), ref.get("side"), _num(ref.get("line"))
    away, home = ref.get("away"), ref.get("home")
    a, h, p1 = fin.get("a"), fin.get("h"), fin.get("p1")
    if m == "ML":
        mine, theirs = (a, h) if side == away else (h, a)
        return "hit" if mine > theirs else "miss"
    if m == "TOT":
        return _vs_line(a + h, line, side)
    if m in ("P1TOT", "P1ML", "F10", "P1PL", "P1TNB"):
        if not p1:
            return None
        a1, h1 = p1["a1"], p1["h1"]
        if m == "P1TOT":
            return _vs_line(a1 + h1, line, side)
        if m == "F10":
            scored = p1.get("first") is not None and p1["first"] < 10
            return "hit" if scored == (side == "Over") else "miss"
        if m == "P1ML":
            won = "TIE" if a1 == h1 else (away if a1 > h1 else home)
            return "hit" if won == side else "miss"
        team, pl_line = (str(side).split("|") + ["0"])[:2] if m == "P1PL" else (side, "0")
        mine, theirs = (a1, h1) if team == away else (h1, a1)
        margin = mine - theirs + float(pl_line)
        if m == "P1TNB" and a1 == h1:
            return "push"
        return "push" if margin == 0 else ("hit" if margin > 0 else "miss")
    if m in ("SOG", "Pts", "Goal"):
        st = (fin.get("players") or {}).get(short_key(ref.get("player", "")))
        if st is None:
            return None
        x = {"SOG": st.get("sog"), "Pts": st.get("pts"), "Goal": st.get("g")}[m]
        return _vs_line(x, line if line is not None else 0.5, side)
    return None
