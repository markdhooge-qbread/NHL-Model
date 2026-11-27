import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import clv  # noqa: E402

DK = {
    "ml": {("DAL", "STL"): {"DAL": -142, "STL": 120}},
    "total": {("DAL", "STL"): {"line": 5.5, "over": -122, "under": 102}},
    "player": {("DAL", "STL", "jasonrobertson", "SOG"): {"line": 3.5, "over": -110, "under": -120}},
    "p1ml": {("DAL", "STL"): {"DAL": 200, "STL": 230, "TIE": 120}},
    "p1pl": {("DAL", "STL"): {"DAL": {"line": 0.5, "odds": -150}, "STL": {"line": -0.5, "odds": 120}}},
    "p1tnb": {}, "p1total": {}, "f10": {},
}


def test_price_for_matches_side_and_line():
    o, other, imp = clv.price_for(DK, clv.make_ref("Total", "DAL @ STL", "Under", 5.5))
    assert (o, other) == (102, -122) and abs(imp - 0.4739) < 1e-3
    assert clv.price_for(DK, clv.make_ref("Total", "DAL @ STL", "Under", 6.5)) == (None, None, None)  # line moved
    o, other, imp = clv.price_for(DK, clv.make_ref("ML", "DAL @ STL", "STL"))
    assert (o, other) == (120, -142) and imp < 0.45
    o, _, _ = clv.price_for(DK, clv.make_ref("SOG", "DAL @ STL", "Over", 3.5, "Jason Robertson"))
    assert o == -110
    o, _, imp = clv.price_for(DK, clv.make_ref("P1 ML", "DAL @ STL", "TIE"))
    assert o == 120 and 0.3 < imp < 0.45  # 3-way no-vig
    o, other, _ = clv.price_for(DK, {**clv.make_ref("P1 PL", "DAL @ STL", ""), "side": "DAL|0.5"})
    assert (o, other) == (-150, 120)
    assert clv.price_for(None, clv.make_ref("ML", "DAL @ STL", "STL")) == (None, None, None)


def _value(pick, market, side, line, odds, imp, p, edge, player=None):
    return {"pick": pick, "market": market, "game": "DAL @ STL", "odds": odds, "other": None, "imp": imp, "p": p,
            "edge": edge, "ref": clv.make_ref(market, "DAL @ STL", side, line, player)}


def test_snapshot_close_and_clv_math():
    vals = [_value("DAL @ STL U 5.5", "Total", "Under", 5.5, 102, 0.4739, 0.6455, 17.16),
            _value("STL ML (DAL @ STL)", "ML", "STL", None, 130, 0.42, 0.47, 5.0),
            _value("small", "ML", "DAL", None, -150, 0.58, 0.60, 2.0)]  # below the 3-point floor
    snap = clv.candidates_snapshot(vals, vals[0], "2026-11-27", "09:54 ET", 3.0, 0.35)
    assert list(snap["pick"]) == ["DAL @ STL U 5.5", "STL ML (DAL @ STL)"] and list(snap["is_top"]) == [1, 0]
    out = clv.with_close(snap, DK, "now")
    u = out.iloc[0]
    assert abs(u["clv_pts"] - 0.0) < 0.05          # same price at the close
    assert abs(u["clv_price_pct"] - (2.02 * u["close_imp"] - 1) * 100) < 0.05
    ml = out.iloc[1]
    assert ml["clv_pts"] > 0                         # took +130, closed +120: market moved our way
    none = clv.with_close(snap, None, "now")
    assert (none["close_note"] == "no saved feed lines").all() and (none["clv_pts"] == "").all()


def test_append_only():
    snap = clv.candidates_snapshot([_value("a", "ML", "STL", None, 120, 0.45, 0.5, 5)], None, "2026-11-25", "x", 3, 0.35)
    rows = clv.with_close(snap, DK, "t1")
    log, n = clv.append_log(pd.DataFrame(), rows)
    assert n == 1
    again = clv.with_close(snap.assign(odds=999), DK, "t2")  # a later attempt for the same date must not overwrite
    log2, n2 = clv.append_log(log, again)
    assert n2 == 0 and log2.equals(log)
    later = clv.with_close(snap.assign(date="2026-11-26"), DK, "t3")
    log3, n3 = clv.append_log(log2, later)
    assert n3 == 1 and log3.iloc[0].equals(log.iloc[0])


def test_ref_from_label_for_old_top_picks():
    r = clv.ref_from_label("DAL @ STL U 5.5", "Total", "DAL @ STL")
    assert (r["ref_mkt"], r["side"], r["line"], r["away"], r["home"]) == ("TOT", "Under", 5.5, "DAL", "STL")
    r = clv.ref_from_label("Shea Theodore (VGK) U 0.5 Pts", "Pts", "VGK @ EDM")
    assert (r["ref_mkt"], r["side"], r["line"], r["player"]) == ("Pts", "Under", 0.5, "Shea Theodore")
    r = clv.ref_from_label("EDM ML (VGK @ EDM)", "ML", "VGK @ EDM")
    assert r["side"] == "EDM"
    assert clv.ref_from_label("weird", "P1 ML", "VGK @ EDM") is None
    row = clv.top_pick_as_candidate({"date": "2026-11-24", "pick": "Shea Theodore (VGK) U 0.5 Pts", "market": "Pts",
                                     "game": "VGK @ EDM", "odds": "-210", "p": "0.7793", "imp": "0.6334",
                                     "edge": "14.59", "saved_at": "12:59 ET"})
    assert row["is_top"] == 1 and row["odds"] == -210


def test_grade_and_summary():
    fin = {"a": 2, "h": 3, "p1": {"a1": 1, "h1": 0, "first": 4.5},
           "players": {clv.short_key("C. McDavid"): {"g": 1, "pts": 2, "sog": 4}}}
    g = lambda m, side, line=None, player=None: clv.grade(clv.make_ref(m, "DAL @ STL", side, line, player), fin)  # noqa: E731
    assert g("ML", "STL") == "hit" and g("ML", "DAL") == "miss"
    assert g("Total", "Under", 5.5) == "hit" and g("Total", "Over", 5.0) == "push"
    assert g("SOG", "Over", 3.5, "Connor McDavid") == "hit" and g("Pts", "Under", 1.5, "Connor McDavid") == "miss"
    assert g("Goal", "Over", 0.5, "Nobody Here") is None
    assert g("P1 ML", "DAL") == "hit" and g("P1 ML", "TIE") == "miss" and g("F10", "Over", 0.5) == "hit"
    assert clv.grade({**clv.make_ref("P1 PL", "DAL @ STL", ""), "side": "STL|0.5"}, fin) == "miss"
    assert clv.grade(clv.make_ref("ML", "DAL @ STL", "STL"), None) is None
    log = pd.DataFrame([{"date": "d", "pick": "a", "clv_pts": 2.0, "clv_price_pct": 3.0},
                        {"date": "d", "pick": "b", "clv_pts": -1.0, "clv_price_pct": -2.0},
                        {"date": "d", "pick": "c", "clv_pts": "", "clv_price_pct": ""}])
    s = clv.summary(log, {("d", "a"): "hit", ("d", "b"): "miss"})
    assert s["n"] == 2 and abs(s["avg_pts"] - 0.5) < 1e-9 and s["beat_close"] == 0.5
    assert [b["hit_rate"] for b in s["by_sign"]] == [1.0, 0.0]
