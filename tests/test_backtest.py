import os
import sys

import numpy as np
import pandas as pd

sys.path[:0] = [os.path.dirname(os.path.dirname(os.path.abspath(__file__))), os.path.dirname(os.path.abspath(__file__))]

import fixtures as F  # noqa: E402
from backtests import asof, data, metrics, run_backtest as rb  # noqa: E402
from model_core import Model  # noqa: E402


def test_teams_asof_uses_only_earlier_games():
    tg, gg, res = F.synthetic_season(days=10)
    d = res["date"].iloc[len(res) // 2]
    t = asof.teams_asof(tg, d)
    used = tg[tg["gameDate"] < asof.date_int(d)]
    assert t["games_played"].max() == used.groupby("playerTeam")["gameId"].nunique().max()
    tor = t[(t["team"] == "TOR") & (t["situation"] == "5on5")]
    assert np.isclose(tor["xGoalsFor"].iloc[0], used[(used["playerTeam"] == "TOR") & (used["situation"] == "5on5")]["xGoalsFor"].sum())
    assert asof.teams_asof(tg, res["date"].min()).empty  # nothing known before opening night


def test_goalies_asof_and_starters():
    tg, gg, res = F.synthetic_season(days=20)
    d = res["date"].iloc[-1]
    g = asof.goalies_asof(gg, d)
    assert set(g.columns) >= {"playerId", "name", "team", "situation", "games_played", "icetime", "xGoals", "goals"}
    assert (g["games_played"] >= 1).all()
    st = asof.actual_starters(gg)
    gid = int(res["id"].iloc[0])
    assert (gid, res["away"].iloc[0] if res["away"].iloc[0] != "TBL" else "T.B") in st


def test_asof_tables_feed_the_live_model():
    tg, gg, res = F.synthetic_season(days=20)
    d = res["date"].iloc[-1]
    m = Model(asof.teams_asof(tg, d), asof.goalies_asof(gg, d), asof.empty_skaters(), set(), n_sims=2000)
    sim = m.simulate_game("TBL", "TOR", m.get_goalie("TBL"), m.get_goalie("TOR"), seed=1)
    assert 0 < sim["h_win"] < 1 and abs(sim["a_win"] + sim["h_win"] - 1) < 1e-9
    assert 3 < sim["proj_tot"] < 9


def test_metrics():
    assert np.isclose(metrics.log_loss_vec([0.5], [1])[0], np.log(2))
    assert np.isclose(metrics.brier_vec([0.7], [0])[0], 0.49)
    mean, se = metrics.paired_diff([1, 2, 3], [1, 1, 1])
    assert np.isclose(mean, 1.0) and se > 0
    cal = metrics.calibration_table([0.1, 0.12, 0.6], [0, 1, 1], [0, 0.5, 1.0])
    assert list(cal["games"]) == [2, 1]


def test_parse_score_day():
    js = {"games": [
        {"id": 1, "gameType": 2, "gameState": "OFF", "gameDate": "2024-10-10",
         "awayTeam": {"abbrev": "TOR", "score": 3}, "homeTeam": {"abbrev": "BOS", "score": 2},
         "gameOutcome": {"lastPeriodType": "SO"},
         "goals": [{"periodDescriptor": {"number": 1}, "timeInPeriod": "04:10", "teamAbbrev": {"default": "TOR"}},
                   {"periodDescriptor": {"number": 1}, "timeInPeriod": "15:00", "teamAbbrev": "BOS"},
                   {"periodDescriptor": {"number": 2}, "timeInPeriod": "01:00", "teamAbbrev": "TOR"}]},
        {"id": 2, "gameType": 1, "gameState": "OFF", "awayTeam": {"abbrev": "A"}, "homeTeam": {"abbrev": "B"}},
        {"id": 3, "gameType": 2, "gameState": "FUT", "awayTeam": {"abbrev": "A"}, "homeTeam": {"abbrev": "B"}},
    ]}
    rows = data.parse_score_day(js, "2024-10-10")
    assert len(rows) == 1
    r = rows[0]
    assert (r["a_goals"], r["h_goals"], r["last"], r["p1_a"], r["p1_h"], r["first_goal_p1_sec"]) == (3, 2, "SO", 1, 1, 250)
    no_goals = data.parse_score_day({"games": [{**js["games"][0], "goals": []}]}, "x")[0]
    assert no_goals["first_goal_p1_sec"] == 9999 and no_goals["p1_a"] == 0
    missing = data.parse_score_day({"games": [{k: v for k, v in js["games"][0].items() if k != "goals"}]}, "x")[0]
    assert missing["p1_a"] is None


def test_variants_include_live_once_and_respect_pairs():
    v = rb.variants(rb.GRID)
    labels = [x[0] for x in v]
    assert labels[0] == "live" and len(labels) == len(set(labels))
    assert not any(lbl == "gamma_shape=30" for lbl in labels)  # the live value runs as "live"
    assert rb.variant_params("other_goals", 0.5) == {"other_a": 0.12, "other_h": 0.135}
    assert rb.variant_params("ot_xg_weight", 0.6) == {"ot_xg_weight": 0.6, "ot_base": 0.2}


def test_end_to_end_on_a_synthetic_season(tmp_path):
    grid = {"gamma_shape": [6.0, 30.0], "home_ice": [0.0, 0.035]}
    vl = rb.variants(grid)
    parts = []
    for season in (2022, 2023):
        tg, gg, res = F.synthetic_season(season=season, days=60, seed=season)
        _, _, prev = F.synthetic_season(season=season - 1, days=60, seed=season - 1)
        parts.append(rb.run_season(season, res, prev, tg, gg, vl, n_sims=500))
    df = pd.concat([p[0] for p in parts], ignore_index=True)
    pmfs = {k: v for p in parts for k, v in p[1].items()}
    assert set(df["variant"]) == {"live", "baseline", "goalie=actual starter", "gamma_shape=6", "home_ice=0"}
    lines = tmp_path / "2023-11-01.csv"
    g = df[(df["season"] == 2023) & (df["variant"] == "live")].iloc[0]
    d = g["date"]
    lines.write_text(f"game,market,player,line,over_odds,under_odds\n{g['away']}@{g['home']},ML,{g['away']},,+120,\n"
                     f"{g['away']}@{g['home']},ML,{g['home']},,-140,\n{g['away']}@{g['home']},TOTAL,,6.5,-110,-110\n")
    os.rename(lines, tmp_path / f"{d}.csv")
    out = tmp_path / "res"
    _, pt, rec = rb.write_report(df, pmfs, grid, [2022], 2023, 0, 5, 500, [str(tmp_path / f"{d}.csv")], out=str(out))
    text = (out / "README.md").read_text()
    for h in ("Model vs a plain baseline", "Calibration", "Goal-total distribution", "one at a time", "Against DraftKings"):
        assert h in text
    assert "No live parameter was changed" in text
    assert set(rec["param"]) == {"gamma_shape", "home_ice"}
    assert (out / "market_check.csv").exists()
    assert pt[pt["live"]]["tuning_diff"].eq(0).all()


def test_recommend_rule():
    pt = pd.DataFrame([
        {"param": "x", "value": 1.0, "live": True, "tuning_diff": 0.0, "held-out_diff": 0.0, "held-out_se": 0.0},
        {"param": "x", "value": 2.0, "live": False, "tuning_diff": -0.01, "held-out_diff": -0.01, "held-out_se": 0.002},
        {"param": "y", "value": 1.0, "live": True, "tuning_diff": 0.0, "held-out_diff": 0.0, "held-out_se": 0.0},
        {"param": "y", "value": 2.0, "live": False, "tuning_diff": -0.01, "held-out_diff": -0.001, "held-out_se": 0.002},
        {"param": "z", "value": 1.0, "live": True, "tuning_diff": 0.0, "held-out_diff": 0.0, "held-out_se": 0.0},
        {"param": "z", "value": 2.0, "live": False, "tuning_diff": 0.01, "held-out_diff": -0.01, "held-out_se": 0.001},
    ])
    r = rb.recommend(pt).set_index("param")["recommendation"]
    assert r["x"].startswith("Evidence") and r["y"].startswith("Weak") and r["z"].startswith("Keep")


def test_backup_params_use_the_actual_starter():
    v = {x[0]: x for x in rb.variants(rb.GRID)}
    assert v["backup_drag=0"][4] == "actual" and v["gamma_shape=6"][4] == "assumed"


def test_live_parameters_approved_nov_27():
    from model_core import DEFAULT_PARAMS as P
    assert (P["gamma_shape"], P["other_a"], P["other_h"]) == (30.0, 0.24, 0.27)  # changes 1 and 2, approved by Mark
    assert (P["backup_prior"], P["backup_mean"], P["backup_drag"]) == (20.0, -0.18, -0.10)  # change 3 not approved
