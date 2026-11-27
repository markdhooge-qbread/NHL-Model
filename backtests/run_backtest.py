"""Backtest the game simulation on past seasons, using only data available before each game.

    python -m backtests.run_backtest                      # full run (tuning 2022-2024, held-out 2025, current 2026)
    python -m backtests.run_backtest --quick              # 2,000 sims and a smaller grid, for a fast check

For every regular-season game it rebuilds MoneyPuck team and goalie tables as of that morning
(season to date, see asof.py), runs model_core.Model exactly as the site does, and scores the
forecast against the final. Each parameter in GRID is moved one at a time with the rest at their
live values. Results go to backtests/results/. Nothing here changes the live parameters.
"""
import argparse
import datetime
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from backtests import asof, data, metrics  # noqa: E402
from model_core import DEFAULT_PARAMS, Model, implied_no_vig, pmf_over_under  # noqa: E402

OUT = os.path.join(ROOT, "backtests", "results")
TOT_K = 16          # store P(total = 0..15); 16+ never happens in practice
BASELINE_POOL = 1312  # one season of games for the league-average baseline

# One parameter at a time. Every list includes the live value. Chosen before looking at any result.
GRID = {
    "gamma_shape": [6.0, 10.0, 18.0, 30.0, 60.0, 1e6],   # 1e6 = plain Poisson
    "b2b_off": [0.90, 0.94, 0.97, 1.00],
    "b2b_def": [1.00, 1.02, 1.04, 1.06],
    "home_ice": [0.0, 0.02, 0.035, 0.05],
    "other_goals": [0.0, 0.5, 1.0, 1.5],     # multiplier on the live "other goals" (+0.24 away / +0.27 home)
    "ot_xg_weight": [0.0, 0.3, 0.6, 0.9],    # OT base = (1 - weight) / 2, so an even game stays 50/50
    "starter_prior": [5.0, 15.0, 30.0, 60.0],
    "backup_prior": [10.0, 20.0, 40.0],
    "backup_mean": [0.0, -0.18, -0.30],
    "backup_drag": [0.0, -0.10, -0.20],
    "p1_share": [0.27, 0.30, 0.33],
    "f10_share": [0.42, 0.48, 0.54],
}
QUICK_GRID = {k: v for k, v in GRID.items() if k in ("gamma_shape", "home_ice", "b2b_off")}
LIVE = {"other_goals": 1.0, **{k: DEFAULT_PARAMS[k] for k in GRID if k in DEFAULT_PARAMS}}
# The score each parameter is judged on, fixed up front: the thing the parameter is meant to move.
PRIMARY = {"gamma_shape": "total_ll", "other_goals": "total_ll", "p1_share": "p1_ll", "f10_share": "f10_ll"}
# Backup-goalie settings only matter when a backup starts, which the assumed-#1 run never has. They are
# tested with the goalie who actually started, against the live values with that same goalie.
ACTUAL_GOALIE = ("backup_prior", "backup_mean", "backup_drag")
ACTUAL_REF = "goalie=actual starter"
LOSSES = ["win_ll", "win_brier", "o55_ll", "o65_ll", "total_ll", "p1_ll", "f10_ll"]
LOSS_LABEL = {"win_ll": "win log loss", "win_brier": "win Brier", "o55_ll": "O/U 5.5 log loss",
              "o65_ll": "O/U 6.5 log loss", "total_ll": "exact-total log loss", "p1_ll": "1st-period goals log loss",
              "f10_ll": "goal in first 10 log loss"}


def variant_params(name, value):
    if name == "other_goals":
        return {"other_a": DEFAULT_PARAMS["other_a"] * value, "other_h": DEFAULT_PARAMS["other_h"] * value}
    if name == "ot_xg_weight":
        return {"ot_xg_weight": value, "ot_base": (1 - value) / 2}
    return {name: value}


def variants(grid):
    """(label, param name, value, params, goalie mode). 'live' first."""
    out = [("live", "live", None, {}, "assumed"), ("goalie=actual starter", "goalie", "actual", {}, "actual")]
    for name, vals in grid.items():
        for v in vals:
            if not np.isclose(v, LIVE[name]):
                out.append((f"{name}={v:g}", name, v, variant_params(name, v),
                            "actual" if name in ACTUAL_GOALIE else "assumed"))
    return out


def _p_at(pmf, k):
    return float(pmf[k]) if k < len(pmf) else 0.0


def game_record(sim, res):
    """Forecasts and per-game losses for one simulated game against its result."""
    tot = res["a_goals"] + res["h_goals"]
    pmf = np.asarray(sim["tot_pmf"], float)
    o55 = pmf_over_under(pmf, 5.5)[0]
    o65 = pmf_over_under(pmf, 6.5)[0]
    p7 = float(pmf[7:].sum())
    rec = {
        "h_win": sim["h_win"], "o55": o55, "o65": o65, "p7": p7, "shutout_p": sim["shutout"],
        "proj_tot": sim["proj_tot"], "alam": sim["alam"], "hlam": sim["hlam"],
        "win_ll": metrics.log_loss_vec([sim["h_win"]], [res["h_goals"] > res["a_goals"]])[0],
        "win_brier": metrics.brier_vec([sim["h_win"]], [res["h_goals"] > res["a_goals"]])[0],
        "o55_ll": metrics.log_loss_vec([o55], [tot > 5.5])[0],
        "o65_ll": metrics.log_loss_vec([o65], [tot > 6.5])[0],
        "total_ll": metrics.cat_log_loss_vec([_p_at(pmf, tot)])[0],
        "p1_ll": np.nan, "f10_ll": np.nan, "p1_tot_exp": sim["p1_tot"], "f10_p": sim["f10"],
    }
    if res.get("p1_a") is not None and not pd.isna(res.get("p1_a")):
        p1 = int(res["p1_a"] + res["p1_h"])
        p1pmf = np.asarray(sim["p1_pmf"], float)
        cats = [_p_at(p1pmf, 0), _p_at(p1pmf, 1), _p_at(p1pmf, 2), float(p1pmf[3:].sum())]
        rec["p1_ll"] = metrics.cat_log_loss_vec([cats[min(p1, 3)]])[0]
    fg = res.get("first_goal_p1_sec")
    if fg is not None and not pd.isna(fg):
        rec["f10_ll"] = metrics.log_loss_vec([sim["f10"]], [fg < 600])[0]
    return rec, pmf


def baseline_record(pool, res):
    """League-average baseline: home win rate and total-goals distribution of the last season of games."""
    tot = res["a_goals"] + res["h_goals"]
    hw = float((pool["h_goals"] > pool["a_goals"]).mean())
    counts = np.bincount(np.minimum(pool["a_goals"] + pool["h_goals"], TOT_K - 1), minlength=TOT_K) + 0.5
    pmf = counts / counts.sum()
    o55, o65 = float(pmf[6:].sum()), float(pmf[7:].sum())
    rec = {"h_win": hw, "o55": o55, "o65": o65, "p7": o65, "shutout_p": float(
        ((pool["a_goals"] == 0) | (pool["h_goals"] == 0)).mean()), "proj_tot": float((pool["a_goals"] + pool["h_goals"]).mean()),
        "win_ll": metrics.log_loss_vec([hw], [res["h_goals"] > res["a_goals"]])[0],
        "win_brier": metrics.brier_vec([hw], [res["h_goals"] > res["a_goals"]])[0],
        "o55_ll": metrics.log_loss_vec([o55], [tot > 5.5])[0], "o65_ll": metrics.log_loss_vec([o65], [tot > 6.5])[0],
        "total_ll": metrics.cat_log_loss_vec([pmf[min(tot, TOT_K - 1)]])[0], "p1_ll": np.nan, "f10_ll": np.nan}
    p1pool = pool.dropna(subset=["p1_a"]) if "p1_a" in pool else pool.iloc[0:0]
    if len(p1pool) and res.get("p1_a") is not None and not pd.isna(res.get("p1_a")):
        c = np.bincount(np.minimum((p1pool["p1_a"] + p1pool["p1_h"]).astype(int), 3), minlength=4) + 0.5
        rec["p1_ll"] = metrics.cat_log_loss_vec([(c / c.sum())[min(int(res["p1_a"] + res["p1_h"]), 3)]])[0]
    fpool = pool.dropna(subset=["first_goal_p1_sec"]) if "first_goal_p1_sec" in pool else pool.iloc[0:0]
    fg = res.get("first_goal_p1_sec")
    if len(fpool) and fg is not None and not pd.isna(fg):
        pf = float((fpool["first_goal_p1_sec"] < 600).mean())
        rec["f10_ll"] = metrics.log_loss_vec([pf], [fg < 600])[0]
    return rec


def run_season(season, results, prev_results, team_games, goalie_games, var_list, n_sims):
    """Every game of one season under every variant. Returns (long rows, live pmfs)."""
    rows, pmfs = [], {}
    starters = asof.actual_starters(goalie_games)
    skaters = asof.empty_skaters()
    played = results.groupby("date").apply(lambda d: set(d["away"]) | set(d["home"])).to_dict()
    history = pd.concat([x for x in (prev_results, results) if len(x)], ignore_index=True).sort_values(["date", "id"])
    for date_str, day in results.groupby("date", sort=True):
        teams = asof.teams_asof(team_games, date_str)
        if teams.empty:
            continue
        goalies = asof.goalies_asof(goalie_games, date_str)
        prev = (datetime.date.fromisoformat(date_str) - datetime.timedelta(days=1)).isoformat()
        b2b = played.get(prev, set())
        pool = history[history["date"] < date_str].tail(BASELINE_POOL)
        gp = teams[teams["situation"] == "all"].set_index("team")["games_played"]
        models = {lbl: Model(teams, goalies, skaters, b2b, params=prm, n_sims=n_sims) for lbl, _, _, prm, _ in var_list}
        for _, res in day.iterrows():
            m0 = models["live"]
            aa, ha = res["away"], res["home"]
            a_mp, h_mp = m0.to_mp(aa), m0.to_mp(ha)
            if a_mp not in gp.index or h_mp not in gp.index:
                continue
            base = {"season": season, "date": date_str, "id": int(res["id"]), "away": aa, "home": ha,
                    "gp_min": int(min(gp[a_mp], gp[h_mp])), "a_b2b": aa in b2b, "h_b2b": ha in b2b,
                    "a_goals": int(res["a_goals"]), "h_goals": int(res["h_goals"]), "last": res["last"],
                    "p1_tot": (res["p1_a"] + res["p1_h"]) if not pd.isna(res.get("p1_a")) else np.nan}
            if len(pool) >= 100:
                rows.append({**base, "variant": "baseline", "param": "baseline", "value": np.nan,
                             **baseline_record(pool, res)})
            for lbl, name, val, _, gmode in var_list:
                m = models[lbl]
                try:
                    if gmode == "actual":
                        ag = m.get_goalie(aa, goalie_id=starters.get((int(res["id"]), a_mp)))
                        hg = m.get_goalie(ha, goalie_id=starters.get((int(res["id"]), h_mp)))
                    else:
                        ag, hg = m.get_goalie(aa), m.get_goalie(ha)
                    sim = m.simulate_game(aa, ha, ag, hg, seed=int(res["id"]) % 100000)
                except (ValueError, IndexError, KeyError, ZeroDivisionError):
                    continue
                rec, pmf = game_record(sim, res)
                rec.update({"a_backup": ag["is_backup"], "h_backup": hg["is_backup"]})
                rows.append({**base, "variant": lbl, "param": name, "value": val, **rec})
                if lbl == "live":
                    pmfs[int(res["id"])] = np.pad(pmf[:TOT_K], (0, max(0, TOT_K - len(pmf))))
    return pd.DataFrame(rows), pmfs


def _season_job(args):
    season, var_list, n_sims, today = args
    results = data.nhl_results(season, today)
    prev = data.nhl_results(season - 1, today)
    tg = data.team_games([season])
    gg = data.goalie_games([season])  # prefetch() already refreshed the season in progress
    return run_season(season, results, prev, tg, gg, var_list, n_sims)


# ----------------------------------------------------------------------------
# Market check: nights with saved DK lines (screenshots in lines/, free-feed snapshots in lines/auto/)
# ----------------------------------------------------------------------------
def market_lines(paths):
    """{(date, away, home): {"ml": {team: odds}, "total": (line, over, under)}} from lines files."""
    out = {}
    for p in paths:
        d = os.path.basename(p)[:10]
        try:
            df = pd.read_csv(p, dtype=str).fillna("")
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
            continue
        for _, r in df.iterrows():
            away, _, home = r.get("game", "").replace(" ", "").upper().partition("@")
            mkt = r.get("market", "").strip().upper()
            key = (d, away, home)
            try:
                if mkt == "ML" and r.get("over_odds"):
                    out.setdefault(key, {}).setdefault("ml", {})[r["player"].strip().upper()] = int(float(r["over_odds"]))
                elif mkt in ("TOTAL", "TOT") and r.get("line") and r.get("over_odds") and r.get("under_odds"):
                    out.setdefault(key, {})["total"] = (float(r["line"]), int(float(r["over_odds"])), int(float(r["under_odds"])))
            except ValueError:
                continue
    return out


def market_check(live, pmfs, lines):
    """Model vs DK no-vig on the same games: win and total log loss. Tiny samples, so treat as a sanity check."""
    rows = []
    for _, g in live.iterrows():
        m = lines.get((g["date"], g["away"], g["home"]))
        if not m:
            continue
        hw = g["h_goals"] > g["a_goals"]
        tot = g["a_goals"] + g["h_goals"]
        ml = m.get("ml", {})
        if g["home"] in ml and g["away"] in ml:
            mk = implied_no_vig(ml[g["home"]], ml[g["away"]])
            rows.append({"market": "ML", "date": g["date"], "game": f"{g['away']}@{g['home']}",
                         "model_p": g["h_win"], "dk_p": mk,
                         "model_ll": metrics.log_loss_vec([g["h_win"]], [hw])[0],
                         "dk_ll": metrics.log_loss_vec([mk], [hw])[0]})
        if "total" in m and g["id"] in pmfs:
            line, ov, un = m["total"]
            po, pu, _ = pmf_over_under(pmfs[g["id"]], line)
            if tot == line:  # push
                continue
            mk = implied_no_vig(ov, un)
            mp = po / max(po + pu, 1e-9)
            rows.append({"market": f"Total {line:g}", "date": g["date"], "game": f"{g['away']}@{g['home']}",
                         "model_p": mp, "dk_p": mk, "model_ll": metrics.log_loss_vec([mp], [tot > line])[0],
                         "dk_ll": metrics.log_loss_vec([mk], [tot > line])[0]})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------
def split_of(season, tune, holdout):
    return "tuning" if season in tune else ("held-out" if season == holdout else "current")


def summarize(df, tune, holdout, min_gp):
    """Mean losses per split and variant (games where both teams have min_gp+ prior games)."""
    d = df[df["gp_min"] >= min_gp].copy()
    d["split"] = [split_of(s, tune, holdout) for s in d["season"]]
    agg = {k: (k, "mean") for k in LOSSES}
    agg.update(games=("id", "count"), shutout_pred=("shutout_p", "mean"), p7_pred=("p7", "mean"))
    return d, d.groupby(["split", "variant"]).agg(**agg).reset_index()


def diff_vs(d, split, variant, ref, loss):
    a = d[(d["split"] == split) & (d["variant"] == variant)].set_index("id")[loss]
    b = d[(d["split"] == split) & (d["variant"] == ref)].set_index("id")[loss]
    j = a.to_frame("a").join(b.to_frame("b"), how="inner").dropna()
    return metrics.paired_diff(j["a"], j["b"]) + (len(j),)


def param_table(d, grid):
    rows = []
    for name, vals in grid.items():
        loss = PRIMARY.get(name, "win_ll")
        ref = ACTUAL_REF if name in ACTUAL_GOALIE else "live"
        for v in vals:
            is_live = bool(np.isclose(v, LIVE[name]))
            lbl = ref if is_live else f"{name}={v:g}"
            row = {"param": name, "value": v, "live": is_live, "judged_on": loss}
            for split in ("tuning", "held-out"):
                sub = d[(d["split"] == split) & (d["variant"] == lbl)]
                row[f"{split}_{loss}"] = sub[loss].mean() if len(sub) else np.nan
                if not is_live:
                    mean, se, n = diff_vs(d, split, lbl, ref, loss)
                    row[f"{split}_diff"], row[f"{split}_se"] = mean, se
                else:
                    row[f"{split}_diff"], row[f"{split}_se"] = 0.0, 0.0
            rows.append(row)
    return pd.DataFrame(rows)


def vlabel(v):
    return "Poisson" if v >= 1e5 else f"{v:g}"


def recommend(pt):
    """Per parameter: suggest a change only if the best tuning value also beats live on the held-out season by 2+ SE."""
    out = []
    for name, sub in pt.groupby("param", sort=False):
        cand = sub[~sub["live"]].sort_values("tuning_diff")
        live_v = sub[sub["live"]]["value"].iloc[0]
        if cand.empty or cand.iloc[0]["tuning_diff"] >= 0:
            out.append({"param": name, "live": live_v, "best_on_tuning": live_v, "held_out_change": 0.0,
                        "held_out_se": 0.0, "recommendation": "Keep: nothing beat the live value on the tuning seasons."})
            continue
        b = cand.iloc[0]
        hd, hse = b["held-out_diff"], b["held-out_se"]
        if hd < 0 and hse > 0 and -hd > 2 * hse:
            rec = f"Evidence for {vlabel(b['value'])}: better on tuning and on the held-out season by more than 2 SE. Your call."
        elif hd < 0:
            rec = f"Weak: {vlabel(b['value'])} was a little better on both, but within noise. Keep."
        else:
            rec = f"Keep: {vlabel(b['value'])} won on tuning but not on the held-out season."
        out.append({"param": name, "live": live_v, "best_on_tuning": b["value"], "held_out_change": hd,
                    "held_out_se": hse, "recommendation": rec})
    return pd.DataFrame(out)


def fmt(x, nd=4):
    return "NA" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def md_table(df, cols, headers=None):
    headers = headers or cols
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(fmt(v) if isinstance(v, (float, np.floating)) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(df, pmfs, grid, tune, holdout, current, min_gp, n_sims, lines_paths, out=OUT):
    os.makedirs(out, exist_ok=True)
    d, summ = summarize(df, tune, holdout, min_gp)
    summ.to_csv(os.path.join(out, "summary_by_variant.csv"), index=False)
    pt = param_table(d, grid)
    pt.to_csv(os.path.join(out, "param_table.csv"), index=False)
    rec = recommend(pt)
    rec.to_csv(os.path.join(out, "recommendations.csv"), index=False)

    live = df[df["variant"] == "live"].copy()
    keep = ["season", "date", "id", "away", "home", "gp_min", "a_b2b", "h_b2b", "a_backup", "h_backup", "a_goals",
            "h_goals", "last", "p1_tot", "alam", "hlam", "proj_tot", "h_win", "o55", "o65", "p7", "shutout_p",
            "p1_tot_exp", "f10_p"]
    live[keep].to_csv(os.path.join(out, "predictions_live.csv"), index=False, float_format="%.4f")

    L = [f"# Backtest results\n\nRun {datetime.date.today().isoformat()} · {n_sims:,} sims per game · "
         f"tuning seasons {', '.join(f'{s}-{(s + 1) % 100:02d}' for s in tune)} · held-out {holdout}-{(holdout + 1) % 100:02d}"
         + (f" · current {current}-{(current + 1) % 100:02d} to date" if current else "") +
         f" · games where both teams had {min_gp}+ prior games.\n",
         "Every game is rebuilt from season-to-date MoneyPuck data as of that morning, with the #1 goalie by games "
         "played so far (what the site assumes before a starter is confirmed). Lower loss is better.\n"]

    # 1. Live model vs baseline
    L.append("## 1. Model vs a plain baseline\n\nBaseline = league average from the last season of games: home win "
             "rate for the winner, and the league total-goals distribution for totals. No team info at all.\n")
    rows = []
    for split in ("tuning", "held-out", "current"):
        for loss in ("win_ll", "win_brier", "o55_ll", "o65_ll", "total_ll", "p1_ll", "f10_ll"):
            mean, se, n = diff_vs(d, split, "live", "baseline", loss)
            if not n:
                continue
            m_live = d[(d["split"] == split) & (d["variant"] == "live")][loss].mean()
            m_base = d[(d["split"] == split) & (d["variant"] == "baseline")][loss].mean()
            rows.append({"split": split, "score": LOSS_LABEL[loss], "games": n, "model": m_live, "baseline": m_base,
                         "model_minus_baseline": mean, "se": se,
                         "verdict": "model better" if mean < -2 * se else ("baseline better" if mean > 2 * se else "no clear difference")})
    vs = pd.DataFrame(rows)
    vs.to_csv(os.path.join(out, "model_vs_baseline.csv"), index=False)
    if len(vs):
        L.append(md_table(vs, ["split", "score", "games", "model", "baseline", "model_minus_baseline", "se", "verdict"],
                          ["Split", "Score", "Games", "Model", "Baseline", "Diff", "SE", "Verdict"]) + "\n")

    # 2. Calibration
    hl = d[(d["variant"] == "live") & (d["split"] == "held-out")]
    allv = d[d["variant"] == "live"]
    for title, col, ycol, edges in (
            ("Home win chance", "h_win", hl["h_goals"] > hl["a_goals"], np.arange(0.20, 0.85, 0.05)),
            ("Over 5.5 chance", "o55", (hl["h_goals"] + hl["a_goals"]) > 5.5, np.arange(0.20, 0.85, 0.05))):
        cal = metrics.calibration_table(hl[col], ycol, edges)
        cal.to_csv(os.path.join(out, f"calibration_{col}.csv"), index=False)
        if len(cal):
            L.append(f"## 2. Calibration, held-out season: {title}\n\nForecast vs what happened, by bucket.\n")
            L.append(md_table(cal, ["bucket", "games", "forecast", "actual"], ["Bucket", "Games", "Forecast", "Actual"]) + "\n")

    # 3. Goal-total distribution
    if pmfs and len(allv):
        ids = [i for i in allv["id"] if i in pmfs]
        mean_pmf = np.mean([pmfs[i] for i in ids], axis=0)
        act = np.bincount(np.minimum((allv["a_goals"] + allv["h_goals"]).astype(int), TOT_K - 1), minlength=TOT_K) / len(allv)
        dist = pd.DataFrame({"total": [str(k) for k in range(10)] + ["10+"],
                             "forecast": list(mean_pmf[:10]) + [mean_pmf[10:].sum()],
                             "actual": list(act[:10]) + [act[10:].sum()]})
        dist.to_csv(os.path.join(out, "goal_distribution.csv"), index=False)
        tails = pd.DataFrame([
            {"event": "A team is shut out", "forecast": allv["shutout_p"].mean(),
             "actual": ((allv["a_goals"] == 0) | (allv["h_goals"] == 0)).mean()},
            {"event": "7+ total goals", "forecast": allv["p7"].mean(), "actual": ((allv["a_goals"] + allv["h_goals"]) >= 7).mean()},
            {"event": "Goes past regulation", "forecast": np.nan, "actual": (allv["last"] != "REG").mean()},
            {"event": "Average total goals", "forecast": allv["proj_tot"].mean(),
             "actual": (allv["a_goals"] + allv["h_goals"]).mean()},
            {"event": "Over 5.5 (forecast = average chance)", "forecast": allv["o55"].mean(),
             "actual": ((allv["a_goals"] + allv["h_goals"]) > 5.5).mean()},
            {"event": "Over 6.5 (forecast = average chance)", "forecast": allv["o65"].mean(),
             "actual": ((allv["a_goals"] + allv["h_goals"]) > 6.5).mean()},
            {"event": "Games the model leans under 5.5", "forecast": (allv["o55"] < 0.5).mean(), "actual": np.nan},
            {"event": "Games the model leans under 6.5", "forecast": (allv["o65"] < 0.5).mean(), "actual": np.nan},
        ])
        tails.to_csv(os.path.join(out, "goal_tails.csv"), index=False)
        L.append("## 3. Goal-total distribution, all seasons\n\nShare of games with each final total (shootout winner counts as a goal, like the books).\n")
        L.append(md_table(dist, ["total", "forecast", "actual"], ["Total", "Forecast", "Actual"]) + "\n")
        L.append(md_table(tails, ["event", "forecast", "actual"], ["Event", "Forecast", "Actual"]) + "\n")

    # 4. Early season
    rows = []
    early = df[(df["variant"].isin(["live", "baseline"]))]
    for lo, hi in ((1, 4), (5, 9), (10, 29), (30, 99)):
        sub = early[(early["gp_min"] >= lo) & (early["gp_min"] <= hi)]
        a = sub[sub["variant"] == "live"].set_index("id")["win_ll"]
        b = sub[sub["variant"] == "baseline"].set_index("id")["win_ll"]
        j = a.to_frame("m").join(b.to_frame("b"), how="inner")
        if len(j):
            mean, se = metrics.paired_diff(j["m"], j["b"])
            rows.append({"prior_games": f"{lo}-{hi}" if hi < 99 else f"{lo}+", "games": len(j), "model": j["m"].mean(),
                         "baseline": j["b"].mean(), "diff": mean, "se": se})
    if rows:
        L.append("## 4. By games played so far (win log loss, all seasons)\n\nHow the season-to-date stats do early on. Feeds Fix 6.\n")
        L.append(md_table(pd.DataFrame(rows), ["prior_games", "games", "model", "baseline", "diff", "se"],
                          ["Fewer team GP", "Games", "Model", "Baseline", "Diff", "SE"]) + "\n")

    # 5. Goalie assumption
    rows = []
    for split in ("tuning", "held-out"):
        for loss in ("win_ll", "total_ll"):
            mean, se, n = diff_vs(d, split, "goalie=actual starter", "live", loss)
            if n:
                rows.append({"split": split, "score": LOSS_LABEL[loss], "games": n, "actual_minus_assumed": mean, "se": se})
    if rows:
        L.append("## 5. Assumed #1 goalie vs the goalie who actually started\n\nNegative = knowing the real starter helps. "
                 "This is the cost of locking a pick on an assumed goalie (Fix 3).\n")
        L.append(md_table(pd.DataFrame(rows), ["split", "score", "games", "actual_minus_assumed", "se"],
                          ["Split", "Score", "Games", "Actual - assumed", "SE"]) + "\n")

    # 6. Parameters
    L.append("## 6. Parameters, one at a time\n\nEach row moves one parameter, everything else at the live value. "
             "Diff = change in the judged score vs live (negative = better), with its standard error. "
             "Rule set before running: suggest a change only if the best tuning value also beats live on the "
             "held-out season by more than 2 SE. The backup-goalie rows use the goalie who actually started "
             "(otherwise no backup ever plays). Moving one parameter at a time can't separate parameters that "
             "push the same way. **No live parameter was changed.**\n")
    show = pt.copy()
    show["value"] = [("Poisson" if v >= 1e5 else f"{v:g}") + (" (live)" if lv else "") for v, lv in zip(show["value"], show["live"])]
    show["judged_on"] = show["judged_on"].map(LOSS_LABEL)
    L.append(md_table(show, ["param", "value", "judged_on", "tuning_diff", "tuning_se", "held-out_diff", "held-out_se"],
                      ["Parameter", "Value", "Judged on", "Tuning diff", "SE", "Held-out diff", "SE"]) + "\n")
    L.append("### Recommendation per parameter\n")
    r2 = rec.copy()
    r2["best_on_tuning"] = [("Poisson" if v >= 1e5 else f"{v:g}") for v in r2["best_on_tuning"]]
    r2["live"] = [f"{v:g}" for v in r2["live"]]
    L.append(md_table(r2, ["param", "live", "best_on_tuning", "recommendation"],
                      ["Parameter", "Live", "Best on tuning", "Recommendation"]) + "\n")
    L.append("Not tested here: the 3-point edge floor and the 35% model floor. They choose bets against prices, and "
             "there are no free historical DraftKings prices, so they can only be judged on the nights the site has "
             "saved lines (see section 7 and the CLV log).\n")

    # 7. Market
    mk = market_check(live, pmfs, market_lines(lines_paths)) if lines_paths else pd.DataFrame()
    L.append("## 7. Against DraftKings (nights with saved lines)\n")
    if len(mk):
        mk.to_csv(os.path.join(out, "market_check.csv"), index=False)
        s = mk.groupby("market").agg(games=("game", "count"), model_ll=("model_ll", "mean"), dk_ll=("dk_ll", "mean")).reset_index()
        L.append(f"{len(mk)} priced outcomes. This is far too few to say anything; it's here so it grows over the season.\n")
        L.append(md_table(s, ["market", "games", "model_ll", "dk_ll"], ["Market", "Games", "Model log loss", "DK no-vig log loss"]) + "\n")
    else:
        L.append("No finished games with saved DraftKings lines matched yet. Free historical closing lines don't exist, "
                 "so the baseline above is league average, as agreed.\n")
    with open(os.path.join(out, "README.md"), "w") as f:
        f.write("\n".join(L))
    return d, pt, rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tune", type=int, nargs="+", default=[2022, 2023, 2024])
    ap.add_argument("--holdout", type=int, default=2025)
    ap.add_argument("--current", type=int, default=2026, help="current season to date (0 to skip)")
    ap.add_argument("--n-sims", type=int, default=10_000)
    ap.add_argument("--min-gp", type=int, default=5)
    ap.add_argument("--quick", action="store_true", help="2,000 sims and 3 parameters, for a fast check")
    ap.add_argument("--lines", nargs="*", default=None, help="lines CSVs for the market check (default: lines/*.csv, lines/auto/*.csv)")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    grid = QUICK_GRID if a.quick else GRID
    n_sims = 2000 if a.quick else a.n_sims
    var_list = variants(grid)
    seasons = a.tune + [a.holdout] + ([a.current] if a.current else [])
    today = datetime.date.today()
    data.prefetch(seasons, current=a.current or None, today=today)  # serial downloads; workers read the cache
    with ProcessPoolExecutor(max_workers=min(a.workers, len(seasons))) as ex:
        parts = list(ex.map(_season_job, [(s, var_list, n_sims, today) for s in seasons]))
    df = pd.concat([p[0] for p in parts], ignore_index=True)
    pmfs = {k: v for p in parts for k, v in p[1].items()}
    lines = a.lines if a.lines is not None else sorted(
        glob.glob(os.path.join(ROOT, "lines", "20*.csv")) + glob.glob(os.path.join(ROOT, "lines", "auto", "20*.csv")))
    write_report(df, pmfs, grid, a.tune, a.holdout, a.current, a.min_gp, n_sims, lines, out=a.out)
    print(open(os.path.join(a.out, "README.md")).read())


if __name__ == "__main__":
    main()
