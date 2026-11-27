"""Model core: team lambdas, the Gamma-Poisson game sim, skater projections and price helpers.

Pure Python (no Streamlit), so app.py and the backtest in backtests/ run the same code.
Every tunable number lives in DEFAULT_PARAMS. app.py uses the defaults; the backtest passes
variants to test one parameter at a time. Changing a default changes the live site.
"""
import math

import numpy as np
import pandas as pd

N_SIMS = 10_000
MP_MAP = {"LAK": "L.A", "NJD": "N.J", "SJS": "S.J", "TBL": "T.B"}

DEFAULT_PARAMS = {
    "gamma_shape": 18.0,      # Gamma-Poisson dispersion of regulation goals (higher = closer to Poisson)
    "b2b_off": 0.94,          # 6% offensive drag for a team on a back-to-back
    "b2b_def": 1.04,          # 4% defensive fatigue penalty (opponent of a B2B team gains)
    "home_ice": 0.035,        # home 5v5 rate x (1 + this), away x (1 - this): 1.035 / 0.965
    "other_a": 0.16,          # away goals not in 5v5 or 5v4 xG (4v4, 3v3, empty net, 5v3 ...)
    "other_h": 0.18,          # same for the home team
    "lam_floor": 0.8,         # minimum team lambda
    "ot_xg_weight": 0.60,     # OT/SO: away win chance = weight x away 5v5 xG share + base
    "ot_base": 0.20,
    "starter_prior": 15.0,    # #1 goalie GSAx/60 regresses to 0 over this many hours of ice time
    "backup_prior": 20.0,     # backups regress over this many hours ...
    "backup_mean": -0.18,     # ... toward this GSAx/60 ...
    "backup_drag": -0.10,     # ... then take this extra drag
    "p1_share": 0.30,         # share of regulation goals scored in the 1st period
    "f10_share": 0.48,        # share of 1st-period goals scored in its first 10 minutes
}
GAMMA_SHAPE = DEFAULT_PARAMS["gamma_shape"]
B2B_OFF = DEFAULT_PARAMS["b2b_off"]
B2B_DEF = DEFAULT_PARAMS["b2b_def"]
P1_SHARE = DEFAULT_PARAMS["p1_share"]
F10_SHARE = DEFAULT_PARAMS["f10_share"]


def poisson_cdf(k, lam):
    """P(X <= k) for X ~ Poisson(lam)."""
    if k < 0:
        return 0.0
    term = total = math.exp(-lam)
    for i in range(1, int(k) + 1):
        term *= lam / i
        total += term
    return min(total, 1.0)


def line_probs(lam, line):
    """Over / under / push probabilities for a Poisson mean against a prop line."""
    if float(line).is_integer():
        k = int(line)
        under = poisson_cdf(k - 1, lam)
        push = poisson_cdf(k, lam) - under
        return 1.0 - under - push, under, push
    under = poisson_cdf(math.floor(line), lam)
    return 1.0 - under, under, 0.0


def american_to_prob(odds):
    if odds is None or odds == 0 or -100 < odds < 100:
        return None
    return (-odds) / (-odds + 100.0) if odds < 0 else 100.0 / (odds + 100.0)


def prob_to_american(p):
    if p <= 0 or p >= 1:
        return "—"
    return f"{-100.0 * p / (1 - p):+.0f}" if p >= 0.5 else f"+{100.0 * (1 - p) / p:.0f}"


def implied_no_vig(mine, other):
    """Implied % for my side; removes the vig when both sides are known."""
    p_m = american_to_prob(mine) if mine is not None else None
    if p_m is None:
        return None
    p_o = american_to_prob(other) if other is not None else None
    return p_m / (p_m + p_o) if p_o else p_m


def implied_3way(prices, key):
    """Implied % for one outcome of a 3-way market; vig removed when all three prices are known."""
    probs = {k: american_to_prob(v) for k, v in (prices or {}).items() if v is not None}
    probs = {k: v for k, v in probs.items() if v}
    if key not in probs:
        return None
    return probs[key] / sum(probs.values()) if len(probs) >= 3 else probs[key]


def pmf_over_under(pmf, line):
    """Over / under / push probabilities from a goals PMF."""
    ks = np.arange(len(pmf))
    over = float(pmf[ks > line].sum())
    push = float(pmf[ks == line].sum()) if float(line).is_integer() else 0.0
    return over, max(1.0 - over - push, 0.0), push


class Model:
    def __init__(self, teams_df, goalies_df, skaters_df, b2b_teams, params=None, n_sims=N_SIMS):
        self.p = {**DEFAULT_PARAMS, **(params or {})}
        self.n_sims = n_sims
        self.teams_df = teams_df
        self.goalies_df = goalies_df
        self.skaters_df = skaters_df
        self.b2b = b2b_teams
        lg_5v5 = teams_df[teams_df["situation"] == "5on5"]
        lg_hrs = lg_5v5["iceTime"].sum() / 3600.0
        self.LG_XG5 = lg_5v5["xGoalsFor"].sum() / lg_hrs
        self.LG_CA60 = lg_5v5["shotAttemptsAgainst"].sum() / lg_hrs
        lg_pp = teams_df[teams_df["situation"] == "5on4"]
        self.LG_XGPP = lg_pp["xGoalsFor"].sum() / (lg_pp["iceTime"].sum() / 3600.0)

    def to_mp(self, abbr):
        return abbr if abbr in self.teams_df["team"].values else MP_MAP.get(abbr, abbr)

    # -- Teams --------------------------------------------------------------
    def get_team(self, abbr):
        t = self.teams_df[self.teams_df["team"] == self.to_mp(abbr)]
        if t.empty:
            raise ValueError(f"No MoneyPuck team data for {abbr}")
        s5, spp, spk, sall = [t[t["situation"] == s].iloc[0] for s in ("5on5", "5on4", "4on5", "all")]
        toi5, toipp, toipk, toiall = [max(x["iceTime"] / 3600.0, 0.05) for x in (s5, spp, spk, sall)]
        xgf_c = "scoreVenueAdjustedxGoalsFor" if "scoreVenueAdjustedxGoalsFor" in s5.index else "xGoalsFor"
        xga_c = "scoreVenueAdjustedxGoalsAgainst" if "scoreVenueAdjustedxGoalsAgainst" in s5.index else "xGoalsAgainst"
        hdcf, hdca = s5["highDangerShotsFor"] / toi5, s5["highDangerShotsAgainst"] / toi5
        return {
            "xgf60": s5[xgf_c] / toi5, "xga60": s5[xga_c] / toi5,
            "cf60": s5["shotAttemptsFor"] / toi5, "ca60": s5["shotAttemptsAgainst"] / toi5,
            "hdcf_pct": (hdcf / max(hdcf + hdca, 0.1)) * 100,
            "pp_xgf60": spp["xGoalsFor"] / toipp, "pk_xga60": spk["xGoalsAgainst"] / toipk,
            "drawn60": sall["penaltiesAgainst"] / toiall, "taken60": sall["penaltiesFor"] / toiall,
        }

    # -- Goalies ------------------------------------------------------------
    def team_goalies(self, abbr):
        g = self.goalies_df
        return g[(g["situation"] == "all") & (g["team"] == self.to_mp(abbr))].sort_values("games_played", ascending=False)

    def get_goalie(self, abbr, goalie_name=None, goalie_id=None):
        """#1 starter (most GP) regresses to 0.00 GSAx/60 over 15 hours of ice time;
        backups regress to -0.18 over 20 hours, then a -0.10 drag (see DEFAULT_PARAMS).
        goalie_name (manual override) wins over goalie_id (NHL-confirmed starter)."""
        P = self.p
        team_g = self.team_goalies(abbr)
        if team_g.empty:
            return {"name": "Unknown", "label": "Unknown", "adj60": 0.0, "is_backup": False, "source": "projected"}
        starter = team_g.iloc[0]["name"]
        source = "projected"
        target = starter
        if goalie_name in team_g["name"].values:
            target, source = goalie_name, "override"
        elif goalie_id is not None and "playerId" in team_g.columns and goalie_id in team_g["playerId"].values:
            target, source = team_g[team_g["playerId"] == goalie_id].iloc[0]["name"], "confirmed"
        g = team_g[team_g["name"] == target].iloc[0]
        gp = int(g["games_played"])
        toi = max(g["icetime"] / 3600.0, 0.1)
        raw60 = (g["xGoals"] - g["goals"]) / toi
        is_backup = target != starter
        if is_backup:
            adj60 = ((raw60 * toi) + (P["backup_mean"] * P["backup_prior"])) / (toi + P["backup_prior"]) + P["backup_drag"]
        else:
            adj60 = (raw60 * toi) / (toi + P["starter_prior"])
        return {"name": target, "label": f"{target} ({gp} GP)", "adj60": adj60, "is_backup": is_backup, "source": source}

    # -- Game simulation ----------------------------------------------------
    def simulate_game(self, aa, ha, ag, hg, seed=42):
        P = self.p
        B2B_OFF, B2B_DEF, shape = P["b2b_off"], P["b2b_def"], P["gamma_shape"]
        a, h = self.get_team(aa), self.get_team(ha)
        a_b2b, h_b2b = aa in self.b2b, ha in self.b2b

        app_m = float(np.clip(5.2 + (a["drawn60"] - 3.3) * 0.65 + (h["taken60"] - 3.3) * 0.65, 2.5, 9.5))
        hpp_m = float(np.clip(5.2 + (h["drawn60"] - 3.3) * 0.65 + (a["taken60"] - 3.3) * 0.65, 2.5, 9.5))
        ev_m = max(60.0 - app_m - hpp_m, 42.0)

        a5_rate = a["xgf60"] * (h["xga60"] / self.LG_XG5) * (1 - P["home_ice"]) * (B2B_OFF if a_b2b else 1.0) * (B2B_DEF if h_b2b else 1.0)
        h5_rate = h["xgf60"] * (a["xga60"] / self.LG_XG5) * (1 + P["home_ice"]) * (B2B_OFF if h_b2b else 1.0) * (B2B_DEF if a_b2b else 1.0)
        a_5v5, h_5v5 = a5_rate * (ev_m / 60.0), h5_rate * (ev_m / 60.0)
        a_pp = a["pp_xgf60"] * (h["pk_xga60"] / self.LG_XGPP) * (app_m / 60.0)
        h_pp = h["pp_xgf60"] * (a["pk_xga60"] / self.LG_XGPP) * (hpp_m / 60.0)

        alam = max(a_5v5 + a_pp - hg["adj60"] + P["other_a"], P["lam_floor"])
        hlam = max(h_5v5 + h_pp - ag["adj60"] + P["other_h"], P["lam_floor"])

        rng = np.random.default_rng(seed)
        ar = rng.poisson(rng.gamma(shape, alam / shape, self.n_sims))
        hr = rng.poisson(rng.gamma(shape, hlam / shape, self.n_sims))
        ties = ar == hr
        af, hf = ar.copy(), hr.copy()
        if ties.sum() > 0:
            away_ot = rng.random(ties.sum()) < (P["ot_xg_weight"] * (a5_rate / (a5_rate + h5_rate)) + P["ot_base"])
            idx = np.where(ties)[0]
            af[idx[away_ot]] += 1
            hf[idx[~away_ot]] += 1
        tot = af + hf
        # 1st period: thin each team's regulation goals to the 1st period, then to its first 10 minutes.
        a1, h1 = rng.binomial(ar, P["p1_share"]), rng.binomial(hr, P["p1_share"])
        t1 = a1 + h1
        f10 = rng.binomial(t1, P["f10_share"])

        return {
            "a": a, "h": h, "a_b2b": a_b2b, "h_b2b": h_b2b,
            "app_m": app_m, "hpp_m": hpp_m, "ev_m": ev_m,
            "a_5v5": a_5v5, "h_5v5": h_5v5, "a_pp": a_pp, "h_pp": h_pp,
            "alam": alam, "hlam": hlam,
            "proj_a": af.mean(), "proj_h": hf.mean(), "proj_tot": tot.mean(),
            "a_win": (af > hf).mean(), "h_win": (hf > af).mean(),
            "o55": (tot > 5.5).mean(),
            "shutout": ((af == 0) | (hf == 0)).mean(),
            "tot_pmf": np.bincount(tot) / len(tot),
            "p1_a": a1.mean(), "p1_h": h1.mean(), "p1_tot": t1.mean(),
            "p1_pmf": np.bincount(t1, minlength=3) / len(t1),
            "p1_a_win": (a1 > h1).mean(), "p1_h_win": (h1 > a1).mean(), "p1_tie": (a1 == h1).mean(),
            "f10": (f10 > 0).mean(),
        }

    # -- Player props -------------------------------------------------------
    def project_skaters(self, team_abbr, opp_abbr, is_home, opp_gsax60, apply_b2b=True, player_ids=None):
        opp = self.get_team(opp_abbr)
        is_b2b = apply_b2b and team_abbr in self.b2b
        opp_b2b = apply_b2b and opp_abbr in self.b2b
        P = self.p
        B2B_OFF, B2B_DEF = P["b2b_off"], P["b2b_def"]
        venue = (1 + P["home_ice"]) if is_home else (1 - P["home_ice"])

        pace_mult = (opp["ca60"] / self.LG_CA60) * venue * (0.95 if is_b2b else 1.0)
        xg_mult = (opp["xga60"] / self.LG_XG5) * venue * (B2B_OFF if is_b2b else 1.0) * (B2B_DEF if opp_b2b else 1.0)
        pp_mult = opp["pk_xga60"] / self.LG_XGPP
        goalie_mult = max(1.0 - (opp_gsax60 / 3.0), 0.75)

        sk = self.skaters_df
        pool = sk[(sk["team"] == self.to_mp(team_abbr)) & (sk["games_played"] >= 1)]
        if player_ids:
            in_lineup = pool[pool["playerId"].isin(player_ids)]
            if not in_lineup.empty:
                pool = in_lineup
        s_all = pool[pool["situation"] == "all"]
        s_5v5 = pool[pool["situation"] == "5on5"].set_index("playerId")
        s_pp = pool[pool["situation"] == "5on4"].set_index("playerId")

        rows = []
        for _, p in s_all.iterrows():
            pid, gp = p["playerId"], max(int(p["games_played"]), 1)
            toi_pg = (p["icetime"] / 60.0) / gp
            p5 = s_5v5.loc[pid] if pid in s_5v5.index else None
            ppp = s_pp.loc[pid] if pid in s_pp.index else None
            ev_pg = ((p5["icetime"] / 60.0) / gp) if p5 is not None else toi_pg * 0.85
            pp_pg = ((ppp["icetime"] / 60.0) / gp) if ppp is not None else 0.0
            all_hrs = max(p["icetime"] / 3600.0, 0.1)

            sog = (p["I_F_shotsOnGoal"] / all_hrs) * (toi_pg / 60.0) * pace_mult
            ixg_5v5 = (p5["I_F_xGoals"] / max(p5["icetime"] / 3600.0, 0.1)) if p5 is not None else 0.0
            ixg_pp = (ppp["I_F_xGoals"] / max(ppp["icetime"] / 3600.0, 0.05)) if (ppp is not None and ppp["icetime"] > 60) else 0.0
            xg = ((ixg_5v5 * (ev_pg / 60.0) * xg_mult) + (ixg_pp * (pp_pg / 60.0) * pp_mult)) * goalie_mult
            ast = ((p["I_F_primaryAssists"] + p["I_F_secondaryAssists"]) / all_hrs) * (toi_pg / 60.0) * xg_mult
            pts = xg + ast
            rows.append({
                "playerId": pid, "name": p["name"], "pos": p["position"], "gp": gp,
                "toi": toi_pg, "pp": pp_pg, "sog": sog, "xg": xg,
                "atg": (1 - np.exp(-xg)) * 100, "pts": pts, "pt_p": (1 - np.exp(-pts)) * 100,
            })
        cols = ["playerId", "name", "pos", "gp", "toi", "pp", "sog", "xg", "atg", "pts", "pt_p"]
        return pd.DataFrame(rows, columns=cols).sort_values("pts", ascending=False)
