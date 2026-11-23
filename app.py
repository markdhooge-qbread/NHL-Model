"""NHL Game Flow & Player Prop Model (Streamlit).

Streamlit port of notebooks/player_model.ipynb:
  1. Slate Sim   - 10,000-iteration Gamma-Poisson Monte Carlo per game
  2. Player Props - SOG, anytime goal %, and 1+ point % for top skaters
  3. Live Tracker - every game's status, period/clock, and model vs. live/final box score
  4. Record       - season-to-date winner hit rate and goal error on final games
"""

import datetime
import io
import math
import os
import re
import unicodedata
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
SEASON = 2026
HEADERS = {"User-Agent": "Mozilla/5.0"}
EST = ZoneInfo("America/New_York")
N_SIMS = 10_000
GAMMA_SHAPE = 18.0
B2B_OFF = 0.94   # 6% offensive drag for a team on a back-to-back
B2B_DEF = 1.04   # 4% defensive fatigue penalty (opponent of a B2B team gains)
MP_BASE = "https://moneypuck.com/moneypuck/playerData/seasonSummary"
NHL_API = "https://api-web.nhle.com/v1"
MP_MAP = {"LAK": "L.A", "NJD": "N.J", "SJS": "S.J", "TBL": "T.B"}
SEASON_START = datetime.date(SEASON, 10, 1)
DK_NHL_URL = "https://sportsbook.draftkings.com/leagues/hockey/nhl"
LINES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lines")
ODDS_API = "https://api.the-odds-api.com/v4/sports/icehockey_nhl"
TEAM_ABBR = {
    "Anaheim Ducks": "ANA", "Boston Bruins": "BOS", "Buffalo Sabres": "BUF", "Calgary Flames": "CGY",
    "Carolina Hurricanes": "CAR", "Chicago Blackhawks": "CHI", "Colorado Avalanche": "COL",
    "Columbus Blue Jackets": "CBJ", "Dallas Stars": "DAL", "Detroit Red Wings": "DET", "Edmonton Oilers": "EDM",
    "Florida Panthers": "FLA", "Los Angeles Kings": "LAK", "Minnesota Wild": "MIN", "Montreal Canadiens": "MTL",
    "Montréal Canadiens": "MTL", "Nashville Predators": "NSH", "New Jersey Devils": "NJD",
    "New York Islanders": "NYI", "New York Rangers": "NYR", "Ottawa Senators": "OTT",
    "Philadelphia Flyers": "PHI", "Pittsburgh Penguins": "PIT", "San Jose Sharks": "SJS", "Seattle Kraken": "SEA",
    "St Louis Blues": "STL", "St. Louis Blues": "STL", "Tampa Bay Lightning": "TBL", "Toronto Maple Leafs": "TOR",
    "Utah Mammoth": "UTA", "Utah Hockey Club": "UTA", "Vancouver Canucks": "VAN", "Vegas Golden Knights": "VGK",
    "Washington Capitals": "WSH", "Winnipeg Jets": "WPG",
}

st.set_page_config(page_title="NHL Game Flow Model", page_icon="🏒", layout="centered")


def st_html(content):
    """Render HTML via st.markdown. Lines are left-stripped and blank lines dropped so Markdown
    never treats indented HTML as a code block (which shows raw tags)."""
    lines = [ln.strip() for ln in str(content).splitlines()]
    st.markdown("\n".join(ln for ln in lines if ln), unsafe_allow_html=True)

st_html(
    """
    <style>
      .block-container {padding-top: 1.2rem; padding-left: 0.8rem; padding-right: 0.8rem;}
      .card {background:#0f172a;border:1px solid #334155;border-radius:10px;padding:10px;
             margin-bottom:12px;font-family:-apple-system,sans-serif;color:#f8fafc;}
      .card-head {display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;
                  gap:4px;margin-bottom:6px;}
      .card-title {font-size:15px;font-weight:800;}
      .card-sub {font-size:11px;color:#94a3b8;font-weight:400;}
      .summary {display:flex;flex-wrap:wrap;gap:6px 14px;justify-content:space-between;
                background:#1e293b;padding:8px 10px;border-radius:6px;margin-bottom:8px;font-size:13px;}
      .tbl-wrap {overflow-x:auto;}
      .tbl {width:100%;border-collapse:collapse;font-size:12px;color:#cbd5e1;}
      .tbl th {color:#94a3b8;background:#1e293b;padding:4px;font-weight:600;}
      .tbl td {padding:4px;text-align:center;border-bottom:1px solid #334155;}
      .tbl td:first-child, .tbl th:first-child {text-align:left;}
      .badge {padding:1px 5px;border-radius:4px;font-size:10px;font-weight:700;white-space:nowrap;}
      .b2b {background:#b45309;color:#fde68a;}
      .conf {background:#14532d;color:#86efac;}
      .backup {background:#991b1b;color:#fecaca;}
      .track {background:#7f1d1d;color:#fca5a5;}
      .lowev {background:#1e3a8a;color:#93c5fd;}
      .bal {background:#334155;color:#cbd5e1;}
      .section {font-size:13px;font-weight:800;color:#38bdf8;margin:4px 0;}
      .meta {color:#94a3b8;font-size:12px;margin-bottom:8px;}
      .dot {display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px;vertical-align:middle;}
      .dot.live {background:#22c55e;box-shadow:0 0 0 0 rgba(34,197,94,.7);animation:pulse 1.6s infinite;}
      .dot.final {background:#ef4444;}
      .dot.sched {background:#64748b;}
      @keyframes pulse {0% {box-shadow:0 0 0 0 rgba(34,197,94,.7);} 70% {box-shadow:0 0 0 7px rgba(34,197,94,0);} 100% {box-shadow:0 0 0 0 rgba(34,197,94,0);}}
    </style>
    """)

B2B_BADGE = " <span class='badge b2b'>B2B</span>"
BACKUP_BADGE = " <span class='badge backup'>BACKUP</span>"
SMALL_SAMPLE_BADGE = " <span class='badge bal'>&lt;5 GP</span>"
CONFIRMED_BADGE = " <span class='badge conf'>✓ CONFIRMED</span>"


def goalie_badges(gl):
    return (CONFIRMED_BADGE if gl.get("source") == "confirmed" else "") + (BACKUP_BADGE if gl["is_backup"] else "")


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
@st.cache_data(ttl=1800, show_spinner="Loading MoneyPuck data...")
def load_moneypuck(season):
    out = {}
    for name in ("teams", "goalies", "skaters"):
        url = f"{MP_BASE}/{season}/regular/{name}.csv"
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        out[name] = pd.read_csv(io.StringIO(resp.text))
    return out["teams"], out["goalies"], out["skaters"]


@st.cache_data(ttl=300, show_spinner=False)
def load_schedule(date_str):
    resp = requests.get(f"{NHL_API}/schedule/{date_str}", timeout=15)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=30, show_spinner=False)
def load_boxscore(game_id):
    resp = requests.get(f"{NHL_API}/gamecenter/{game_id}/boxscore", timeout=15)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=20, show_spinner=False)
def load_scores(date_str):
    """Live scoreboard for a date: state, score, SOG, period and clock for every game, keyed by game id."""
    resp = requests.get(f"{NHL_API}/score/{date_str}", timeout=15)
    resp.raise_for_status()
    return {g["id"]: g for g in resp.json().get("games", [])}


@st.cache_data(ttl=3600, show_spinner=False)
def load_roster(abbr):
    """Current NHL roster skater ids for a team, or None if unavailable."""
    try:
        resp = requests.get(f"{NHL_API}/roster/{abbr}/current", timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception:  # noqa: BLE001
        return None
    ids = {p.get("id") for grp in ("forwards", "defensemen") for p in data.get(grp, [])}
    ids.discard(None)
    return ids or None


def dressed_ids(box, side):
    """Skater ids who dressed for one side ('awayTeam'/'homeTeam'), or None before the game."""
    team = box.get("playerByGameStats", {}).get(side, {})
    ids = {p.get("playerId") for grp in ("forwards", "defense") for p in team.get(grp, [])}
    ids.discard(None)
    return ids or None


def starter_goalie_id(box, side):
    """NHL-confirmed starting goalie id for 'awayTeam'/'homeTeam', or None."""
    if not box:
        return None
    for p in box.get("playerByGameStats", {}).get(side, {}).get("goalies", []):
        if p.get("starter"):
            return p.get("playerId")
    return None


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


@st.cache_data(ttl=3600, show_spinner="Loading season results...")
def load_season_games(start_str, end_str):
    """Completed regular-season games between two dates, plus teams that played each day."""
    start = datetime.date.fromisoformat(start_str)
    end = datetime.date.fromisoformat(end_str)
    finals, played = {}, {}
    d = start
    while d <= end:
        try:
            resp = requests.get(f"{NHL_API}/schedule/{d.isoformat()}", timeout=15)
            resp.raise_for_status()
            week = resp.json().get("gameWeek", [])
        except Exception:  # noqa: BLE001
            week = []
        for day in week:
            ds = day.get("date")
            for g in day.get("games", []):
                played.setdefault(ds, set()).update({g["awayTeam"]["abbrev"], g["homeTeam"]["abbrev"]})
                if g.get("gameType", 2) != 2 or g.get("gameState") not in ("FINAL", "OFF"):
                    continue
                if not (start_str <= ds <= end_str):
                    continue
                finals[g["id"]] = {
                    "id": g["id"], "date": ds,
                    "away": g["awayTeam"]["abbrev"], "home": g["homeTeam"]["abbrev"],
                    "a_goals": g["awayTeam"].get("score", 0) or 0, "h_goals": g["homeTeam"].get("score", 0) or 0,
                    "last": (g.get("gameOutcome") or {}).get("lastPeriodType", "REG"),
                }
        d += datetime.timedelta(days=7)
    return sorted(finals.values(), key=lambda x: (x["date"], x["id"])), played


@st.cache_data(ttl=3600, show_spinner="Grading the model on this season's games...")
def season_record(season, start_str, end_str):
    teams_df, goalies_df, skaters_df = load_moneypuck(season)
    finals, played = load_season_games(start_str, end_str)
    rows = []
    for g in finals:
        prev = (datetime.date.fromisoformat(g["date"]) - datetime.timedelta(days=1)).isoformat()
        m = Model(teams_df, goalies_df, skaters_df, played.get(prev, set()))
        try:
            sim = m.simulate_game(g["away"], g["home"], m.get_goalie(g["away"]), m.get_goalie(g["home"]), seed=g["id"] % 100000)
        except Exception:  # noqa: BLE001
            continue
        pick = g["away"] if sim["a_win"] > sim["h_win"] else g["home"]
        winner = g["away"] if g["a_goals"] > g["h_goals"] else g["home"]
        act_tot = g["a_goals"] + g["h_goals"]
        rows.append({
            "date": g["date"], "away": g["away"], "home": g["home"],
            "proj_a": sim["proj_a"], "proj_h": sim["proj_h"], "proj_tot": sim["proj_tot"],
            "a_goals": g["a_goals"], "h_goals": g["h_goals"], "act_tot": act_tot, "last": g["last"],
            "pick": pick, "pick_p": max(sim["a_win"], sim["h_win"]), "hit": pick == winner,
            "o55_p": sim["o55"], "o55_hit": (sim["o55"] >= 0.5) == (act_tot > 5.5),
            "tot_err": act_tot - sim["proj_tot"],
        })
    return pd.DataFrame(rows)


def games_on(sched, date_str):
    games = []
    for d in sched.get("gameWeek", []):
        if d.get("date") != date_str:
            continue
        for g in d.get("games", []):
            start = datetime.datetime.fromisoformat(g["startTimeUTC"].replace("Z", "+00:00")).astimezone(EST)
            games.append({
                "id": g["id"],
                "away": g["awayTeam"]["abbrev"],
                "home": g["homeTeam"]["abbrev"],
                "state": g.get("gameState", "FUT"),
                "time": start.strftime("%-I:%M %p ET"),
                "start": start,
            })
    return games


def teams_played_on(sched, date_str):
    return {
        g[k]["abbrev"]
        for d in sched.get("gameWeek", []) if d.get("date") == date_str
        for g in d.get("games", [])
        for k in ("awayTeam", "homeTeam")
    }


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class Model:
    def __init__(self, teams_df, goalies_df, skaters_df, b2b_teams):
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
        """#1 starter (most GP) regresses to 0.00 GSAx/60 over 15 games;
        backups regress to -0.18 over 20 games, then a -0.10 drag.
        goalie_name (manual override) wins over goalie_id (NHL-confirmed starter)."""
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
            adj60 = ((raw60 * toi) + (-0.18 * 20.0)) / (toi + 20.0) - 0.10
        else:
            adj60 = (raw60 * toi) / (toi + 15.0)
        return {"name": target, "label": f"{target} ({gp} GP)", "adj60": adj60, "is_backup": is_backup, "source": source}

    # -- Game simulation ----------------------------------------------------
    def simulate_game(self, aa, ha, ag, hg, seed=42):
        a, h = self.get_team(aa), self.get_team(ha)
        a_b2b, h_b2b = aa in self.b2b, ha in self.b2b

        app_m = float(np.clip(5.2 + (a["drawn60"] - 3.3) * 0.65 + (h["taken60"] - 3.3) * 0.65, 2.5, 9.5))
        hpp_m = float(np.clip(5.2 + (h["drawn60"] - 3.3) * 0.65 + (a["taken60"] - 3.3) * 0.65, 2.5, 9.5))
        ev_m = max(60.0 - app_m - hpp_m, 42.0)

        a5_rate = a["xgf60"] * (h["xga60"] / self.LG_XG5) * 0.965 * (B2B_OFF if a_b2b else 1.0) * (B2B_DEF if h_b2b else 1.0)
        h5_rate = h["xgf60"] * (a["xga60"] / self.LG_XG5) * 1.035 * (B2B_OFF if h_b2b else 1.0) * (B2B_DEF if a_b2b else 1.0)
        a_5v5, h_5v5 = a5_rate * (ev_m / 60.0), h5_rate * (ev_m / 60.0)
        a_pp = a["pp_xgf60"] * (h["pk_xga60"] / self.LG_XGPP) * (app_m / 60.0)
        h_pp = h["pp_xgf60"] * (a["pk_xga60"] / self.LG_XGPP) * (hpp_m / 60.0)

        alam = max(a_5v5 + a_pp - hg["adj60"] + 0.16, 0.8)
        hlam = max(h_5v5 + h_pp - ag["adj60"] + 0.18, 0.8)

        rng = np.random.default_rng(seed)
        ar = rng.poisson(rng.gamma(GAMMA_SHAPE, alam / GAMMA_SHAPE, N_SIMS))
        hr = rng.poisson(rng.gamma(GAMMA_SHAPE, hlam / GAMMA_SHAPE, N_SIMS))
        ties = ar == hr
        af, hf = ar.copy(), hr.copy()
        if ties.sum() > 0:
            away_ot = rng.random(ties.sum()) < (0.60 * (a5_rate / (a5_rate + h5_rate)) + 0.20)
            idx = np.where(ties)[0]
            af[idx[away_ot]] += 1
            hf[idx[~away_ot]] += 1
        tot = af + hf

        return {
            "a": a, "h": h, "a_b2b": a_b2b, "h_b2b": h_b2b,
            "app_m": app_m, "hpp_m": hpp_m, "ev_m": ev_m,
            "a_5v5": a_5v5, "h_5v5": h_5v5, "a_pp": a_pp, "h_pp": h_pp,
            "proj_a": af.mean(), "proj_h": hf.mean(), "proj_tot": tot.mean(),
            "a_win": (af > hf).mean(), "h_win": (hf > af).mean(),
            "o55": (tot > 5.5).mean(),
            "tot_pmf": np.bincount(tot) / len(tot),
        }

    # -- Player props -------------------------------------------------------
    def project_skaters(self, team_abbr, opp_abbr, is_home, opp_gsax60, apply_b2b=True, player_ids=None):
        opp = self.get_team(opp_abbr)
        is_b2b = apply_b2b and team_abbr in self.b2b
        opp_b2b = apply_b2b and opp_abbr in self.b2b
        venue = 1.035 if is_home else 0.965

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


# ----------------------------------------------------------------------------
# Rendering helpers
# ----------------------------------------------------------------------------
def pace_tag(comb_cf, total):
    if comb_cf >= 122 and total >= 6.2:
        return "<span class='badge track'>🔥 TRACK MEET</span>"
    if comb_cf <= 112 and total <= 5.6:
        return "<span class='badge lowev'>🧱 LOW-EVENT</span>"
    return "<span class='badge bal'>⚖️ BALANCED</span>"


def dk_game_rows(aa, ha, sim, dk):
    """DK moneyline and total rows for a slate card (NA when no line)."""
    def edge_txt(p, imp):
        if imp is None:
            return ""
        e = (p - imp) * 100
        color = "#86efac" if e >= 3 else ("#fca5a5" if e <= -3 else "#cbd5e1")
        return f" <b style='color:{color}'>{e:+.0f}</b>"

    ml = (dk or {}).get("ml", {}).get((aa, ha), {})
    a_o, h_o = ml.get(aa), ml.get(ha)
    a_cell = f"{fmt_odds(a_o)}{edge_txt(sim['a_win'], implied_no_vig(a_o, h_o))}" if a_o is not None else "NA"
    h_cell = f"{fmt_odds(h_o)}{edge_txt(sim['h_win'], implied_no_vig(h_o, a_o))}" if h_o is not None else "NA"
    tot = (dk or {}).get("total", {}).get((aa, ha))
    if tot and tot.get("line") is not None:
        p_o, p_u, _ = pmf_over_under(sim["tot_pmf"], tot["line"])
        o_cell = f"O{tot['line']:g} {fmt_odds(tot.get('over'))}{edge_txt(p_o, implied_no_vig(tot.get('over'), tot.get('under')))}"
        u_cell = f"U{tot['line']:g} {fmt_odds(tot.get('under'))}{edge_txt(p_u, implied_no_vig(tot.get('under'), tot.get('over')))}"
        tot_row = f"<tr><td>DK total (edge)</td><td>{o_cell}</td><td>{u_cell}</td></tr>"
    else:
        tot_row = "<tr><td>DK total (edge)</td><td colspan='2'>NA</td></tr>"
    return f"<tr><td>DK ML (edge)</td><td>{a_cell}</td><td>{h_cell}</td></tr>{tot_row}"


def slate_card(g, sim, ag, hg, dk=None):
    aa, ha, a, h = g["away"], g["home"], sim["a"], sim["h"]
    tag = pace_tag(a["cf60"] + h["cf60"], sim["proj_tot"])
    a_b = B2B_BADGE if sim["a_b2b"] else ""
    h_b = B2B_BADGE if sim["h_b2b"] else ""
    ag_b = goalie_badges(ag)
    hg_b = goalie_badges(hg)
    return f"""
    <div class='card'>
      <div class='card-head'>
        <div class='card-title'>{aa}{a_b} @ {ha}{h_b} <span class='card-sub'>({g['time']})</span></div>
        <div>{tag}</div>
      </div>
      <div class='summary'>
        <div><b>Proj:</b> {aa} {sim['proj_a']:.2f} — {ha} {sim['proj_h']:.2f}</div>
        <div><b>Win:</b> {aa} {sim['a_win']*100:.0f}% | {ha} {sim['h_win']*100:.0f}%</div>
        <div><b>Tot:</b> {sim['proj_tot']:.2f} (O5.5: {sim['o55']*100:.0f}%)</div>
      </div>
      <div class='tbl-wrap'><table class='tbl'>
        <tr><th>Per-Game</th><th>{aa}</th><th>{ha}</th></tr>
        <tr><td>5v5 xG ({sim['ev_m']:.0f}m)</td><td><b>{sim['a_5v5']:.2f}</b></td><td><b>{sim['h_5v5']:.2f}</b></td></tr>
        <tr><td>PP xG (PP mins)</td><td><b>{sim['a_pp']:.2f}</b> ({sim['app_m']:.1f}m)</td><td><b>{sim['h_pp']:.2f}</b> ({sim['hpp_m']:.1f}m)</td></tr>
        <tr><td>Pace CF/60 | HDCF%</td><td>{a['cf60']:.1f} | {a['hdcf_pct']:.0f}%</td><td>{h['cf60']:.1f} | {h['hdcf_pct']:.0f}%</td></tr>
        <tr><td>Goalie GSAx/Game</td><td><b>{ag['label']}</b>{ag_b} ({ag['adj60']:+.2f})</td><td><b>{hg['label']}</b>{hg_b} ({hg['adj60']:+.2f})</td></tr>
        {dk_game_rows(aa, ha, sim, dk)}
      </table></div>
    </div>
    """


def props_card(title, df, dk=None, game=None):
    rows = "".join(
        f"<tr><td><b>{r['name']}</b> <span class='card-sub'>({r['pos']})</span>"
        f"{SMALL_SAMPLE_BADGE if r['gp'] < 5 else ''}</td>"
        f"<td>{r['toi']:.1f}m <span style='color:#38bdf8;font-size:10px;'>({r['pp']:.1f} PP)</span></td>"
        f"<td><b>{r['sog']:.2f}</b></td>"
        f"<td>{r['xg']:.2f} ({r['atg']:.0f}%)</td>"
        f"<td><b>{r['pts']:.2f}</b> ({r['pt_p']:.0f}%)</td>"
        f"<td>{dk_cell(dk_player(dk, r['name'], 'Pts', game or r.get('game')), r['pts'], 'Pts')}</td></tr>"
        for _, r in df.iterrows()
    )
    return f"""
    <div class='card'>
      <div class='section'>{title}</div>
      <div class='tbl-wrap'><table class='tbl'>
        <tr><th>Player</th><th>TOI (PP)</th><th>SOG</th><th>xG (ATG%)</th><th>Pts (1+%)</th><th>DK Pts</th></tr>{rows}
      </table></div>
    </div>
    """


def ranked_card(title, rows_df, dk=None, mkt="Pts", key="pts"):
    rows = "".join(
        f"<tr><td>{i}</td>"
        f"<td><b>{r['name']}</b> <span class='card-sub'>({r['pos']}) {r['team']} vs {r['opp']}</span>"
        f"{SMALL_SAMPLE_BADGE if r['gp'] < 5 else ''}</td>"
        f"<td><b>{r['sog']:.2f}</b></td>"
        f"<td>{r['xg']:.2f} ({r['atg']:.0f}%)</td>"
        f"<td><b>{r['pts']:.2f}</b> ({r['pt_p']:.0f}%)</td>"
        f"<td>{dk_cell(dk_player(dk, r['name'], mkt, r.get('game')), r[key], mkt)}</td></tr>"
        for i, (_, r) in enumerate(rows_df.iterrows(), start=1)
    )
    return f"""
    <div class='card'>
      <div class='section'>{title}</div>
      <div class='tbl-wrap'><table class='tbl'>
        <tr><th>#</th><th>Player</th><th>SOG</th><th>xG (ATG%)</th><th>Pts (1+%)</th><th>DK {mkt}</th></tr>{rows}
      </table></div>
    </div>
    """


@st.cache_data(ttl=86400, show_spinner=False)
def load_final_boxscore(game_id):
    """Box score for a finished game (cached for a day, since it no longer changes)."""
    resp = requests.get(f"{NHL_API}/gamecenter/{game_id}/boxscore", timeout=15)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=21600, show_spinner="Grading that slate's top projected players...")
def daily_top_players(season, date_str, top_n=50):
    """Top-N projected-points skaters on a past slate, with their actual box score lines."""
    teams_df, goalies_df, skaters_df = load_moneypuck(season)
    day = datetime.date.fromisoformat(date_str)
    prev = (day - datetime.timedelta(days=1)).isoformat()
    day_games = [g for g in games_on(load_schedule(date_str), date_str)]
    m = Model(teams_df, goalies_df, skaters_df, teams_played_on(load_schedule(prev), prev))
    rows = []
    for g in day_games:
        try:
            box = load_final_boxscore(g["id"])
        except Exception:  # noqa: BLE001
            continue
        if box.get("gameState") not in ("FINAL", "OFF"):
            continue
        actual = parse_box_players(box)
        aa, ha = g["away"], g["home"]
        try:
            ag = m.get_goalie(aa, goalie_id=starter_goalie_id(box, "awayTeam"))
            hg = m.get_goalie(ha, goalie_id=starter_goalie_id(box, "homeTeam"))
            df_a = m.project_skaters(aa, ha, False, hg["adj60"], player_ids=dressed_ids(box, "awayTeam"))
            df_h = m.project_skaters(ha, aa, True, ag["adj60"], player_ids=dressed_ids(box, "homeTeam"))
        except Exception:  # noqa: BLE001
            continue
        for df, team, opp in ((df_a, aa, ha), (df_h, ha, aa)):
            for _, r in df.iterrows():
                act = actual.get(r["playerId"])
                if act is None:
                    continue
                rows.append({**{k: r[k] for k in ("name", "pos", "sog", "xg", "atg", "pts", "pt_p")},
                             "team": team, "opp": opp, "act_pts": act["pts"], "act_g": act["g"],
                             "act_a": act["a"], "act_sog": act["sog"]})
    cols = ["name", "pos", "team", "opp", "sog", "xg", "atg", "pts", "pt_p", "act_pts", "act_g", "act_a", "act_sog"]
    df = pd.DataFrame(rows, columns=cols)
    return df.sort_values("pts", ascending=False).head(top_n).reset_index(drop=True)


def parse_box_players(box):
    out = {}
    stats = box.get("playerByGameStats", {})
    for side in ("awayTeam", "homeTeam"):
        for group in ("forwards", "defense"):
            for p in stats.get(side, {}).get(group, []):
                out[p.get("playerId")] = {
                    "g": p.get("goals", 0), "a": p.get("assists", 0),
                    "pts": p.get("points", 0), "sog": p.get("sog", 0),
                }
    return out


def backtest_rows(df, actual):
    html = ""
    for _, r in df.iterrows():
        act = actual.get(r["playerId"])
        if act is None:
            continue
        html += (
            f"<tr><td><b>{r['name']}</b></td>"
            f"<td>{r['sog']:.1f} vs <b>{act['sog']}</b></td>"
            f"<td>{r['pts']:.2f} vs <b>{act['pts']}</b> ({act['g']}G {act['a']}A)</td></tr>"
        )
    return html


LIVE_STATES = ("LIVE", "CRIT")
FINAL_STATES = ("FINAL", "OFF")
STATUS_ORDER = {"live": 0, "sched": 1, "final": 2}


def _ordinal(n):
    return {1: "1st", 2: "2nd", 3: "3rd"}.get(n, f"{n}th")


def _period_label(pdesc):
    num = pdesc.get("number") or 0
    ptype = pdesc.get("periodType", "REG")
    if ptype == "SO":
        return "SO"
    if ptype == "OT" or num > 3:
        n_ot = max(num - 3, 1)
        return "OT" if n_ot == 1 else f"{n_ot}OT"
    return f"P{num}" if num else ""


def game_status(g, box):
    """Status dot kind ('live'/'final'/'sched') and display text, e.g. 'P2 · 12:34'."""
    box = box or {}
    state = box.get("gameState", g["state"])
    pdesc = box.get("periodDescriptor", {}) or {}
    clock = box.get("clock", {}) or {}
    if state in LIVE_STATES:
        if clock.get("inIntermission"):
            num = pdesc.get("number") or 0
            text = f"{_ordinal(num)} INT" + (f" · {clock.get('timeRemaining')}" if clock.get("timeRemaining") else "")
        else:
            label = _period_label(pdesc)
            text = f"{label} · {clock.get('timeRemaining', '')}".strip(" ·") if label != "SO" else "Shootout"
        return {"kind": "live", "label": "LIVE", "text": text, "state": state}
    if state in FINAL_STATES:
        label = _period_label(pdesc)
        suffix = f"/{label}" if label in ("SO",) or "OT" in label else ""
        return {"kind": "final", "label": f"FINAL{suffix}", "text": "", "state": state}
    return {"kind": "sched", "label": g["time"], "text": "", "state": state}


def live_card(g, box, status, sim):
    aa, ha = g["away"], g["home"]
    box = box or {}
    started = status["kind"] != "sched"
    act_a = (box.get("awayTeam", {}).get("score") or 0) if started else None
    act_h = (box.get("homeTeam", {}).get("score") or 0) if started else None
    sog_a = box.get("awayTeam", {}).get("sog") or 0
    sog_h = box.get("homeTeam", {}).get("sog") or 0

    dot = f"<span class='dot {status['kind']}'></span>"
    status_txt = f"<b>{status['label']}</b>" + (f" <span class='card-sub'>{status['text']}</span>" if status["text"] else "")
    a_b = B2B_BADGE if sim["a_b2b"] else ""
    h_b = B2B_BADGE if sim["h_b2b"] else ""

    pred_w = aa if sim["proj_a"] > sim["proj_h"] else ha
    if status["kind"] == "final":
        act_w = aa if act_a > act_h else (ha if act_h > act_a else "TIE")
        call = "✅ HIT" if pred_w == act_w else "❌ MISS"
    elif status["kind"] == "live":
        leader = aa if act_a > act_h else (ha if act_h > act_a else None)
        call = "⏳ TIED" if leader is None else ("⏳ ON TRACK" if leader == pred_w else "⏳ TRAILING")
    else:
        call = "🕒 PREGAME"

    if started:
        act_tot = act_a + act_h
        score = f"{aa}{a_b} <b>{act_a}</b> — <b>{act_h}</b> {ha}{h_b}"
        table = f"""
          <div class='tbl-wrap'><table class='tbl' style='font-size:13px;'>
            <tr><th>Metric</th><th>Proj</th><th>Actual</th><th>Diff</th></tr>
            <tr><td><b>{aa} Goals</b></td><td>{sim['proj_a']:.2f}</td><td><b>{act_a}</b></td><td>{act_a - sim['proj_a']:+.2f}</td></tr>
            <tr><td><b>{ha} Goals</b></td><td>{sim['proj_h']:.2f}</td><td><b>{act_h}</b></td><td>{act_h - sim['proj_h']:+.2f}</td></tr>
            <tr><td><b>Total Goals</b></td><td>{sim['proj_tot']:.2f}</td><td><b>{act_tot}</b></td><td>{act_tot - sim['proj_tot']:+.2f}</td></tr>
            <tr><td><b>Shots on Goal</b></td><td>Pace {sim['a']['ca60']:.0f} / {sim['h']['ca60']:.0f}</td><td><b>{sog_a} — {sog_h}</b></td><td>{ha} {sog_h - sog_a:+d}</td></tr>
          </table></div>"""
    else:
        score = f"{aa}{a_b} @ {ha}{h_b}"
        table = f"""
          <div class='summary'>
            <div><b>Proj:</b> {aa} {sim['proj_a']:.2f} — {ha} {sim['proj_h']:.2f}</div>
            <div><b>Win:</b> {aa} {sim['a_win']*100:.0f}% | {ha} {sim['h_win']*100:.0f}%</div>
            <div><b>Tot:</b> {sim['proj_tot']:.2f}</div>
          </div>"""
    return f"""
    <div class='card' style='margin-bottom:4px;'>
      <div class='card-head'>
        <div style='font-size:12px;'>{dot}{status_txt}</div>
        <div class='badge bal' style='font-size:11px;'>Model pick {pred_w}: {call}</div>
      </div>
      <div class='card-title' style='margin-bottom:8px;'>{score}</div>
      {table}
    </div>
    """


def live_label(g, box, status):
    """One-line collapsed row: status dot, clock, and score (or start time)."""
    aa, ha = g["away"], g["home"]
    box = box or {}
    if status["kind"] == "sched":
        return f"⚪ {g['time']} · **{aa} @ {ha}**"
    a = box.get("awayTeam", {}).get("score") or 0
    h = box.get("homeTeam", {}).get("score") or 0
    if status["kind"] == "live":
        clock = f" {status['text']}" if status["text"] else ""
        return f"🟢 LIVE{clock} · **{aa} {a} – {h} {ha}**"
    return f"🔴 {status['label']} · **{aa} {a} – {h} {ha}**"


def skater_compare_card(aa, a_df, ha, h_df, actual):
    empty = "<tr><td colspan='3'>No box score yet</td></tr>"
    a_rows = backtest_rows(a_df, actual) or empty
    h_rows = backtest_rows(h_df, actual) or empty
    return f"""
    <div class='card'>
      <div class='section'>{aa} Skaters (Proj vs Actual)</div>
      <div class='tbl-wrap'><table class='tbl' style='margin-bottom:10px;'>
        <tr><th>Player</th><th>SOG</th><th>Points</th></tr>{a_rows}
      </table></div>
      <div class='section'>{ha} Skaters (Proj vs Actual)</div>
      <div class='tbl-wrap'><table class='tbl'>
        <tr><th>Player</th><th>SOG</th><th>Points</th></tr>{h_rows}
      </table></div>
    </div>
    """


# ----------------------------------------------------------------------------
# DraftKings lines: lines/{date}.csv in the repo, The Odds API (optional key), manual edits
# ----------------------------------------------------------------------------
LINE_MARKETS = {"ML": "ML", "TOTAL": "TOT", "TOT": "TOT", "SOG": "SOG", "PTS": "Pts", "POINTS": "Pts",
                "GOAL": "Goal", "ATG": "Goal"}


def norm_name(name):
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", s.lower())


def dk_player(dk, name, mkt, game=None):
    """DK prop for a player: game-specific entry first, then name-only (two players can share a name)."""
    players = (dk or {}).get("player", {})
    nm = norm_name(name)
    if game and (game[0], game[1], nm, mkt) in players:
        return players[(game[0], game[1], nm, mkt)]
    return players.get((nm, mkt))


def _odds(v):
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)) or str(v).strip() == "":
            return None
        return int(float(str(v).replace("+", "")))
    except ValueError:
        return None


def _line(v):
    try:
        if v is None or str(v).strip() == "" or (isinstance(v, float) and math.isnan(v)):
            return None
        return float(v)
    except ValueError:
        return None


def empty_lines():
    return {"ml": {}, "total": {}, "player": {}}


@st.cache_data(ttl=300, show_spinner=False)
def load_file_lines(date_str):
    """DK lines transcribed into lines/{date}.csv (columns: game,market,player,line,over_odds,under_odds)."""
    out = empty_lines()
    path = os.path.join(LINES_DIR, f"{date_str}.csv")
    if not os.path.exists(path):
        return out, 0
    df = pd.read_csv(path, dtype=str).fillna("")
    n = 0
    for _, r in df.iterrows():
        mkt = LINE_MARKETS.get(r.get("market", "").strip().upper())
        game = r.get("game", "").replace(" ", "").upper()
        away, _, home = game.partition("@")
        if not mkt:
            continue
        if mkt == "ML" and away and home:
            odds = _odds(r.get("over_odds"))
            if odds is not None:
                out["ml"].setdefault((away, home), {})[r.get("player", "").strip().upper()] = odds
                n += 1
        elif mkt == "TOT" and away and home:
            out["total"][(away, home)] = {"line": _line(r.get("line")), "over": _odds(r.get("over_odds")),
                                         "under": _odds(r.get("under_odds"))}
            n += 1
        elif r.get("player"):
            line = _line(r.get("line"))
            rec = {"line": 0.5 if (mkt == "Goal" and line is None) else line,
                   "over": _odds(r.get("over_odds")), "under": _odds(r.get("under_odds"))}
            nm = norm_name(r["player"])
            if away and home:
                out["player"][(away, home, nm, mkt)] = rec
            out["player"].setdefault((nm, mkt), rec)
            n += 1
    return out, n


@st.cache_data(ttl=900, show_spinner="Fetching DraftKings odds...")
def load_odds_api(api_key, date_str, include_props):
    """DraftKings lines from The Odds API. Returns (lines, count, error)."""
    out, n = empty_lines(), 0
    base = {"apiKey": api_key, "regions": "us", "bookmakers": "draftkings", "oddsFormat": "american"}
    try:
        resp = requests.get(f"{ODDS_API}/odds", params={**base, "markets": "h2h,totals"}, timeout=20)
        resp.raise_for_status()
        events = resp.json()
    except Exception as exc:  # noqa: BLE001
        return out, 0, str(exc)
    keep = []
    for ev in events:
        start = datetime.datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00")).astimezone(EST)
        if start.strftime("%Y-%m-%d") != date_str:
            continue
        away, home = TEAM_ABBR.get(ev.get("away_team")), TEAM_ABBR.get(ev.get("home_team"))
        if not (away and home):
            continue
        keep.append((ev["id"], away, home))
        for bk in ev.get("bookmakers", []):
            for m in bk.get("markets", []):
                if m["key"] == "h2h":
                    for o in m["outcomes"]:
                        ab = TEAM_ABBR.get(o["name"])
                        if ab:
                            out["ml"].setdefault((away, home), {})[ab] = int(o["price"])
                            n += 1
                elif m["key"] == "totals":
                    t = {"line": None, "over": None, "under": None}
                    for o in m["outcomes"]:
                        t["line"] = float(o.get("point")) if o.get("point") is not None else t["line"]
                        t["over" if o["name"] == "Over" else "under"] = int(o["price"])
                    out["total"][(away, home)] = t
                    n += 1
    if include_props:
        prop_map = {"player_shots_on_goal": "SOG", "player_points": "Pts", "player_goal_scorer_anytime": "Goal"}
        for eid, away, home in keep:
            try:
                resp = requests.get(f"{ODDS_API}/events/{eid}/odds",
                                    params={**base, "markets": ",".join(prop_map)}, timeout=20)
                resp.raise_for_status()
                ev = resp.json()
            except Exception:  # noqa: BLE001
                continue
            for bk in ev.get("bookmakers", []):
                for m in bk.get("markets", []):
                    mkt = prop_map.get(m["key"])
                    if not mkt:
                        continue
                    for o in m["outcomes"]:
                        player = o.get("description") or o.get("name")
                        rec = out["player"].setdefault((away, home, norm_name(player), mkt),
                                                       {"line": 0.5 if mkt == "Goal" else None, "over": None, "under": None})
                        out["player"].setdefault((norm_name(player), mkt), rec)
                        if o.get("point") is not None:
                            rec["line"] = float(o["point"])
                        side = "under" if o.get("name") in ("Under", "No") else "over"
                        rec[side] = int(o["price"])
                        n += 1
    return out, n, None


def merge_lines(*sources):
    out = empty_lines()
    for src in sources:
        for k in out:
            for key, val in src.get(k, {}).items():
                if k == "ml":
                    out[k].setdefault(key, {}).update(val)
                else:
                    out[k][key] = val
    return out


def side_price(dk_rec, line, side):
    """DK odds for one side, only if the DK line matches the line being priced."""
    if not dk_rec or dk_rec.get("line") is None or line is None or abs(dk_rec["line"] - line) > 1e-9:
        return None, None
    mine = dk_rec.get("over" if side == "Over" else "under")
    other = dk_rec.get("under" if side == "Over" else "over")
    return mine, other


def implied_no_vig(mine, other):
    """Implied % for my side; removes the vig when both sides are known."""
    p_m = american_to_prob(mine) if mine is not None else None
    if p_m is None:
        return None
    p_o = american_to_prob(other) if other is not None else None
    return p_m / (p_m + p_o) if p_o else p_m


def fmt_odds(o):
    return f"{o:+d}" if o is not None else "NA"


def dk_cell(dk_rec, mean, market_label):
    """Compact 'O0.5 -180 · edge +4' for a player table, or NA."""
    if not dk_rec or dk_rec.get("line") is None or dk_rec.get("over") is None:
        return "<span class='card-sub'>NA</span>"
    p_o, p_u, _ = line_probs(float(mean), dk_rec["line"])
    side = "Over" if p_o >= p_u else "Under"
    mine, other = side_price(dk_rec, dk_rec["line"], side)
    imp = implied_no_vig(mine, other)
    if imp is None:
        return f"O{dk_rec['line']:g} {fmt_odds(dk_rec['over'])}"
    p = p_o if side == "Over" else p_u
    edge = (p - imp) * 100
    color = "#86efac" if edge >= 3 else ("#fca5a5" if edge <= -3 else "#cbd5e1")
    return f"{side[0]}{dk_rec['line']:g} {fmt_odds(mine)} · <b style='color:{color}'>{edge:+.0f}</b>"


# ----------------------------------------------------------------------------
# Bets board helpers
# ----------------------------------------------------------------------------
STATUS_ICON = {"pregame": "⚪", "on": "🟢", "off": "🟠", "tied": "🟡", "hit": "✅", "miss": "❌", "push": "➖"}
STATUS_TEXT = {"pregame": "Pregame", "on": "On pace", "off": "Off pace", "tied": "Tied",
               "hit": "Hit", "miss": "Miss", "push": "Push"}


def pmf_over_under(pmf, line):
    """Over / under / push probabilities from a goals PMF."""
    ks = np.arange(len(pmf))
    over = float(pmf[ks > line].sum())
    push = float(pmf[ks == line].sum()) if float(line).is_integer() else 0.0
    return over, max(1.0 - over - push, 0.0), push


def default_total_line(proj_tot):
    return 6.5 if proj_tot >= 6.0 else 5.5


def default_sog_line(mean):
    return 1.5 if mean < 2.2 else (2.5 if mean < 3.4 else 3.5)


def elapsed_fraction(live):
    """Share of regulation played (0-1) from a scoreboard entry; None before puck drop."""
    if not live or live.get("gameState") not in LIVE_STATES + FINAL_STATES:
        return None
    if live.get("gameState") in FINAL_STATES:
        return 1.0
    pdesc = live.get("periodDescriptor", {}) or {}
    clock = live.get("clock", {}) or {}
    num = pdesc.get("number") or 1
    if num > 3 or pdesc.get("periodType") in ("OT", "SO"):
        return 1.0
    if clock.get("inIntermission"):
        return min(num * 20.0 / 60.0, 1.0)
    try:
        mm, ss = str(clock.get("timeRemaining", "20:00")).split(":")
        remaining = int(mm) + int(ss) / 60.0
    except ValueError:
        remaining = 20.0
    return max(min(((num - 1) * 20.0 + (20.0 - remaining)) / 60.0, 1.0), 0.0)


def grade_count(side, line, current, final, frac):
    """Status for an over/under on a counting stat (goals, SOG, points)."""
    if frac is None:
        return "pregame", None
    pace = current / max(frac, 0.08) if frac < 1.0 else float(current)
    if side == "Over" and current > line:
        return "hit", pace
    if side == "Under" and current > line:
        return "miss", pace
    if final:
        if current == line:
            return "push", pace
        return ("hit" if side == "Under" else "miss"), pace
    return ("on" if (pace > line) == (side == "Over") else "off"), pace


def choose_side(p_over, p_under, odds_over, odds_under):
    """Pick the side with the higher model chance; return side, model %, odds, implied %, edge.
    Implied % is no-vig when both sides' odds are known."""
    if p_over >= p_under:
        side, p, odds, other = "Over", p_over, odds_over, odds_under
    else:
        side, p, odds, other = "Under", p_under, odds_under, odds_over
    imp = implied_no_vig(odds, other)
    edge = (p - imp) * 100 if imp else None
    return side, p, odds, imp, edge


def price_pick(p_over, p_under, line, ov, dk_rec):
    """Choose the model's side and price it: manual odds win, else DK odds at a matching line, else NA."""
    side = "Over" if p_over >= p_under else "Under"
    if "odds" in ov:
        side, p, odds, imp, edge = choose_side(p_over, p_under, ov["odds"], ov["odds"])
        imp = american_to_prob(odds)
        return side, p, odds, imp, ((p - imp) * 100 if imp else None), "Manual"
    o_over, o_under = side_price(dk_rec, line, "Over")
    side, p, odds, imp, edge = choose_side(p_over, p_under, o_over, o_under)
    return side, p, odds, imp, edge, "DK"


def bet_row_label(b):
    icon = STATUS_ICON[b["status"]]
    if b["odds"] is not None:
        price = f" · {b['src']} {fmt_odds(b['odds'])}"
        price += f" · edge {b['edge']:+.0f}" if b["edge"] is not None else ""
    else:
        price = " · DK NA"
    return f"{icon} **{b['pick']}** · {b['p']*100:.0f}%{price} · {b['now']}"


def bet_detail_card(b):
    odds_txt = f"{b['src']} {fmt_odds(b['odds'])}" if b["odds"] is not None else "DK NA"
    imp_txt = f"{b['imp']*100:.1f}%" if b["imp"] else "NA"
    edge_txt = f"{b['edge']:+.1f}%" if b["edge"] is not None else "NA"
    return f"""
    <div class='card'>
      <div class='section'>{b['pick']} <span class='card-sub'>· {b['game']} · {b['time']}</span></div>
      <div class='tbl-wrap'><table class='tbl'>
        <tr><th>Pregame (locked)</th><th>Live / result</th></tr>
        <tr><td>Proj {b['proj_txt']}<br>Model {b['p']*100:.1f}% · fair {prob_to_american(b['p'])}</td>
            <td>{STATUS_ICON[b['status']]} {STATUS_TEXT[b['status']]}<br>{b['now']}</td></tr>
        <tr><td>Line {b['line_txt']} · {odds_txt}</td><td>Implied {imp_txt} · Edge {edge_txt}</td></tr>
      </table></div>
    </div>
    """


# ----------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------
st.title("🏒 NHL Game Flow Model")

est_now = datetime.datetime.now(EST)
slate_date = st.date_input("Slate date (ET)", value=est_now.date())
date_str = slate_date.strftime("%Y-%m-%d")
prev_str = (slate_date - datetime.timedelta(days=1)).strftime("%Y-%m-%d")

try:
    teams_df, goalies_df, skaters_df = load_moneypuck(SEASON)
    sched = load_schedule(date_str)
    sched_prev = load_schedule(prev_str)
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not load data: {exc}")
    st.stop()

games = games_on(sched, date_str)
b2b_teams = teams_played_on(sched_prev, prev_str)
model = Model(teams_df, goalies_df, skaters_df, b2b_teams)

st_html(
    f"<div class='meta'>📅 <b>{date_str}</b> ({len(games)} games) &nbsp;|&nbsp; "
    f"Auto-B2B: <b>{', '.join(sorted(b2b_teams)) if b2b_teams else 'None'}</b></div>")

if not games:
    st.info("No games scheduled for this date.")
    st.stop()

# Goalie overrides (shared by Slate and Props tabs)
if "goalie_overrides" not in st.session_state:
    st.session_state.goalie_overrides = {}
overrides = st.session_state.goalie_overrides

slate_teams = sorted({g["away"] for g in games} | {g["home"] for g in games})
with st.expander("🥅 Goalie overrides"):
    c1, c2 = st.columns(2)
    sel_team = c1.selectbox("Team", slate_teams, key="ovr_team")
    options = model.team_goalies(sel_team)["name"].tolist()
    current = overrides.get(sel_team, options[0] if options else None)
    sel_goalie = c2.selectbox(
        "Goalie", options, index=options.index(current) if current in options else 0, key=f"ovr_goalie_{sel_team}"
    )
    b1, b2 = st.columns(2)
    if b1.button("Swap & re-sim", type="primary", width="stretch") and sel_goalie:
        overrides[sel_team] = sel_goalie
    if b2.button("Reset all", width="stretch"):
        overrides.clear()
    if overrides:
        st.caption("Active: " + ", ".join(f"{t}: {n}" for t, n in sorted(overrides.items())))

# Box scores for games at or past warmups: confirmed starting goalies and dressed lineups.
boxes = {}
_soon = datetime.datetime.now(EST) + datetime.timedelta(minutes=45)
for g in games:
    if g["state"] != "FUT" or g["start"] <= _soon:
        try:
            boxes[g["id"]] = load_boxscore(g["id"])
        except Exception:  # noqa: BLE001
            pass


def resolve_goalie(g, side, use_override=True):
    """Manual override > NHL-confirmed starter > projected #1 (most GP)."""
    abbr = g[side]
    box_side = "awayTeam" if side == "away" else "homeTeam"
    return model.get_goalie(
        abbr,
        overrides.get(abbr) if use_override else None,
        starter_goalie_id(boxes.get(g["id"]), box_side),
    )


# DraftKings lines: repo file for the date, then The Odds API if a key is set (API wins where both exist).
try:
    file_lines, n_file = load_file_lines(date_str)
except Exception:  # noqa: BLE001
    file_lines, n_file = empty_lines(), 0
try:
    odds_key = st.secrets.get("ODDS_API_KEY")
except Exception:  # noqa: BLE001
    odds_key = None
api_lines, n_api, api_err = empty_lines(), 0, None
with st.expander("📈 DraftKings lines"):
    src_txt = [f"lines file: {n_file} lines" if n_file else "no lines file for this date"]
    if odds_key:
        want_props = st.toggle("Also fetch DK player props (uses more API credits)", value=True, key="odds_props")
        api_lines, n_api, api_err = load_odds_api(odds_key, date_str, want_props)
        src_txt.append(f"The Odds API: {n_api} prices" if not api_err else f"The Odds API error: {api_err}")
    else:
        src_txt.append("no ODDS_API_KEY in Streamlit secrets")
    st.caption(" · ".join(src_txt) + ". Picks without a DK line show NA.")
    st.link_button("Open NHL lines on DraftKings ↗", DK_NHL_URL, width="stretch")
dk_lines = merge_lines(file_lines, api_lines)

tab_bets, tab_slate, tab_props, tab_live, tab_rec = st.tabs(["Bets", "Slate", "Player Props", "Live Tracker", "Record"])

with tab_slate:
    slate_open = st.toggle("Expand all games", key="slate_expand_all")
    for g in games:
        aa, ha = g["away"], g["home"]
        try:
            ag, hg = resolve_goalie(g, "away"), resolve_goalie(g, "home")
            sim = model.simulate_game(aa, ha, ag, hg)
            fav = aa if sim["a_win"] >= sim["h_win"] else ha
            fav_p = max(sim["a_win"], sim["h_win"]) * 100
            b2b = " · B2B" if (sim["a_b2b"] or sim["h_b2b"]) else ""
            label = (f"**{aa} @ {ha}** · {g['time']} · {sim['proj_a']:.1f}–{sim['proj_h']:.1f}"
                     f" · {fav} {fav_p:.0f}% · Tot {sim['proj_tot']:.1f}{b2b}")
            with st.expander(label, expanded=slate_open):
                st_html(slate_card(g, sim, ag, hg, dk_lines))
        except Exception as exc:  # noqa: BLE001
            st.error(f"{aa} @ {ha}: could not simulate ({exc})")

def project_game(g):
    """Goalies and skater projections for one game, using the best available lineup."""
    aa, ha = g["away"], g["home"]
    box = boxes.get(g["id"])
    a_ids, h_ids = dressed_ids(box, "awayTeam") if box else None, dressed_ids(box, "homeTeam") if box else None
    lineup_src = "tonight's dressed lineups" if (a_ids and h_ids) else "current NHL rosters"
    a_ids = a_ids or load_roster(aa)
    h_ids = h_ids or load_roster(ha)
    if not (a_ids and h_ids):
        lineup_src = "MoneyPuck season data (roster unavailable)"
    ag, hg = resolve_goalie(g, "away"), resolve_goalie(g, "home")
    df_a = model.project_skaters(aa, ha, False, hg["adj60"], player_ids=a_ids)
    df_h = model.project_skaters(ha, aa, True, ag["adj60"], player_ids=h_ids)
    return ag, hg, df_a, df_h, lineup_src


with tab_props:
    # -- Player search + prop line checker --------------------------------
    st_html("<div class='section'>🔎 Player search & line check</div>")
    index = []
    for g in games:
        try:
            ag, hg, df_a, df_h, _ = project_game(g)
        except Exception:  # noqa: BLE001
            continue
        for df, team, opp, opp_g in ((df_a, g["away"], g["home"], hg), (df_h, g["home"], g["away"], ag)):
            for _, r in df.iterrows():
                index.append({**r.to_dict(), "team": team, "opp": opp, "opp_goalie": opp_g, "time": g["time"],
                              "game": (g["away"], g["home"])})
    index.sort(key=lambda r: r["name"])
    labels_ix = [f"{r['name']} ({r['team']} vs {r['opp']})" for r in index]
    sel = st.selectbox("Player", labels_ix, index=None, placeholder="Type a name: anyone playing tonight", key="search_player")
    if sel:
        r = index[labels_ix.index(sel)]
        st_html(
            props_card(f"{r['name']} · {r['team']} vs {r['opp']} ({r['time']}) · opp G {r['opp_goalie']['name']}{goalie_badges(r['opp_goalie'])}",
                       pd.DataFrame([r]), dk_lines))
        markets = {"Shots on Goal": ("sog", 2.5), "Points": ("pts", 0.5), "Goals": ("xg", 0.5)}
        c1, c2 = st.columns(2)
        market = c1.selectbox("Market", list(markets), key="lc_market")
        key, default_line = markets[market]
        line = c2.number_input("Line", min_value=0.0, max_value=15.0, value=default_line, step=0.5, key=f"lc_line_{key}")
        c3, c4 = st.columns(2)
        o_odds = c3.number_input("Over odds (American)", value=-110, step=5, key="lc_over")
        u_odds = c4.number_input("Under odds (American)", value=-110, step=5, key="lc_under")

        lam = float(r[key])
        p_over, p_under, p_push = line_probs(lam, line)
        imp_o, imp_u = american_to_prob(o_odds), american_to_prob(u_odds)
        if imp_o and imp_u:
            vig = imp_o + imp_u
            imp_o, imp_u = imp_o / vig, imp_u / vig
        # Edge vs. no-vig market, using probabilities that exclude pushes.
        settled = max(p_over + p_under, 1e-9)
        m_o, m_u = p_over / settled, p_under / settled
        edge_o = (m_o - imp_o) * 100 if imp_o else None
        edge_u = (m_u - imp_u) * 100 if imp_u else None

        def edge_cell(e):
            if e is None:
                return "—"
            color = "#86efac" if e >= 3 else ("#fca5a5" if e <= -3 else "#cbd5e1")
            return f"<b style='color:{color}'>{e:+.1f}%</b>"

        push_txt = f" · Push {p_push*100:.0f}%" if p_push > 0 else ""
        st_html(
            f"""
            <div class='card'>
              <div class='section'>{market} {line:g} · model mean {lam:.2f}{push_txt}</div>
              <div class='tbl-wrap'><table class='tbl'>
                <tr><th>Side</th><th>Model %</th><th>Fair odds</th><th>Market (no-vig)</th><th>Edge</th></tr>
                <tr><td><b>Over</b></td><td>{p_over*100:.1f}%</td><td>{prob_to_american(m_o)}</td><td>{f"{imp_o*100:.1f}%" if imp_o else "—"}</td><td>{edge_cell(edge_o)}</td></tr>
                <tr><td><b>Under</b></td><td>{p_under*100:.1f}%</td><td>{prob_to_american(m_u)}</td><td>{f"{imp_u*100:.1f}%" if imp_u else "—"}</td><td>{edge_cell(edge_u)}</td></tr>
              </table></div>
            </div>
            """)
        st.caption("Model % assumes a Poisson count around the projected mean. Edge is the model's chance minus the market's no-vig chance; green ≥ +3%.")
        st.link_button("Open NHL player props on DraftKings ↗", DK_NHL_URL, width="stretch")

    st.divider()

    view = st.radio("View", ["All players (ranked)", "By game"], horizontal=True, key="props_view")

    if view == "All players (ranked)":
        sort_opts = {"Points": "pts", "Shots on Goal": "sog", "Goals (xG)": "xg"}
        c1, c2 = st.columns(2)
        sort_by = c1.selectbox("Rank by", list(sort_opts), key="rank_by")
        top = c2.selectbox("Show", ["Top 25", "Top 50", "All"], index=1, key="rank_top")
        ranked = pd.DataFrame(index).sort_values(sort_opts[sort_by], ascending=False) if index else pd.DataFrame()
        if ranked.empty:
            st.info("No skater projections available for this slate.")
        else:
            if top != "All":
                ranked = ranked.head(int(top.split()[1]))
            st.caption(f"{len(ranked)} of {len(index)} skaters across {len(games)} games, ranked by projected {sort_by.lower()}.")
            mkt_key = {"Points": ("Pts", "pts"), "Shots on Goal": ("SOG", "sog"), "Goals (xG)": ("Goal", "xg")}[sort_by]
            st_html(ranked_card(f"Tonight's top projected {sort_by.lower()}", ranked, dk_lines, *mkt_key))

    else:
        # -- Full matchup board --------------------------------------------
        labels = [f"{g['away']} @ {g['home']} ({g['time']})" for g in games]
        pick = st.selectbox("Matchup", labels, key="props_game")
        g = games[labels.index(pick)]
        aa, ha = g["away"], g["home"]
        show = st.radio("Show", ["All", "Top 8", "Top 12"], horizontal=True, key="props_show")
        try:
            ag, hg, df_a, df_h, lineup_src = project_game(g)
            if show != "All":
                n = int(show.split()[1])
                df_a, df_h = df_a.head(n), df_h.head(n)
            st.caption(f"{len(df_a)} {aa} and {len(df_h)} {ha} skaters, from {lineup_src}. Sorted by projected points.")
            a_b = B2B_BADGE if aa in b2b_teams else ""
            h_b = B2B_BADGE if ha in b2b_teams else ""
            st_html(
                props_card(f"{aa}{a_b} Skaters (vs {hg['name']}{goalie_badges(hg)})", df_a, dk_lines, (aa, ha))
                + props_card(f"{ha}{h_b} Skaters (vs {ag['name']}{goalie_badges(ag)})", df_h, dk_lines, (aa, ha)))
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not project {aa} @ {ha}: {exc}")


def build_bets(scores, bet_overrides, dk):
    """Every game (ML, total) and player (SOG, points, goal) pick on the slate, with status."""
    bets = []
    for g in games:
        aa, ha, gid = g["away"], g["home"], g["id"]
        live = scores.get(gid, {})
        frac = elapsed_fraction(live)
        final = live.get("gameState") in FINAL_STATES
        a_now = (live.get("awayTeam", {}) or {}).get("score") or 0
        h_now = (live.get("homeTeam", {}) or {}).get("score") or 0
        status_txt = game_status(g, live or None)
        clock_txt = status_txt["text"] if status_txt["kind"] == "live" else ("Final" if final else g["time"])
        game_lbl = f"{aa} @ {ha}"
        try:
            ag, hg, df_a, df_h, _ = project_game(g)
            sim = model.simulate_game(aa, ha, ag, hg)
        except Exception:  # noqa: BLE001
            continue

        # Moneyline: model's side; manual odds > DK odds > NA.
        bid = f"{gid}-ML"
        ov = bet_overrides.get(bid, {})
        pick_team = aa if sim["a_win"] >= sim["h_win"] else ha
        opp_team = ha if pick_team == aa else aa
        p = max(sim["a_win"], sim["h_win"])
        dk_ml = dk["ml"].get((aa, ha), {})
        if "odds" in ov:
            odds, other, src = ov["odds"], None, "Manual"
        else:
            odds, other, src = dk_ml.get(pick_team), dk_ml.get(opp_team), "DK"
        imp = implied_no_vig(odds, other)
        if frac is None:
            status, now = "pregame", g["time"]
        else:
            mine, theirs = (a_now, h_now) if pick_team == aa else (h_now, a_now)
            if final:
                status = "hit" if mine > theirs else "miss"
            else:
                status = "on" if mine > theirs else ("off" if mine < theirs else "tied")
            now = f"{aa} {a_now}–{h_now} {ha} · {clock_txt}"
        bets.append({"id": bid, "kind": "Game", "market": "ML", "pick": f"{pick_team} ML ({game_lbl})",
                     "game": game_lbl, "time": g["time"], "p": p, "odds": odds, "imp": imp, "src": src,
                     "edge": (p - imp) * 100 if imp else None, "line": None, "line_txt": "ML",
                     "proj_txt": f"{aa} {sim['proj_a']:.2f} – {sim['proj_h']:.2f} {ha}",
                     "status": status, "now": now})

        # Game total
        bid = f"{gid}-TOT"
        ov = bet_overrides.get(bid, {})
        dk_tot = dk["total"].get((aa, ha))
        line = ov.get("line", (dk_tot or {}).get("line") or default_total_line(sim["proj_tot"]))
        p_o, p_u, _ = pmf_over_under(sim["tot_pmf"], line)
        side, p, odds, imp, edge, src = price_pick(p_o, p_u, line, ov, dk_tot)
        status, pace = grade_count(side, line, a_now + h_now, final, frac)
        now = g["time"] if frac is None else f"{a_now + h_now} goals · {clock_txt}" + (f" · pace {pace:.1f}" if not final else "")
        bets.append({"id": bid, "kind": "Game", "market": "Total", "pick": f"{game_lbl} {side[0]} {line:g}",
                     "game": game_lbl, "time": g["time"], "p": p, "odds": odds, "imp": imp, "edge": edge, "src": src,
                     "line": line, "line_txt": f"{side} {line:g}", "proj_txt": f"{sim['proj_tot']:.2f} goals",
                     "status": status, "now": now})

        # Players
        box = boxes.get(gid)
        actual = parse_box_players(box) if box else {}
        for df, team in ((df_a, aa), (df_h, ha)):
            for _, r in df.iterrows():
                act = actual.get(r["playerId"], {"g": 0, "a": 0, "pts": 0, "sog": 0})
                last = r["name"].split()[-1] if isinstance(r["name"], str) else r["name"]
                markets = (
                    ("SOG", "sog", default_sog_line(r["sog"]), act["sog"], None),
                    ("Pts", "pts", 0.5, act["pts"], None),
                    ("Goal", "xg", 0.5, act["g"], None),
                )
                for mkt, key, dline, cur, dodds in markets:
                    bid = f"{gid}-{r['playerId']}-{mkt}"
                    ov = bet_overrides.get(bid, {})
                    dk_p = dk_player(dk, r["name"], mkt, (aa, ha))
                    line = ov.get("line", (dk_p or {}).get("line") or dline)
                    p_o, p_u, _ = line_probs(float(r[key]), line)
                    if mkt == "Goal":
                        side, p = "Over", p_o
                        if "odds" in ov:
                            odds, src, other = ov["odds"], "Manual", None
                        else:
                            odds, other = side_price(dk_p, line, "Over")
                            src = "DK"
                        imp = implied_no_vig(odds, other)
                        edge = (p - imp) * 100 if imp else None
                        pick = f"{r['name']} ({team}) anytime goal"
                        line_txt = "Anytime goal"
                    else:
                        side, p, odds, imp, edge, src = price_pick(p_o, p_u, line, ov, dk_p)
                        pick = f"{r['name']} ({team}) {side[0]} {line:g} {mkt}"
                        line_txt = f"{side} {line:g} {mkt}"
                    in_lineup = (r["playerId"] in actual) if actual else True
                    status, pace = grade_count(side, line, cur, final, frac if in_lineup else None)
                    if frac is None or not in_lineup:
                        now = g["time"] if in_lineup else "Not in lineup"
                    else:
                        now = f"{cur} {mkt} · {clock_txt}" + (f" · pace {pace:.1f}" if not final else "")
                    bets.append({"id": bid, "kind": "Player", "market": mkt, "pick": pick, "game": game_lbl,
                                 "time": g["time"], "p": p, "odds": odds, "imp": imp, "edge": edge,
                                 "src": src, "line": line, "line_txt": line_txt,
                                 "proj_txt": f"{float(r[key]):.2f} {mkt}", "status": status, "now": now})
    return bets


with tab_bets:
    if "bet_overrides" not in st.session_state:
        st.session_state.bet_overrides = {}
    bet_overrides = st.session_state.bet_overrides

    c1, c2 = st.columns(2)
    bet_kind = c1.radio("Show", ["All", "Games", "Players"], horizontal=True, key="bets_kind")
    min_p = c2.slider("Min model %", 50, 80, 55, step=1, key="bets_min_p")
    c3, c4 = st.columns(2)
    n_show = c3.selectbox("How many", [15, 25, 50, 100], index=1, key="bets_n")
    status_f = c4.selectbox("Status", ["Any", "Pregame", "Live", "Settled"], key="bets_status")

    try:
        scores_now = load_scores(date_str)
    except Exception:  # noqa: BLE001
        scores_now = {}
    all_bets = build_bets(scores_now, bet_overrides, dk_lines)

    def keep(b):
        if bet_kind == "Games" and b["kind"] != "Game":
            return False
        if bet_kind == "Players" and b["kind"] != "Player":
            return False
        if b["p"] * 100 < min_p:
            return False
        if status_f == "Pregame" and b["status"] != "pregame":
            return False
        if status_f == "Live" and b["status"] not in ("on", "off", "tied"):
            return False
        if status_f == "Settled" and b["status"] not in ("hit", "miss", "push"):
            return False
        return True

    rank_by = st.radio("Rank by", ["Model %", "Edge (odds entered)"], horizontal=True, key="bets_rank")

    def rank_key(b):
        if rank_by.startswith("Edge"):
            return (b["edge"] is not None, b["edge"] if b["edge"] is not None else b["p"] * 100)
        return (True, b["p"] * 100)

    shown = sorted((b for b in all_bets if keep(b)), key=rank_key, reverse=True)[:n_show]
    settled = [b for b in shown if b["status"] in ("hit", "miss")]
    live_n = sum(b["status"] in ("on", "off", "tied") for b in shown)
    on_n = sum(b["status"] == "on" for b in shown)
    summary = f"{len(shown)} picks"
    if live_n:
        summary += f" · {on_n}/{live_n} live on pace"
    if settled:
        summary += f" · {sum(b['status'] == 'hit' for b in settled)}-{sum(b['status'] == 'miss' for b in settled)} settled"
    st.caption(summary + ". ⚪ pregame · 🟢 on pace · 🟠 off pace · 🟡 tied · ✅ hit · ❌ miss")

    if not shown:
        st.info("No picks match these filters. Try lowering the minimum model %.")
    for b in shown:
        with st.expander(bet_row_label(b)):
            st_html(bet_detail_card(b))

    with st.expander("✏️ Edit lines & odds"):
        st.caption("Type the book's line and American odds for any pick. Edge = model % minus the odds' implied %; "
                   "picks without odds show model % only.")
        ed = pd.DataFrame([{"id": b["id"], "Pick": b["pick"], "Line": b["line"], "Odds": b["odds"]} for b in shown])
        if not ed.empty:
            edited = st.data_editor(
                ed, key="bets_editor", hide_index=True, width="stretch",
                column_config={
                    "id": None,
                    "Pick": st.column_config.TextColumn(disabled=True),
                    "Line": st.column_config.NumberColumn(step=0.5, min_value=0.0, max_value=15.0),
                    "Odds": st.column_config.NumberColumn(step=5, min_value=-2000, max_value=5000),
                },
            )
            changed = False
            for (_, before), (_, after) in zip(ed.iterrows(), edited.iterrows()):
                ov = dict(bet_overrides.get(after["id"], {}))
                for col, key in (("Line", "line"), ("Odds", "odds")):
                    new, old = after[col], before[col]
                    if pd.isna(new) and pd.isna(old):
                        continue
                    if pd.isna(new) or pd.isna(old) or float(new) != float(old):
                        if pd.isna(new):
                            ov.pop(key, None)
                        else:
                            ov[key] = float(new) if key == "line" else int(new)
                        changed = True
                bet_overrides[after["id"]] = ov
            if changed:
                # Rows can re-rank after an edit, so drop the editor's row-indexed edits before re-rendering.
                st.session_state.pop("bets_editor", None)
                st.rerun()
        if st.button("Reset all lines", key="bets_reset"):
            bet_overrides.clear()
            st.session_state.pop("bets_editor", None)
            st.rerun()
        st.link_button("Open NHL lines on DraftKings ↗", DK_NHL_URL, width="stretch")

with tab_live:
    now_et = datetime.datetime.now(EST)

    def has_started(g, state):
        return state not in ("FUT", "PRE") or g["start"] <= now_et

    # The schedule is cached for 5 minutes, so also treat any game past its start time as possibly live.
    any_live = any(
        g["state"] in LIVE_STATES or (g["state"] not in FINAL_STATES and g["start"] <= now_et) for g in games
    )

    st.toggle("Expand all games", key="live_expand_all")

    @st.fragment(run_every=60 if any_live else None)
    def live_tracker():
        c1, c2 = st.columns([3, 1])
        c1.caption("Auto-refreshes every 60s while games are live." if any_live else "Projection vs. live/final box score.")
        live_open = st.session_state.get("live_expand_all", False)
        if c2.button("Refresh", width="stretch", key="live_refresh"):
            load_scores.clear()
            load_boxscore.clear()

        try:
            scores = load_scores(date_str)
            score_err = None
        except Exception as exc:  # noqa: BLE001
            scores, score_err = {}, exc

        # Uses the NHL-confirmed starter when known (no manual overrides) and no player B2B, as in the notebook.
        entries, box_errs = [], []
        for g in games:
            live = scores.get(g["id"], {})
            state = live.get("gameState", g["state"])
            box = dict(live) if live else None
            if has_started(g, state):
                try:
                    full = load_boxscore(g["id"])
                    boxes[g["id"]] = full
                    box = {**(box or {}), **full}
                    # The scoreboard updates faster than the box score for state and clock.
                    for k in ("gameState", "clock", "periodDescriptor"):
                        if k in live:
                            box[k] = live[k]
                    for side in ("awayTeam", "homeTeam"):
                        live_side = live.get(side, {})
                        merged = dict(full.get(side, {}))
                        merged.update({k: live_side[k] for k in ("score", "sog") if k in live_side})
                        box[side] = merged
                except Exception as exc:  # noqa: BLE001
                    box_errs.append(f"{g['away']}@{g['home']}: {exc}")
            entries.append((g, box, game_status(g, box)))
        entries.sort(key=lambda e: (STATUS_ORDER[e[2]["kind"]], games.index(e[0])))

        if score_err and box_errs:
            st.warning(f"Live data unavailable from the NHL API right now ({score_err}). Showing projections only.")
        elif box_errs:
            st.caption("Box score not available yet for " + ", ".join(e.split(":")[0] for e in box_errs) + ".")

        for g, box, status in entries:
            aa, ha = g["away"], g["home"]
            try:
                ag, hg = resolve_goalie(g, "away", use_override=False), resolve_goalie(g, "home", use_override=False)
                sim = model.simulate_game(aa, ha, ag, hg)
                with st.expander(live_label(g, box, status), expanded=live_open):
                    st_html(live_card(g, box, status, sim))
                    if box and status["kind"] != "sched":
                        actual = parse_box_players(box)
                        a_df = model.project_skaters(aa, ha, False, hg["adj60"], apply_b2b=False, player_ids=dressed_ids(box, "awayTeam"))
                        h_df = model.project_skaters(ha, aa, True, ag["adj60"], apply_b2b=False, player_ids=dressed_ids(box, "homeTeam"))
                        st_html(skater_compare_card(aa, a_df, ha, h_df, actual))
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not load {aa} @ {ha}: {exc}")

    live_tracker()

with tab_rec:
    # -- Daily top-50 player review ---------------------------------------
    st_html("<div class='section'>📋 Daily top 50: projected points vs. actual</div>")
    last_done = max(min(slate_date, est_now.date()) - datetime.timedelta(days=1), SEASON_START)
    review_d = st.date_input("Slate to review", value=last_done, max_value=last_done,
                             min_value=SEASON_START, key="review_date")
    try:
        top50 = daily_top_players(SEASON, review_d.isoformat())
    except Exception as exc:  # noqa: BLE001
        top50 = pd.DataFrame()
        st.error(f"Could not grade {review_d}: {exc}")
    if top50.empty:
        st.info(f"No completed games found for {review_d}.")
    else:
        hit = (top50["act_pts"] >= 1)
        c1, c2 = st.columns(2)
        c1.metric("1+ point hit rate", f"{hit.mean()*100:.0f}%  ({int(hit.sum())}/{len(top50)})",
                  f"model expected {top50['pt_p'].mean():.0f}%", delta_color="off")
        c2.metric("Points: proj vs actual", f"{top50['pts'].sum():.1f} vs {int(top50['act_pts'].sum())}",
                  f"{top50['act_pts'].sum() - top50['pts'].sum():+.1f} total", delta_color="off")
        c3, c4 = st.columns(2)
        scored = (top50["act_g"] >= 1)
        c3.metric("Scored a goal", f"{scored.mean()*100:.0f}%", f"model ATG {top50['atg'].mean():.0f}%", delta_color="off")
        c4.metric("SOG: proj vs actual", f"{top50['sog'].mean():.2f} vs {top50['act_sog'].mean():.2f}", "per player", delta_color="off")
        rows = "".join(
            f"<tr><td>{i}</td><td><b>{r['name']}</b> <span class='card-sub'>{r['team']} vs {r['opp']}</span></td>"
            f"<td>{r['pts']:.2f} ({r['pt_p']:.0f}%)</td>"
            f"<td><b>{int(r['act_pts'])}</b> ({int(r['act_g'])}G {int(r['act_a'])}A) {'✅' if r['act_pts'] >= 1 else '❌'}</td>"
            f"<td>{r['sog']:.1f} vs <b>{int(r['act_sog'])}</b></td></tr>"
            for i, (_, r) in enumerate(top50.iterrows(), start=1)
        )
        with st.expander(f"All {len(top50)} players for {review_d:%b %-d}"):
            st_html(f"""
            <div class='card'><div class='tbl-wrap'><table class='tbl'>
              <tr><th>#</th><th>Player</th><th>Proj pts (1+%)</th><th>Actual</th><th>SOG</th></tr>{rows}
            </table></div></div>
            """)

    if st.button("Grade the last 7 slates", width="stretch", key="grade_week"):
        hist = ""
        d = last_done
        graded = 0
        while graded < 7 and d >= SEASON_START:
            try:
                day_df = daily_top_players(SEASON, d.isoformat())
            except Exception:  # noqa: BLE001
                day_df = pd.DataFrame()
            if not day_df.empty:
                h = (day_df["act_pts"] >= 1)
                hist += (f"<tr><td>{d:%a %b %-d}</td><td>{len(day_df)}</td>"
                         f"<td><b>{h.mean()*100:.0f}%</b></td><td>{day_df['pt_p'].mean():.0f}%</td>"
                         f"<td>{day_df['act_pts'].sum() - day_df['pts'].sum():+.1f}</td></tr>")
                graded += 1
            d -= datetime.timedelta(days=1)
        if hist:
            st_html(f"""
            <div class='card'><div class='section'>Top-50 hit rate by slate</div>
            <div class='tbl-wrap'><table class='tbl'>
              <tr><th>Slate</th><th>Players</th><th>1+ pt hit</th><th>Model exp.</th><th>Pts vs proj</th></tr>{hist}
            </table></div></div>
            """)
        else:
            st.info("No completed slates found in that range.")
    st.caption(
        "Re-projected with today's season-to-date stats, the NHL-confirmed starting goalies, and only players who "
        "dressed, so this runs a bit optimistic versus true pregame picks."
    )
    st.divider()
    st_html("<div class='section'>🏆 Season: team winner picks</div>")

with tab_rec:
    end_d = min(slate_date, est_now.date()) - datetime.timedelta(days=1)
    if end_d < SEASON_START:
        st.info("No completed games yet this season.")
    else:
        try:
            rec = season_record(SEASON, SEASON_START.isoformat(), end_d.isoformat())
        except Exception as exc:  # noqa: BLE001
            rec = pd.DataFrame()
            st.error(f"Could not build the season record: {exc}")
        if rec.empty:
            st.info("No completed games found.")
        else:
            wins = int(rec["hit"].sum())
            c1, c2 = st.columns(2)
            c1.metric("Winner picks", f"{wins}-{len(rec) - wins}", f"{rec['hit'].mean()*100:.1f}% hit rate", delta_color="off")
            c2.metric("Total goals error", f"{rec['tot_err'].abs().mean():.2f} per game",
                      f"bias {rec['tot_err'].mean():+.2f} (actual − proj)", delta_color="off")
            c3, c4 = st.columns(2)
            c3.metric("O/U 5.5 calls", f"{rec['o55_hit'].mean()*100:.1f}%", f"{len(rec)} games", delta_color="off")
            team_err = pd.concat([(rec["a_goals"] - rec["proj_a"]).abs(), (rec["h_goals"] - rec["proj_h"]).abs()]).mean()
            c4.metric("Team goals error", f"{team_err:.2f} per team", f"avg proj total {rec['proj_tot'].mean():.2f}", delta_color="off")

            bins = [(0.5, 0.55, "50–55%"), (0.55, 0.60, "55–60%"), (0.60, 0.65, "60–65%"), (0.65, 1.01, "65%+")]
            conf_rows = ""
            for lo, hi, lab in bins:
                sub = rec[(rec["pick_p"] >= lo) & (rec["pick_p"] < hi)]
                if len(sub):
                    conf_rows += f"<tr><td>{lab}</td><td>{len(sub)}</td><td>{sub['pick_p'].mean()*100:.0f}%</td><td><b>{sub['hit'].mean()*100:.0f}%</b></td></tr>"
            last7 = rec[rec["date"] > (end_d - datetime.timedelta(days=7)).isoformat()]
            l7 = f"{int(last7['hit'].sum())}-{len(last7) - int(last7['hit'].sum())}" if len(last7) else "—"
            recent = ""
            for _, r in rec.sort_values(["date"], ascending=False).head(20).iterrows():
                sfx = "" if r["last"] == "REG" else f" ({r['last']})"
                recent += (
                    f"<tr><td>{r['date'][5:]}</td><td>{r['away']} @ {r['home']}</td>"
                    f"<td>{r['proj_a']:.1f}–{r['proj_h']:.1f}</td><td><b>{r['a_goals']}–{r['h_goals']}</b>{sfx}</td>"
                    f"<td>{r['pick']} {'✅' if r['hit'] else '❌'}</td></tr>"
                )
            st_html(
                f"""
                <div class='card'>
                  <div class='section'>By confidence · last 7 days {l7}</div>
                  <div class='tbl-wrap'><table class='tbl'>
                    <tr><th>Pick win %</th><th>Games</th><th>Avg model %</th><th>Hit rate</th></tr>{conf_rows}
                  </table></div>
                </div>
                <div class='card'>
                  <div class='section'>Recent results</div>
                  <div class='tbl-wrap'><table class='tbl'>
                    <tr><th>Date</th><th>Game</th><th>Proj</th><th>Final</th><th>Pick</th></tr>{recent}
                  </table></div>
                </div>
                """)
            st.caption(
                "Graded with today's season-to-date MoneyPuck stats and each team's #1 goalie, so earlier games are "
                "partly in-sample and this record runs a bit optimistic. Shootout wins count for the winner."
            )

st.caption(
    f"MoneyPuck {SEASON} season summary · NHL API schedule & box scores · "
    f"{N_SIMS:,} Gamma-Poisson sims (shape {GAMMA_SHAPE:g}) · B2B {B2B_OFF:.2f} off / {B2B_DEF:.2f} def"
)
