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


def slate_card(g, sim, ag, hg):
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
      </table></div>
    </div>
    """


def props_card(title, df):
    rows = "".join(
        f"<tr><td><b>{r['name']}</b> <span class='card-sub'>({r['pos']})</span>"
        f"{SMALL_SAMPLE_BADGE if r['gp'] < 5 else ''}</td>"
        f"<td>{r['toi']:.1f}m <span style='color:#38bdf8;font-size:10px;'>({r['pp']:.1f} PP)</span></td>"
        f"<td><b>{r['sog']:.2f}</b></td>"
        f"<td>{r['xg']:.2f} ({r['atg']:.0f}%)</td>"
        f"<td><b>{r['pts']:.2f}</b> ({r['pt_p']:.0f}%)</td></tr>"
        for _, r in df.iterrows()
    )
    return f"""
    <div class='card'>
      <div class='section'>{title}</div>
      <div class='tbl-wrap'><table class='tbl'>
        <tr><th>Player</th><th>TOI (PP)</th><th>SOG</th><th>xG (ATG%)</th><th>Pts (1+%)</th></tr>{rows}
      </table></div>
    </div>
    """


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
    if b1.button("Swap & re-sim", type="primary", use_container_width=True) and sel_goalie:
        overrides[sel_team] = sel_goalie
    if b2.button("Reset all", use_container_width=True):
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


tab_slate, tab_props, tab_live, tab_rec = st.tabs(["Slate", "Player Props", "Live Tracker", "Record"])

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
                st_html(slate_card(g, sim, ag, hg))
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
                index.append({**r.to_dict(), "team": team, "opp": opp, "opp_goalie": opp_g, "time": g["time"]})
    index.sort(key=lambda r: r["name"])
    labels_ix = [f"{r['name']} ({r['team']} vs {r['opp']})" for r in index]
    sel = st.selectbox("Player", labels_ix, index=None, placeholder="Type a name: anyone playing tonight", key="search_player")
    if sel:
        r = index[labels_ix.index(sel)]
        st_html(
            props_card(f"{r['name']} · {r['team']} vs {r['opp']} ({r['time']}) · opp G {r['opp_goalie']['name']}{goalie_badges(r['opp_goalie'])}",
                       pd.DataFrame([r])))
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

    st.divider()

    # -- Full matchup board ------------------------------------------------
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
            props_card(f"{aa}{a_b} Skaters (vs {hg['name']}{goalie_badges(hg)})", df_a)
            + props_card(f"{ha}{h_b} Skaters (vs {ag['name']}{goalie_badges(ag)})", df_h))
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not project {aa} @ {ha}: {exc}")

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
        if c2.button("Refresh", use_container_width=True, key="live_refresh"):
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
