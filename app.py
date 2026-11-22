"""NHL Game Flow & Player Prop Model (Streamlit).

Streamlit port of notebooks/player_model.ipynb:
  1. Slate Sim   - 10,000-iteration Gamma-Poisson Monte Carlo per game
  2. Player Props - SOG, anytime goal %, and 1+ point % for top skaters
  3. Backtest    - model projection vs. live/final NHL API box score
"""

import datetime
import io
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

st.set_page_config(page_title="NHL Game Flow Model", page_icon="🏒", layout="centered")

st.markdown(
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
      .backup {background:#991b1b;color:#fecaca;}
      .track {background:#7f1d1d;color:#fca5a5;}
      .lowev {background:#1e3a8a;color:#93c5fd;}
      .bal {background:#334155;color:#cbd5e1;}
      .section {font-size:13px;font-weight:800;color:#38bdf8;margin:4px 0;}
      .meta {color:#94a3b8;font-size:12px;margin-bottom:8px;}
    </style>
    """,
    unsafe_allow_html=True,
)

B2B_BADGE = " <span class='badge b2b'>B2B</span>"
BACKUP_BADGE = " <span class='badge backup'>BACKUP</span>"


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


@st.cache_data(ttl=60, show_spinner=False)
def load_boxscore(game_id):
    resp = requests.get(f"{NHL_API}/gamecenter/{game_id}/boxscore", timeout=15)
    resp.raise_for_status()
    return resp.json()


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

    def get_goalie(self, abbr, goalie_name=None):
        """#1 starter (most GP) regresses to 0.00 GSAx/60 over 15 games;
        backups regress to -0.18 over 20 games, then a -0.10 drag."""
        team_g = self.team_goalies(abbr)
        if team_g.empty:
            return {"name": "Unknown", "label": "Unknown", "adj60": 0.0, "is_backup": False}
        starter = team_g.iloc[0]["name"]
        target = goalie_name if goalie_name in team_g["name"].values else starter
        g = team_g[team_g["name"] == target].iloc[0]
        gp = int(g["games_played"])
        toi = max(g["icetime"] / 3600.0, 0.1)
        raw60 = (g["xGoals"] - g["goals"]) / toi
        is_backup = target != starter
        if is_backup:
            adj60 = ((raw60 * toi) + (-0.18 * 20.0)) / (toi + 20.0) - 0.10
        else:
            adj60 = (raw60 * toi) / (toi + 15.0)
        return {"name": target, "label": f"{target} ({gp} GP)", "adj60": adj60, "is_backup": is_backup}

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
    def project_skaters(self, team_abbr, opp_abbr, is_home, opp_gsax60, apply_b2b=True):
        opp = self.get_team(opp_abbr)
        is_b2b = apply_b2b and team_abbr in self.b2b
        opp_b2b = apply_b2b and opp_abbr in self.b2b
        venue = 1.035 if is_home else 0.965

        pace_mult = (opp["ca60"] / self.LG_CA60) * venue * (0.95 if is_b2b else 1.0)
        xg_mult = (opp["xga60"] / self.LG_XG5) * venue * (B2B_OFF if is_b2b else 1.0) * (B2B_DEF if opp_b2b else 1.0)
        pp_mult = opp["pk_xga60"] / self.LG_XGPP
        goalie_mult = max(1.0 - (opp_gsax60 / 3.0), 0.75)

        sk = self.skaters_df
        pool = sk[(sk["team"] == self.to_mp(team_abbr)) & (sk["games_played"] >= 5)]
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
                "playerId": pid, "name": p["name"], "pos": p["position"],
                "toi": toi_pg, "pp": pp_pg, "sog": sog, "xg": xg,
                "atg": (1 - np.exp(-xg)) * 100, "pts": pts, "pt_p": (1 - np.exp(-pts)) * 100,
            })
        cols = ["playerId", "name", "pos", "toi", "pp", "sog", "xg", "atg", "pts", "pt_p"]
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
    ag_b = BACKUP_BADGE if ag["is_backup"] else ""
    hg_b = BACKUP_BADGE if hg["is_backup"] else ""
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
        f"<tr><td><b>{r['name']}</b> <span class='card-sub'>({r['pos']})</span></td>"
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
    for _, r in df.head(6).iterrows():
        act = actual.get(r["playerId"], {"g": 0, "a": 0, "pts": 0, "sog": 0})
        html += (
            f"<tr><td><b>{r['name']}</b></td>"
            f"<td>{r['sog']:.1f} vs <b>{act['sog']}</b></td>"
            f"<td>{r['pts']:.2f} vs <b>{act['pts']}</b> ({act['g']}G {act['a']}A)</td></tr>"
        )
    return html


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

st.markdown(
    f"<div class='meta'>📅 <b>{date_str}</b> ({len(games)} games) &nbsp;|&nbsp; "
    f"Auto-B2B: <b>{', '.join(sorted(b2b_teams)) if b2b_teams else 'None'}</b></div>",
    unsafe_allow_html=True,
)

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

tab_slate, tab_props, tab_bt = st.tabs(["Slate", "Player Props", "Backtest"])

with tab_slate:
    cards = []
    for g in games:
        try:
            ag = model.get_goalie(g["away"], overrides.get(g["away"]))
            hg = model.get_goalie(g["home"], overrides.get(g["home"]))
            sim = model.simulate_game(g["away"], g["home"], ag, hg)
            cards.append(slate_card(g, sim, ag, hg))
        except Exception as exc:  # noqa: BLE001
            cards.append(f"<div class='card'>{g['away']} @ {g['home']}: could not simulate ({exc})</div>")
    st.markdown("".join(cards), unsafe_allow_html=True)

with tab_props:
    labels = [f"{g['away']} @ {g['home']} ({g['time']})" for g in games]
    pick = st.selectbox("Matchup", labels, key="props_game")
    g = games[labels.index(pick)]
    aa, ha = g["away"], g["home"]
    n_show = st.slider("Skaters per team", 4, 14, 8)
    try:
        ag = model.get_goalie(aa, overrides.get(aa))
        hg = model.get_goalie(ha, overrides.get(ha))
        df_a = model.project_skaters(aa, ha, False, hg["adj60"]).head(n_show)
        df_h = model.project_skaters(ha, aa, True, ag["adj60"]).head(n_show)
        hg_b = BACKUP_BADGE if hg["is_backup"] else ""
        ag_b = BACKUP_BADGE if ag["is_backup"] else ""
        a_b = B2B_BADGE if aa in b2b_teams else ""
        h_b = B2B_BADGE if ha in b2b_teams else ""
        st.markdown(
            props_card(f"{aa}{a_b} Skaters (vs {hg['name']}{hg_b})", df_a)
            + props_card(f"{ha}{h_b} Skaters (vs {ag['name']}{ag_b})", df_h),
            unsafe_allow_html=True,
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not project {aa} @ {ha}: {exc}")

with tab_bt:
    bt_labels = [f"{g['away']} @ {g['home']} [{g['state']}]" for g in games]
    c1, c2 = st.columns([3, 1])
    bt_pick = c1.selectbox("Test game", bt_labels, key="bt_game")
    if c2.button("Refresh box score", use_container_width=True):
        load_boxscore.clear()
    g = games[bt_labels.index(bt_pick)]
    aa, ha, gid = g["away"], g["home"], g["id"]
    try:
        box = load_boxscore(gid)
        state = box.get("gameState", g["state"])
        act_a = box.get("awayTeam", {}).get("score", 0) or 0
        act_h = box.get("homeTeam", {}).get("score", 0) or 0
        act_a_sog = box.get("awayTeam", {}).get("sog", 0) or 0
        act_h_sog = box.get("homeTeam", {}).get("sog", 0) or 0

        # Backtest uses each team's #1 starter (no overrides) and no player B2B, as in the notebook.
        ag = model.get_goalie(aa)
        hg = model.get_goalie(ha)
        sim = model.simulate_game(aa, ha, ag, hg)
        act_tot = act_a + act_h
        pred_w = aa if sim["proj_a"] > sim["proj_h"] else ha
        act_w = aa if act_a > act_h else (ha if act_h > act_a else "TIE")
        if state not in ("FINAL", "OFF"):
            hit = "⏳ LIVE" if state in ("LIVE", "CRIT") else "🕒 NOT STARTED"
        else:
            hit = "✅ HIT" if pred_w == act_w else "❌ MISS"

        actual = parse_box_players(box)
        a_rows = backtest_rows(model.project_skaters(aa, ha, False, hg["adj60"], apply_b2b=False), actual)
        h_rows = backtest_rows(model.project_skaters(ha, aa, True, ag["adj60"], apply_b2b=False), actual)
        sog_diff = act_h_sog - act_a_sog

        st.markdown(
            f"""
            <div class='card'>
              <div class='card-head'>
                <div class='card-title'>BACKTEST: {aa} @ {ha} <span class='card-sub'>({state})</span></div>
                <div class='badge bal' style='font-size:12px;'>Winner call: {hit}</div>
              </div>
              <div class='tbl-wrap'><table class='tbl' style='font-size:13px;margin-bottom:10px;'>
                <tr><th>Metric</th><th>Proj</th><th>Actual</th><th>Diff</th></tr>
                <tr><td><b>{aa} Goals</b></td><td>{sim['proj_a']:.2f}</td><td><b>{act_a}</b></td><td>{act_a - sim['proj_a']:+.2f}</td></tr>
                <tr><td><b>{ha} Goals</b></td><td>{sim['proj_h']:.2f}</td><td><b>{act_h}</b></td><td>{act_h - sim['proj_h']:+.2f}</td></tr>
                <tr><td><b>Total Goals</b></td><td>{sim['proj_tot']:.2f}</td><td><b>{act_tot}</b></td><td>{act_tot - sim['proj_tot']:+.2f}</td></tr>
                <tr><td><b>Shots on Goal</b></td><td>Pace {sim['a']['ca60']:.0f} / {sim['h']['ca60']:.0f}</td><td><b>{act_a_sog} — {act_h_sog}</b></td><td>{ha} {sog_diff:+d}</td></tr>
              </table></div>
              <div class='section'>{aa} Top Skaters (Proj vs Actual)</div>
              <div class='tbl-wrap'><table class='tbl' style='margin-bottom:10px;'>
                <tr><th>Player</th><th>SOG</th><th>Points</th></tr>{a_rows}
              </table></div>
              <div class='section'>{ha} Top Skaters (Proj vs Actual)</div>
              <div class='tbl-wrap'><table class='tbl'>
                <tr><th>Player</th><th>SOG</th><th>Points</th></tr>{h_rows}
              </table></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not run backtest for {aa} @ {ha}: {exc}")

st.caption(
    f"MoneyPuck {SEASON} season summary · NHL API schedule & box scores · "
    f"{N_SIMS:,} Gamma-Poisson sims (shape {GAMMA_SHAPE:g}) · B2B {B2B_OFF:.2f} off / {B2B_DEF:.2f} def"
)
