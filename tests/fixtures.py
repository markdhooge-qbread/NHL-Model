"""Small synthetic MoneyPuck-shaped tables for tests (the real CSVs aren't reachable from CI sandboxes)."""
import numpy as np
import pandas as pd

TEAMS = ["TOR", "BOS", "T.B", "EDM", "DAL", "STL"]
SITS = ("all", "5on5", "5on4", "4on5", "other")


def teams_df(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for t in TEAMS:
        gp = 20
        for s in SITS:
            hrs = {"all": 60 * gp / 60, "5on5": 48 * gp / 60, "5on4": 5 * gp / 60, "4on5": 5 * gp / 60, "other": 2 * gp / 60}[s]
            xgf = {"all": 3.0, "5on5": 2.4, "5on4": 7.0, "4on5": 0.8, "other": 3.0}[s] * hrs * rng.uniform(0.85, 1.15)
            xga = {"all": 3.0, "5on5": 2.4, "5on4": 0.8, "4on5": 7.0, "other": 3.0}[s] * hrs * rng.uniform(0.85, 1.15)
            rows.append({
                "team": t, "situation": s, "games_played": gp, "iceTime": hrs * 3600,
                "xGoalsFor": xgf, "xGoalsAgainst": xga,
                "scoreVenueAdjustedxGoalsFor": xgf * 0.99, "scoreVenueAdjustedxGoalsAgainst": xga * 1.01,
                "goalsFor": xgf * rng.uniform(0.8, 1.2), "goalsAgainst": xga * rng.uniform(0.8, 1.2),
                "shotAttemptsFor": 57 * hrs * rng.uniform(0.9, 1.1), "shotAttemptsAgainst": 57 * hrs * rng.uniform(0.9, 1.1),
                "highDangerShotsFor": 12 * hrs, "highDangerShotsAgainst": 12 * hrs * rng.uniform(0.8, 1.2),
                "penaltiesFor": 3.3 * hrs * rng.uniform(0.8, 1.2), "penaltiesAgainst": 3.3 * hrs * rng.uniform(0.8, 1.2),
            })
    return pd.DataFrame(rows)


def goalies_df(seed=0):
    rng = np.random.default_rng(seed + 1)
    rows = []
    pid = 8470000
    for t in TEAMS:
        for gp, nm in ((16, "Starter"), (4, "Backup")):
            pid += 1
            hrs = gp * 1.0
            xg = 2.9 * hrs
            rows.append({"playerId": pid, "name": f"{t} {nm}", "team": t, "situation": "all", "games_played": gp,
                         "icetime": hrs * 3600, "xGoals": xg, "goals": xg + rng.normal(0, 3)})
    return pd.DataFrame(rows)


def skaters_df(seed=0):
    rng = np.random.default_rng(seed + 2)
    rows = []
    pid = 8480000
    for t in TEAMS:
        for i in range(6):
            pid += 1
            gp = 20
            for s, share in (("all", 1.0), ("5on5", 0.82), ("5on4", 0.12)):
                hrs = 18 * gp / 60 * share
                rows.append({"playerId": pid, "name": f"{t} Skater{i}", "team": t, "position": "C" if i < 4 else "D",
                             "situation": s, "games_played": gp, "icetime": hrs * 3600,
                             "I_F_shotsOnGoal": 8 * hrs * rng.uniform(0.6, 1.4), "I_F_xGoals": 0.9 * hrs * rng.uniform(0.5, 1.5),
                             "I_F_primaryAssists": 0.8 * hrs, "I_F_secondaryAssists": 0.5 * hrs})
    return pd.DataFrame(rows)


# --- Game-by-game data in the MoneyPuck / NHL API shapes the backtest reads -------------------------------
NHL_TEAMS = ["TOR", "BOS", "TBL", "EDM", "DAL", "STL"]
MP_CODE = {"TBL": "T.B"}


def synthetic_season(season=2024, days=40, seed=0):
    """(team_games, goalie_games, results) for a fake season where every team plays every other day."""
    import datetime
    rng = np.random.default_rng(seed)
    strength = {t: rng.normal(0, 0.3) for t in NHL_TEAMS}
    tg, gg, res = [], [], []
    gid = season * 1000000 + 20000
    d0 = datetime.date(season, 10, 8)
    for day in range(days):
        d = d0 + datetime.timedelta(days=day)
        order = list(rng.permutation(NHL_TEAMS)) if day % 2 == 0 else []
        for i in range(0, len(order), 2):
            gid += 1
            away, home = order[i], order[i + 1]
            lam = {away: 2.9 + strength[away] - strength[home] * 0.5, home: 3.1 + strength[home] - strength[away] * 0.5}
            goals = {t: int(rng.poisson(max(lam[t], 0.5))) for t in (away, home)}
            last = "REG"
            if goals[away] == goals[home]:
                goals[home if rng.random() < 0.5 else away] += 1
                last = "OT"
            p1 = {t: int(rng.binomial(goals[t], 0.3)) for t in (away, home)}
            res.append({"id": gid, "date": d.isoformat(), "away": away, "home": home, "a_goals": goals[away],
                        "h_goals": goals[home], "last": last, "p1_a": p1[away], "p1_h": p1[home],
                        "first_goal_p1_sec": int(rng.integers(0, 1200)) if p1[away] + p1[home] else 9999})
            for t, o, ha in ((away, home, "AWAY"), (home, away, "HOME")):
                for sit, hrs, xgr in (("all", 1.0, 3.0), ("5on5", 0.8, 2.4), ("5on4", 0.09, 7.0), ("4on5", 0.09, 0.8), ("other", 0.02, 3.0)):
                    xgf = max(lam[t] / 3.0 * xgr * hrs * rng.uniform(0.7, 1.3), 0.01)
                    xga = max(lam[o] / 3.0 * xgr * hrs * rng.uniform(0.7, 1.3), 0.01) if sit not in ("5on4", "4on5") else \
                        max((7.0 if sit == "4on5" else 0.8) * hrs, 0.01)
                    tg.append({"season": season, "gameId": gid, "playerTeam": MP_CODE.get(t, t), "opposingTeam": MP_CODE.get(o, o),
                               "home_or_away": ha, "gameDate": int(d.strftime("%Y%m%d")), "situation": sit,
                               "iceTime": hrs * 3600, "xGoalsFor": xgf, "xGoalsAgainst": xga,
                               "scoreVenueAdjustedxGoalsFor": xgf, "scoreVenueAdjustedxGoalsAgainst": xga,
                               "shotAttemptsFor": 57 * hrs, "shotAttemptsAgainst": 57 * hrs,
                               "highDangerShotsFor": 12 * hrs, "highDangerShotsAgainst": 12 * hrs,
                               "goalsFor": goals[t] * hrs, "goalsAgainst": goals[o] * hrs,
                               "penaltiesFor": 3.3 * hrs, "penaltiesAgainst": 3.3 * hrs, "playoffGame": 0})
                starter = 1 if rng.random() < 0.8 else 2
                pid = 8470000 + NHL_TEAMS.index(t) * 10 + starter
                gg.append({"playerId": pid, "season": season, "name": f"{t} G{starter}", "gameId": gid,
                           "playerTeam": MP_CODE.get(t, t), "gameDate": int(d.strftime("%Y%m%d")), "situation": "all",
                           "icetime": 3600.0, "xGoals": lam[o] * rng.uniform(0.8, 1.2), "goals": float(goals[o])})
    return pd.DataFrame(tg), pd.DataFrame(gg), pd.DataFrame(res)
