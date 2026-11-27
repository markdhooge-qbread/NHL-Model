"""Download and cache the free data the backtest needs.

  * MoneyPuck team game-by-game file (all seasons, one CSV, ~130 MB):
      https://moneypuck.com/moneypuck/playerData/careers/gameByGame/all_teams.csv
  * MoneyPuck goalie game-by-game files, one per goalie (ids come from each season's goalies.csv):
      https://moneypuck.com/moneypuck/playerData/careers/gameByGame/regular/goalies/{playerId}.csv
  * NHL API daily scores (final score, OT/SO, goals with period and time):
      https://api-web.nhle.com/v1/score/{YYYY-MM-DD}

Everything is cached under backtests/.cache (git-ignored), so a rerun only fetches what's new.
Past days are cached for good; today and later are always refetched.
"""
import datetime
import io
import json
import os
import time

import pandas as pd
import requests

MP = "https://moneypuck.com/moneypuck/playerData"
NHL_API = "https://api-web.nhle.com/v1"
HEADERS = {"User-Agent": "Mozilla/5.0"}  # same as the app; MoneyPuck turns away unusual agents
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")

TEAM_COLS = ["season", "gameId", "playerTeam", "opposingTeam", "home_or_away", "gameDate", "situation", "iceTime",
             "xGoalsFor", "xGoalsAgainst", "scoreVenueAdjustedxGoalsFor", "scoreVenueAdjustedxGoalsAgainst",
             "shotAttemptsFor", "shotAttemptsAgainst", "highDangerShotsFor", "highDangerShotsAgainst",
             "goalsFor", "goalsAgainst", "penaltiesFor", "penaltiesAgainst", "playoffGame"]
GOALIE_COLS = ["playerId", "season", "name", "gameId", "playerTeam", "gameDate", "situation", "icetime",
               "xGoals", "goals"]


def _get(url, tries=4, timeout=120):
    for i in range(tries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r
        except requests.RequestException:
            if i == tries - 1:
                raise
            time.sleep(2 ** i)
    return None


def _path(*parts):
    p = os.path.join(CACHE, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def team_games(seasons, refresh=False):
    """Team-level game rows (regular season) for the given seasons, MoneyPuck team codes (T.B, N.J ...)."""
    pkl = _path("mp", "team_games.pkl")
    df = None
    if os.path.exists(pkl) and not refresh:
        df = pd.read_pickle(pkl)
        if not set(seasons) <= set(df["season"].unique()):
            df = None
    if df is None:
        r = _get(f"{MP}/careers/gameByGame/all_teams.csv", timeout=600)
        head = r.content[:300].decode("utf-8", "ignore")
        if "season" not in head.split("\n")[0]:
            raise RuntimeError(f"MoneyPuck all_teams.csv didn't come back as the CSV (got: {head[:120]!r})")
        df = pd.read_csv(io.BytesIO(r.content), usecols=lambda c: c in TEAM_COLS)
        df = df[df["season"] >= min(seasons)]
        df.to_pickle(pkl)
    df = df[df["season"].isin(seasons)]
    if "playoffGame" in df.columns:
        df = df[df["playoffGame"] == 0]
    return df.reset_index(drop=True)


def goalie_ids(season):
    r = _get(f"{MP}/seasonSummary/{season}/regular/goalies.csv")
    if r is None:
        return []
    return sorted(pd.read_csv(io.StringIO(r.text))["playerId"].unique().tolist())


def goalie_games(seasons, refresh=False):
    """Goalie game rows (situation 'all') for the seasons. refresh=True refetches files older than 6 hours
    (for the season in progress)."""
    frames = []
    ids = sorted({pid for s in seasons for pid in goalie_ids(s)})
    for pid in ids:
        f = _path("mp", "goalies", f"{pid}.csv")
        stale = refresh and os.path.exists(f) and time.time() - os.path.getmtime(f) > 6 * 3600
        if not os.path.exists(f) or stale:
            r = _get(f"{MP}/careers/gameByGame/regular/goalies/{pid}.csv")
            if r is None:
                continue
            with open(f, "w") as fh:
                fh.write(r.text)
            time.sleep(0.2)
        g = pd.read_csv(f, usecols=lambda c: c in GOALIE_COLS)
        frames.append(g[(g["situation"] == "all") & g["season"].isin(seasons)])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=GOALIE_COLS)


def _team(t):
    if isinstance(t, dict):
        return t.get("default") or t.get("abbrev")
    return t


def parse_score_day(js, date_str):
    """Regular-season finals from an NHL /score/{date} payload, with 1st-period goals and first-goal time."""
    out = []
    for g in js.get("games", []):
        if g.get("gameType") != 2 or g.get("gameState") not in ("FINAL", "OFF"):
            continue
        away, home = g["awayTeam"]["abbrev"], g["homeTeam"]["abbrev"]
        rec = {"id": g["id"], "date": g.get("gameDate", date_str), "away": away, "home": home,
               "a_goals": int(g["awayTeam"].get("score") or 0), "h_goals": int(g["homeTeam"].get("score") or 0),
               "last": (g.get("gameOutcome") or {}).get("lastPeriodType", "REG"),
               "p1_a": None, "p1_h": None, "first_goal_p1_sec": None}
        goals = g.get("goals")
        if isinstance(goals, list):
            p1 = [x for x in goals if (x.get("periodDescriptor") or {}).get("number", x.get("period")) == 1]
            rec["p1_a"] = sum(_team(x.get("teamAbbrev")) == away for x in p1)
            rec["p1_h"] = sum(_team(x.get("teamAbbrev")) == home for x in p1)
            if rec["p1_a"] + rec["p1_h"] != len(p1):  # couldn't tell whose goals they were
                rec["p1_a"] = rec["p1_h"] = None
            secs = []
            for x in p1:
                mm, _, ss = str(x.get("timeInPeriod", "")).partition(":")
                if mm.isdigit() and ss.isdigit():
                    secs.append(int(mm) * 60 + int(ss))
            rec["first_goal_p1_sec"] = min(secs) if secs else (None if p1 else 9999)
        out.append(rec)
    return out


def season_dates(season, today=None):
    today = today or datetime.date.today()
    d, end = datetime.date(season, 10, 1), min(datetime.date(season + 1, 4, 30), today - datetime.timedelta(days=1))
    while d <= end:
        yield d
        d += datetime.timedelta(days=1)


def nhl_results(season, today=None):
    """Every completed regular-season game of a season (through yesterday), one row per game."""
    rows = []
    today = today or datetime.date.today()
    for d in season_dates(season, today):
        ds = d.isoformat()
        f = _path("nhl", "score", f"{ds}.json")
        if os.path.exists(f):
            with open(f) as fh:
                js = json.load(fh)
        else:
            r = _get(f"{NHL_API}/score/{ds}")
            js = r.json() if r is not None else {"games": []}
            if d < today - datetime.timedelta(days=1):  # yesterday may still be updating
                with open(f, "w") as fh:
                    json.dump({"games": js.get("games", [])}, fh)
            time.sleep(0.1)
        rows += parse_score_day(js, ds)
    df = pd.DataFrame(rows)
    return df.drop_duplicates("id").sort_values(["date", "id"]).reset_index(drop=True) if len(df) else df
