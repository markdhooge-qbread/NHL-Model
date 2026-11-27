"""Season-to-date tables as of a date, in the same shape as MoneyPuck's season summary CSVs.

The live site feeds model_core.Model the season summary files (teams.csv, goalies.csv). Here the same
columns are rebuilt from game-by-game rows, keeping only games played before the date, so each
backtest game sees only what was known that morning: no later games, no hindsight goalie stats.
"""
import pandas as pd

TEAM_SUM = ["iceTime", "xGoalsFor", "xGoalsAgainst", "scoreVenueAdjustedxGoalsFor", "scoreVenueAdjustedxGoalsAgainst",
            "shotAttemptsFor", "shotAttemptsAgainst", "highDangerShotsFor", "highDangerShotsAgainst",
            "goalsFor", "goalsAgainst", "penaltiesFor", "penaltiesAgainst"]
SKATER_COLS = ["playerId", "name", "team", "position", "situation", "games_played", "icetime", "I_F_shotsOnGoal",
               "I_F_xGoals", "I_F_primaryAssists", "I_F_secondaryAssists"]


def date_int(date_str):
    return int(date_str.replace("-", ""))


def teams_asof(team_games, date_str):
    """Team season-to-date totals by situation for games before date_str (YYYY-MM-DD)."""
    prior = team_games[team_games["gameDate"] < date_int(date_str)]
    if prior.empty:
        return pd.DataFrame(columns=["team", "situation", "games_played"] + TEAM_SUM)
    cols = [c for c in TEAM_SUM if c in prior.columns]
    out = prior.groupby(["playerTeam", "situation"])[cols].sum()
    out["games_played"] = prior.groupby(["playerTeam", "situation"])["gameId"].nunique()
    return out.reset_index().rename(columns={"playerTeam": "team"})


def goalies_asof(goalie_games, date_str):
    """Goalie season-to-date totals (situation 'all') for games before date_str. A traded goalie gets a row per team."""
    prior = goalie_games[(goalie_games["gameDate"] < date_int(date_str)) & (goalie_games["icetime"] > 0)]
    cols = ["playerId", "name", "team", "situation", "games_played", "icetime", "xGoals", "goals"]
    if prior.empty:
        return pd.DataFrame(columns=cols)
    out = prior.groupby(["playerId", "name", "playerTeam"]).agg(
        games_played=("gameId", "nunique"), icetime=("icetime", "sum"), xGoals=("xGoals", "sum"),
        goals=("goals", "sum")).reset_index().rename(columns={"playerTeam": "team"})
    out["situation"] = "all"
    return out[cols]


def actual_starters(goalie_games):
    """(gameId, MoneyPuck team) -> playerId of the goalie with the most ice time in that game."""
    g = goalie_games[goalie_games["icetime"] > 0].sort_values("icetime", ascending=False)
    g = g.drop_duplicates(["gameId", "playerTeam"])
    return {(int(r.gameId), r.playerTeam): int(r.playerId) for r in g.itertuples()}


def empty_skaters():
    return pd.DataFrame(columns=SKATER_COLS)
