# DraftKings lines

Lines now load automatically from two free feeds (no API key, no credits):

- **Game lines** (moneyline, total, puck line) from ESPN's public NHL scoreboard, which carries DraftKings prices.
- **Player props** (SOG, points, anytime goal) from DraftKings' own public sportsbook feed.

Neither is an official API, so either can break or be blocked without warning. The app refetches at most hourly, and only for games that haven't started; once a game starts, its last pregame lines are kept so live odds never replace them. Each day's feed lines are saved to `lines/auto/YYYY-MM-DD.csv` on the `picks-data` branch (needs `GITHUB_TOKEN`), so the Record tab can grade them later.

When a feed fails, the 📈 DraftKings lines box says **screenshots needed**. Then transcribe DK screenshots into `lines/YYYY-MM-DD.csv` (Eastern date of the slate) as before. Screenshot lines win where both have the same line, so they fill or fix feed gaps. Picks with no line anywhere show **NA**.

If an `ODDS_API_KEY` is set in Streamlit secrets, The Odds API is loaded too and wins over both. Manual edits on the Bets tab win over everything.

## Columns

| column | meaning |
|---|---|
| `game` | `AWAY@HOME` using NHL abbreviations, e.g. `BOS@TOR` |
| `market` | `ML`, `TOTAL`, `SOG`, `PTS`, `GOAL` (anytime goal), `P1TOTAL` (1st-period total), `P1ML` (1st-period 3-way), `P1PL` (1st-period puck line, line ±0.5, team in `player`), `P1TNB` (1st-period tie no bet, team in `player`), or `F10` (goal in the first 10 minutes) |
| `player` | `ML` / `P1ML`: the team abbreviation the odds are for, or `TIE` for the 1st-period tie. Player markets: the player's full name. Blank for `TOTAL`, `P1TOTAL`, `F10` |
| `line` | Total or prop line, e.g. `6.5`, `2.5`, `0.5`. Blank for `ML`. For `GOAL` it defaults to `0.5` |
| `over_odds` | American odds for the over (or the team for `ML` / `P1ML`, or "yes" for `GOAL` and `F10`), e.g. `-130`, `+115` |
| `under_odds` | American odds for the under ("no" for `F10`). Optional. When both sides are given, the edge is figured without the vig |

See `TEMPLATE.csv` for an example of each market.
