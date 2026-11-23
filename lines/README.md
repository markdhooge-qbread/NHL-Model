# DraftKings lines

The app reads `lines/YYYY-MM-DD.csv` (Eastern date of the slate) and compares each projection to the DraftKings price. Picks with no line here show **NA**.

If an `ODDS_API_KEY` is set in Streamlit secrets, lines from The Odds API (DraftKings) are loaded too. Those win where both sources have the same line. Manual edits on the Bets tab win over both.

## Columns

| column | meaning |
|---|---|
| `game` | `AWAY@HOME` using NHL abbreviations, e.g. `BOS@TOR` |
| `market` | `ML`, `TOTAL`, `SOG`, `PTS`, `GOAL` (anytime goal), `P1TOTAL` (1st-period total), `P1ML` (1st-period 3-way), or `F10` (goal in the first 10 minutes) |
| `player` | `ML` / `P1ML`: the team abbreviation the odds are for, or `TIE` for the 1st-period tie. Player markets: the player's full name. Blank for `TOTAL`, `P1TOTAL`, `F10` |
| `line` | Total or prop line, e.g. `6.5`, `2.5`, `0.5`. Blank for `ML`. For `GOAL` it defaults to `0.5` |
| `over_odds` | American odds for the over (or the team for `ML` / `P1ML`, or "yes" for `GOAL` and `F10`), e.g. `-130`, `+115` |
| `under_odds` | American odds for the under ("no" for `F10`). Optional. When both sides are given, the edge is figured without the vig |

See `TEMPLATE.csv` for an example of each market.
