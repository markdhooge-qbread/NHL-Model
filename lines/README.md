# DraftKings lines

The app reads `lines/YYYY-MM-DD.csv` (Eastern date of the slate) and compares each projection to the DraftKings price. Picks with no line here show **NA**.

If an `ODDS_API_KEY` is set in Streamlit secrets, lines from The Odds API (DraftKings) are loaded too. Those win where both sources have the same line. Manual edits on the Bets tab win over both.

## Columns

| column | meaning |
|---|---|
| `game` | `AWAY@HOME` using NHL abbreviations, e.g. `BOS@TOR` |
| `market` | `ML`, `TOTAL`, `SOG`, `PTS`, or `GOAL` (anytime goal) |
| `player` | `ML`: the team abbreviation the odds are for. Player markets: the player's full name. Blank for `TOTAL` |
| `line` | Total or prop line, e.g. `6.5`, `2.5`, `0.5`. Blank for `ML`. For `GOAL` it defaults to `0.5` |
| `over_odds` | American odds for the over (or the team for `ML`, or "yes" for `GOAL`), e.g. `-130`, `+115` |
| `under_odds` | American odds for the under. Optional. When both sides are given, the edge is figured without the vig |

See `TEMPLATE.csv` for an example of each market.
