# Backtest

Scores the game simulation on past seasons, rebuilding every game from data that was available that
morning. It runs the same `model_core.Model` the site uses, so a backtest result is a result for the live code.

Results are in [`results/README.md`](results/README.md) (tables) and the CSVs next to it.

## How to rerun

**On GitHub (no setup):** Actions tab → **Backtest** → **Run workflow**. Leave the box empty for the full run
(about an hour), or type `--quick` for a 10-minute check. The results are attached to the run as
`backtest-results`; on any branch other than `main` they're also committed to `backtests/results/`.

**In Colab or on a laptop** (needs internet access to moneypuck.com and api-web.nhle.com):

```
git clone https://github.com/markdhooge-qbread/NHL-Model && cd NHL-Model
pip install numpy pandas requests
python -m backtests.run_backtest            # full run
python -m backtests.run_backtest --quick    # fast check
python -m backtests.run_backtest --tune 2021 2022 2023 --holdout 2024 --current 0
```

Downloads are cached in `backtests/.cache/` (not committed), so later runs only fetch new days.

## What it does

* **Data.** MoneyPuck's team game-by-game file and each goalie's game-by-game file, plus NHL API daily scores
  (final, OT/SO, and goal times for the 1st-period markets). All free.
* **No hindsight.** For each game date the team and goalie tables are rebuilt from games *before* that date
  (season to date), in the same columns as MoneyPuck's season summary. The goalie is the team's #1 by games
  played so far, which is what the site assumes until a starter is confirmed. A second run uses the goalie who
  actually started, to show what knowing the starter is worth.
* **Scores.** Log loss and Brier score for the home team winning; log loss for over/under 5.5 and 6.5 (there are
  no free historical closing lines, so these fixed lines stand in for them); log loss of the exact total; the
  forecast vs actual share of each total, shutouts and 7+ goal games; calibration by probability bucket; and
  1st-period goals and goal-in-first-10. Lower loss is better.
* **Baseline.** League average from the last season of games: the home win rate and the league's total-goals
  distribution, with no team information. On nights with saved DraftKings lines (`lines/*.csv` and the
  `lines/auto/` snapshots on the `picks-data` branch) the model is also scored against DK's no-vig price.
* **Parameters, one at a time.** Each value in `GRID` (run_backtest.py) is tried with everything else at its live
  value. Tuning seasons: 2022-23 to 2024-25. Held-out season: 2025-26. The grid, the score each parameter is
  judged on, and the rule for a recommendation were all fixed before the first run: a change is suggested only if
  the best value on the tuning seasons also beats the live value on the held-out season by more than two standard
  errors. **The backtest never changes a live parameter.** That's your call.
* Not covered: the 3-point edge floor and 35% model floor (need historical prices, see the CLV log), and player
  props (Fix 5).

## Files

| File | What |
|---|---|
| `data.py` | downloads and caches the data |
| `asof.py` | season-to-date tables as of a date |
| `metrics.py` | log loss, Brier, calibration, paired standard errors |
| `run_backtest.py` | the run and the report |
| `results/` | `README.md` report, `param_table.csv`, `recommendations.csv`, `model_vs_baseline.csv`, calibration and goal-distribution CSVs, `predictions_live.csv` (every game's forecast) |
