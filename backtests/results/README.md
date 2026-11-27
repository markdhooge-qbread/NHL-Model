# Backtest results

Run 2026-11-27 · 10,000 sims per game · tuning seasons 2022-23, 2023-24, 2024-25 · held-out 2025-26 · current 2026-27 to date · games where both teams had 5+ prior games.

Every game is rebuilt from season-to-date MoneyPuck data as of that morning, with the #1 goalie by games played so far (what the site assumes before a starter is confirmed). Lower loss is better.

## 1. Model vs a plain baseline

Baseline = league average from the last season of games: home win rate for the winner, and the league total-goals distribution for totals. No team info at all.

| Split | Score | Games | Model | Baseline | Diff | SE | Verdict |
|---|---|---|---|---|---|---|---|
| tuning | win log loss | 3676 | 0.6671 | 0.6894 | -0.0223 | 0.0041 | model better |
| tuning | win Brier | 3676 | 0.2373 | 0.2481 | -0.0108 | 0.0019 | model better |
| tuning | O/U 5.5 log loss | 3676 | 0.6958 | 0.6832 | 0.0126 | 0.0038 | baseline better |
| tuning | O/U 6.5 log loss | 3676 | 0.6994 | 0.6884 | 0.0110 | 0.0037 | baseline better |
| tuning | exact-total log loss | 3676 | 2.2091 | 2.1759 | 0.0332 | 0.0053 | baseline better |
| tuning | 1st-period goals log loss | 3676 | 1.3697 | 1.3663 | 0.0035 | 0.0023 | no clear difference |
| tuning | goal in first 10 log loss | 3676 | 0.6801 | 0.6808 | -0.0007 | 0.0015 | no clear difference |
| held-out | win log loss | 1226 | 0.6848 | 0.6941 | -0.0093 | 0.0052 | no clear difference |
| held-out | win Brier | 1226 | 0.2458 | 0.2505 | -0.0046 | 0.0025 | no clear difference |
| held-out | O/U 5.5 log loss | 1226 | 0.7017 | 0.6835 | 0.0182 | 0.0059 | baseline better |
| held-out | O/U 6.5 log loss | 1226 | 0.7110 | 0.6938 | 0.0172 | 0.0062 | baseline better |
| held-out | exact-total log loss | 1226 | 2.2002 | 2.1627 | 0.0376 | 0.0089 | baseline better |
| held-out | 1st-period goals log loss | 1226 | 1.3700 | 1.3701 | -0.0001 | 0.0036 | no clear difference |
| held-out | goal in first 10 log loss | 1226 | 0.6797 | 0.6779 | 0.0018 | 0.0021 | no clear difference |
| current | win log loss | 314 | 0.6558 | 0.6926 | -0.0368 | 0.0165 | model better |
| current | win Brier | 314 | 0.2324 | 0.2497 | -0.0173 | 0.0076 | model better |
| current | O/U 5.5 log loss | 314 | 0.6900 | 0.6670 | 0.0230 | 0.0117 | no clear difference |
| current | O/U 6.5 log loss | 314 | 0.7203 | 0.6964 | 0.0239 | 0.0121 | no clear difference |
| current | exact-total log loss | 314 | 2.2213 | 2.1663 | 0.0550 | 0.0191 | baseline better |
| current | 1st-period goals log loss | 314 | 1.3699 | 1.3593 | 0.0106 | 0.0082 | no clear difference |
| current | goal in first 10 log loss | 314 | 0.6742 | 0.6734 | 0.0008 | 0.0045 | no clear difference |

## 2. Calibration, held-out season: Home win chance

Forecast vs what happened, by bucket.

| Bucket | Games | Forecast | Actual |
|---|---|---|---|
| 25-30% | 4 | 0.2793 | 0.0000 |
| 30-35% | 22 | 0.3315 | 0.4091 |
| 35-40% | 57 | 0.3804 | 0.4211 |
| 40-45% | 145 | 0.4267 | 0.4621 |
| 45-50% | 215 | 0.4766 | 0.4930 |
| 50-55% | 299 | 0.5250 | 0.4883 |
| 55-60% | 236 | 0.5752 | 0.5975 |
| 60-65% | 147 | 0.6232 | 0.5714 |
| 65-70% | 75 | 0.6719 | 0.6000 |
| 70-75% | 18 | 0.7247 | 0.7222 |
| 75-80% | 8 | 0.7776 | 0.7500 |

## 2. Calibration, held-out season: Over 5.5 chance

Forecast vs what happened, by bucket.

| Bucket | Games | Forecast | Actual |
|---|---|---|---|
| 20-25% | 1 | 0.2488 | 1.0000 |
| 25-30% | 10 | 0.2815 | 0.5000 |
| 30-35% | 34 | 0.3300 | 0.3235 |
| 35-40% | 116 | 0.3791 | 0.5948 |
| 40-45% | 257 | 0.4281 | 0.5953 |
| 45-50% | 346 | 0.4750 | 0.5289 |
| 50-55% | 282 | 0.5228 | 0.5922 |
| 55-60% | 134 | 0.5701 | 0.6567 |
| 60-65% | 41 | 0.6222 | 0.6098 |
| 65-70% | 5 | 0.6716 | 0.8000 |

## 3. Goal-total distribution, all seasons

Share of games with each final total (shootout winner counts as a goal, like the books).

| Total | Forecast | Actual |
|---|---|---|
| 0 | 0.0000 | 0.0000 |
| 1 | 0.0321 | 0.0123 |
| 2 | 0.0338 | 0.0138 |
| 3 | 0.1411 | 0.1018 |
| 4 | 0.0943 | 0.0792 |
| 5 | 0.2051 | 0.2193 |
| 6 | 0.1047 | 0.1129 |
| 7 | 0.1584 | 0.2053 |
| 8 | 0.0674 | 0.0775 |
| 9 | 0.0797 | 0.1066 |
| 10+ | 0.0821 | 0.0713 |

| Event | Forecast | Actual |
|---|---|---|
| A team is shut out | 0.1489 | 0.0991 |
| 7+ total goals | 0.3888 | 0.4607 |
| Goes past regulation | NA | 0.2237 |
| Average total goals | 5.8565 | 6.2372 |

## 4. By games played so far (win log loss, all seasons)

How the season-to-date stats do early on. Feeds Fix 6.

| Fewer team GP | Games | Model | Baseline | Diff | SE |
|---|---|---|---|---|---|
| 1-4 | 332 | 0.6845 | 0.6891 | -0.0047 | 0.0248 |
| 5-9 | 409 | 0.6837 | 0.6891 | -0.0054 | 0.0143 |
| 10-29 | 1522 | 0.6759 | 0.6946 | -0.0187 | 0.0062 |
| 30+ | 3285 | 0.6664 | 0.6891 | -0.0226 | 0.0039 |

## 5. Assumed #1 goalie vs the goalie who actually started

Negative = knowing the real starter helps. This is the cost of locking a pick on an assumed goalie (Fix 3).

| Split | Score | Games | Actual - assumed | SE |
|---|---|---|---|---|
| tuning | win log loss | 3676 | -0.0002 | 0.0015 |
| tuning | exact-total log loss | 3676 | -0.0078 | 0.0021 |
| held-out | win log loss | 1226 | -0.0022 | 0.0026 |
| held-out | exact-total log loss | 1226 | -0.0172 | 0.0039 |

## 6. Parameters, one at a time

Each row moves one parameter, everything else at the live value. Diff = change in the judged score vs live (negative = better), with its standard error. Rule set before running: suggest a change only if the best tuning value also beats live on the held-out season by more than 2 SE. The backup-goalie rows use the goalie who actually started (otherwise no backup ever plays). Moving one parameter at a time can't separate parameters that push the same way. **No live parameter was changed.**

| Parameter | Value | Judged on | Tuning diff | SE | Held-out diff | SE |
|---|---|---|---|---|---|---|
| gamma_shape | 6 | exact-total log loss | 0.0408 | 0.0022 | 0.0438 | 0.0038 |
| gamma_shape | 10 | exact-total log loss | 0.0138 | 0.0012 | 0.0161 | 0.0021 |
| gamma_shape | 18 (live) | exact-total log loss | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| gamma_shape | 30 | exact-total log loss | -0.0073 | 0.0010 | -0.0071 | 0.0017 |
| gamma_shape | 60 | exact-total log loss | -0.0120 | 0.0013 | -0.0106 | 0.0025 |
| gamma_shape | Poisson | exact-total log loss | -0.0153 | 0.0017 | -0.0157 | 0.0030 |
| b2b_off | 0.9 | win log loss | -0.0004 | 0.0002 | -0.0001 | 0.0004 |
| b2b_off | 0.94 (live) | win log loss | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| b2b_off | 0.97 | win log loss | 0.0005 | 0.0002 | 0.0003 | 0.0003 |
| b2b_off | 1 | win log loss | 0.0009 | 0.0003 | 0.0008 | 0.0006 |
| b2b_def | 1 | win log loss | 0.0006 | 0.0002 | 0.0006 | 0.0004 |
| b2b_def | 1.02 | win log loss | 0.0003 | 0.0001 | 0.0001 | 0.0002 |
| b2b_def | 1.04 (live) | win log loss | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| b2b_def | 1.06 | win log loss | -0.0003 | 0.0001 | 0.0001 | 0.0002 |
| home_ice | 0 | win log loss | 0.0024 | 0.0008 | 0.0005 | 0.0013 |
| home_ice | 0.02 | win log loss | 0.0009 | 0.0004 | -0.0001 | 0.0006 |
| home_ice | 0.035 (live) | win log loss | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| home_ice | 0.05 | win log loss | -0.0003 | 0.0004 | 0.0003 | 0.0006 |
| other_goals | 0 | exact-total log loss | 0.0271 | 0.0022 | 0.0376 | 0.0039 |
| other_goals | 0.5 | exact-total log loss | 0.0091 | 0.0012 | 0.0156 | 0.0024 |
| other_goals | 1 (live) | exact-total log loss | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| other_goals | 1.5 | exact-total log loss | -0.0086 | 0.0014 | -0.0117 | 0.0023 |
| ot_xg_weight | 0 | win log loss | 0.0002 | 0.0002 | 0.0002 | 0.0003 |
| ot_xg_weight | 0.3 | win log loss | 0.0001 | 0.0001 | 0.0001 | 0.0002 |
| ot_xg_weight | 0.6 (live) | win log loss | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ot_xg_weight | 0.9 | win log loss | -0.0000 | 0.0001 | -0.0001 | 0.0002 |
| starter_prior | 5 | win log loss | 0.0038 | 0.0007 | 0.0040 | 0.0013 |
| starter_prior | 15 (live) | win log loss | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| starter_prior | 30 | win log loss | -0.0014 | 0.0005 | -0.0018 | 0.0008 |
| starter_prior | 60 | win log loss | -0.0020 | 0.0008 | -0.0026 | 0.0015 |
| backup_prior | 10 | win log loss | 0.0011 | 0.0004 | 0.0019 | 0.0007 |
| backup_prior | 20 (live) | win log loss | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| backup_prior | 40 | win log loss | -0.0007 | 0.0003 | -0.0015 | 0.0005 |
| backup_mean | 0 | win log loss | -0.0002 | 0.0004 | 0.0006 | 0.0007 |
| backup_mean | -0.18 (live) | win log loss | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| backup_mean | -0.3 | win log loss | 0.0007 | 0.0003 | -0.0001 | 0.0005 |
| backup_drag | 0 | win log loss | -0.0002 | 0.0004 | 0.0004 | 0.0006 |
| backup_drag | -0.1 (live) | win log loss | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| backup_drag | -0.2 | win log loss | 0.0007 | 0.0004 | -0.0001 | 0.0006 |
| p1_share | 0.27 | 1st-period goals log loss | 0.0140 | 0.0021 | 0.0176 | 0.0036 |
| p1_share | 0.3 (live) | 1st-period goals log loss | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| p1_share | 0.33 | 1st-period goals log loss | 0.0010 | 0.0019 | -0.0021 | 0.0033 |
| f10_share | 0.42 | goal in first 10 log loss | 0.0087 | 0.0015 | 0.0121 | 0.0026 |
| f10_share | 0.48 (live) | goal in first 10 log loss | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| f10_share | 0.54 | goal in first 10 log loss | -0.0003 | 0.0014 | -0.0039 | 0.0024 |

### Recommendation per parameter

| Parameter | Live | Best on tuning | Recommendation |
|---|---|---|---|
| gamma_shape | 18 | Poisson | Evidence for Poisson: better on tuning and on the held-out season by more than 2 SE. Your call. |
| b2b_off | 0.94 | 0.9 | Weak: 0.9 was a little better on both, but within noise. Keep. |
| b2b_def | 1.04 | 1.06 | Keep: 1.06 won on tuning but not on the held-out season. |
| home_ice | 0.035 | 0.05 | Keep: 0.05 won on tuning but not on the held-out season. |
| other_goals | 1 | 1.5 | Evidence for 1.5: better on tuning and on the held-out season by more than 2 SE. Your call. |
| ot_xg_weight | 0.6 | 0.9 | Weak: 0.9 was a little better on both, but within noise. Keep. |
| starter_prior | 15 | 60 | Weak: 60 was a little better on both, but within noise. Keep. |
| backup_prior | 20 | 40 | Evidence for 40: better on tuning and on the held-out season by more than 2 SE. Your call. |
| backup_mean | -0.18 | 0 | Keep: 0 won on tuning but not on the held-out season. |
| backup_drag | -0.1 | 0 | Keep: 0 won on tuning but not on the held-out season. |
| p1_share | 0.3 | 0.3 | Keep: nothing beat the live value on the tuning seasons. |
| f10_share | 0.48 | 0.54 | Weak: 0.54 was a little better on both, but within noise. Keep. |

Not tested here: the 3-point edge floor and the 35% model floor. They choose bets against prices, and there are no free historical DraftKings prices, so they can only be judged on the nights the site has saved lines (see section 7 and the CLV log).

## 7. Against DraftKings (nights with saved lines)

16 priced outcomes. This is far too few to say anything; it's here so it grows over the season.

| Market | Games | Model log loss | DK no-vig log loss |
|---|---|---|---|
| ML | 8 | 0.6517 | 0.6376 |
| Total 5.5 | 1 | 0.5314 | 0.7603 |
| Total 6.5 | 7 | 0.7276 | 0.6973 |
