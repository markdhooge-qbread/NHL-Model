# Backtest results

Run 2026-11-27 · 10,000 sims per game · tuning seasons 2022-23, 2023-24, 2024-25 · held-out 2025-26 · current 2026-27 to date · games where both teams had 5+ prior games.

Every game is rebuilt from season-to-date MoneyPuck data as of that morning, with the #1 goalie by games played so far (what the site assumes before a starter is confirmed). Lower loss is better.

## 1. Model vs a plain baseline

Baseline = league average from the last season of games: home win rate for the winner, and the league total-goals distribution for totals. No team info at all.

| Split | Score | Games | Model | Baseline | Diff | SE | Verdict |
|---|---|---|---|---|---|---|---|
| tuning | win log loss | 3676 | 0.6672 | 0.6894 | -0.0221 | 0.0041 | model better |
| tuning | win Brier | 3676 | 0.2374 | 0.2481 | -0.0107 | 0.0019 | model better |
| tuning | O/U 5.5 log loss | 3676 | 0.6890 | 0.6832 | 0.0058 | 0.0032 | no clear difference |
| tuning | O/U 6.5 log loss | 3676 | 0.6939 | 0.6884 | 0.0055 | 0.0032 | no clear difference |
| tuning | exact-total log loss | 3676 | 2.1926 | 2.1759 | 0.0167 | 0.0046 | baseline better |
| tuning | 1st-period goals log loss | 3676 | 1.3681 | 1.3663 | 0.0018 | 0.0021 | no clear difference |
| tuning | goal in first 10 log loss | 3676 | 0.6793 | 0.6808 | -0.0015 | 0.0012 | no clear difference |
| held-out | win log loss | 1226 | 0.6849 | 0.6941 | -0.0092 | 0.0052 | no clear difference |
| held-out | win Brier | 1226 | 0.2459 | 0.2505 | -0.0046 | 0.0025 | no clear difference |
| held-out | O/U 5.5 log loss | 1226 | 0.6913 | 0.6835 | 0.0078 | 0.0049 | no clear difference |
| held-out | O/U 6.5 log loss | 1226 | 0.7007 | 0.6938 | 0.0069 | 0.0051 | no clear difference |
| held-out | exact-total log loss | 1226 | 2.1819 | 2.1627 | 0.0192 | 0.0078 | baseline better |
| held-out | 1st-period goals log loss | 1226 | 1.3680 | 1.3701 | -0.0020 | 0.0033 | no clear difference |
| held-out | goal in first 10 log loss | 1226 | 0.6779 | 0.6779 | -0.0001 | 0.0018 | no clear difference |
| current | win log loss | 314 | 0.6562 | 0.6926 | -0.0364 | 0.0167 | model better |
| current | win Brier | 314 | 0.2326 | 0.2497 | -0.0171 | 0.0077 | model better |
| current | O/U 5.5 log loss | 314 | 0.6775 | 0.6670 | 0.0105 | 0.0099 | no clear difference |
| current | O/U 6.5 log loss | 314 | 0.7095 | 0.6964 | 0.0131 | 0.0104 | no clear difference |
| current | exact-total log loss | 314 | 2.2014 | 2.1663 | 0.0351 | 0.0166 | baseline better |
| current | 1st-period goals log loss | 314 | 1.3664 | 1.3593 | 0.0071 | 0.0074 | no clear difference |
| current | goal in first 10 log loss | 314 | 0.6737 | 0.6734 | 0.0003 | 0.0039 | no clear difference |

## 2. Calibration, held-out season: Home win chance

Forecast vs what happened, by bucket.

| Bucket | Games | Forecast | Actual |
|---|---|---|---|
| 25-30% | 5 | 0.2816 | 0.2000 |
| 30-35% | 20 | 0.3311 | 0.4000 |
| 35-40% | 59 | 0.3808 | 0.4068 |
| 40-45% | 147 | 0.4273 | 0.4626 |
| 45-50% | 200 | 0.4767 | 0.4850 |
| 50-55% | 302 | 0.5251 | 0.5066 |
| 55-60% | 221 | 0.5741 | 0.5701 |
| 60-65% | 166 | 0.6218 | 0.5783 |
| 65-70% | 76 | 0.6722 | 0.6053 |
| 70-75% | 20 | 0.7219 | 0.7000 |
| 75-80% | 9 | 0.7748 | 0.8889 |

## 2. Calibration, held-out season: Over 5.5 chance

Forecast vs what happened, by bucket.

| Bucket | Games | Forecast | Actual |
|---|---|---|---|
| 25-30% | 3 | 0.2829 | 0.6667 |
| 30-35% | 17 | 0.3238 | 0.4118 |
| 35-40% | 59 | 0.3791 | 0.4576 |
| 40-45% | 164 | 0.4288 | 0.5854 |
| 45-50% | 317 | 0.4759 | 0.5521 |
| 50-55% | 329 | 0.5247 | 0.5866 |
| 55-60% | 232 | 0.5726 | 0.5776 |
| 60-65% | 78 | 0.6217 | 0.6795 |
| 65-70% | 25 | 0.6669 | 0.6800 |
| 70-75% | 2 | 0.7160 | 1.0000 |

## 3. Goal-total distribution, all seasons

Share of games with each final total (shootout winner counts as a goal, like the books).

| Total | Forecast | Actual |
|---|---|---|
| 0 | 0.0000 | 0.0000 |
| 1 | 0.0254 | 0.0123 |
| 2 | 0.0289 | 0.0138 |
| 3 | 0.1278 | 0.1018 |
| 4 | 0.0897 | 0.0792 |
| 5 | 0.2048 | 0.2193 |
| 6 | 0.1073 | 0.1129 |
| 7 | 0.1683 | 0.2053 |
| 8 | 0.0717 | 0.0775 |
| 9 | 0.0870 | 0.1066 |
| 10+ | 0.0879 | 0.0713 |

| Event | Forecast | Actual |
|---|---|---|
| A team is shut out | 0.1293 | 0.0991 |
| 7+ total goals | 0.4161 | 0.4607 |
| Goes past regulation | NA | 0.2237 |
| Average total goals | 6.0266 | 6.2372 |
| Over 5.5 (forecast = average chance) | 0.5233 | 0.5736 |
| Over 6.5 (forecast = average chance) | 0.4161 | 0.4607 |
| Games the model leans under 5.5 | 0.3859 | NA |
| Games the model leans under 6.5 | 0.8585 | NA |

## 4. By games played so far (win log loss, all seasons)

How the season-to-date stats do early on. Feeds Fix 6.

| Fewer team GP | Games | Model | Baseline | Diff | SE |
|---|---|---|---|---|---|
| 1-4 | 332 | 0.6865 | 0.6891 | -0.0026 | 0.0255 |
| 5-9 | 409 | 0.6843 | 0.6891 | -0.0048 | 0.0145 |
| 10-29 | 1522 | 0.6767 | 0.6946 | -0.0179 | 0.0063 |
| 30+ | 3285 | 0.6663 | 0.6891 | -0.0228 | 0.0040 |

## 5. Assumed #1 goalie vs the goalie who actually started

Negative = knowing the real starter helps. This is the cost of locking a pick on an assumed goalie (Fix 3).

| Split | Score | Games | Actual - assumed | SE |
|---|---|---|---|---|
| tuning | win log loss | 3676 | -0.0004 | 0.0015 |
| tuning | exact-total log loss | 3676 | -0.0006 | 0.0021 |
| held-out | win log loss | 1226 | -0.0020 | 0.0026 |
| held-out | exact-total log loss | 1226 | -0.0135 | 0.0042 |

## 6. Parameters, one at a time

Each row moves one parameter, everything else at the live value. Diff = change in the judged score vs live (negative = better), with its standard error. Rule set before running: suggest a change only if the best tuning value also beats live on the held-out season by more than 2 SE. The backup-goalie rows use the goalie who actually started (otherwise no backup ever plays). Moving one parameter at a time can't separate parameters that push the same way. **No live parameter was changed.**

| Parameter | Value | Judged on | Tuning diff | SE | Held-out diff | SE |
|---|---|---|---|---|---|---|
| gamma_shape | 6 | exact-total log loss | 0.0495 | 0.0027 | 0.0531 | 0.0049 |
| gamma_shape | 10 | exact-total log loss | 0.0239 | 0.0016 | 0.0222 | 0.0032 |
| gamma_shape | 18 | exact-total log loss | 0.0079 | 0.0010 | 0.0066 | 0.0022 |
| gamma_shape | 30 (live) | exact-total log loss | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| gamma_shape | 60 | exact-total log loss | -0.0026 | 0.0009 | -0.0048 | 0.0016 |
| gamma_shape | Poisson | exact-total log loss | -0.0068 | 0.0013 | -0.0108 | 0.0022 |
| b2b_off | 0.9 | win log loss | -0.0005 | 0.0002 | -0.0000 | 0.0004 |
| b2b_off | 0.94 (live) | win log loss | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| b2b_off | 0.97 | win log loss | 0.0005 | 0.0002 | 0.0002 | 0.0003 |
| b2b_off | 1 | win log loss | 0.0010 | 0.0003 | 0.0004 | 0.0006 |
| b2b_def | 1 | win log loss | 0.0006 | 0.0002 | 0.0003 | 0.0004 |
| b2b_def | 1.02 | win log loss | 0.0003 | 0.0001 | 0.0001 | 0.0002 |
| b2b_def | 1.04 (live) | win log loss | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| b2b_def | 1.06 | win log loss | -0.0002 | 0.0001 | -0.0001 | 0.0002 |
| home_ice | 0 | win log loss | 0.0022 | 0.0008 | 0.0003 | 0.0014 |
| home_ice | 0.02 | win log loss | 0.0007 | 0.0004 | -0.0002 | 0.0006 |
| home_ice | 0.035 (live) | win log loss | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| home_ice | 0.05 | win log loss | -0.0003 | 0.0004 | 0.0007 | 0.0006 |
| other_goals | 0 | exact-total log loss | 0.0382 | 0.0033 | 0.0501 | 0.0057 |
| other_goals | 0.5 | exact-total log loss | 0.0138 | 0.0017 | 0.0188 | 0.0030 |
| other_goals | 1 (live) | exact-total log loss | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| other_goals | 1.5 | exact-total log loss | -0.0007 | 0.0016 | -0.0105 | 0.0030 |
| ot_xg_weight | 0 | win log loss | 0.0002 | 0.0002 | 0.0001 | 0.0003 |
| ot_xg_weight | 0.3 | win log loss | 0.0001 | 0.0001 | 0.0000 | 0.0002 |
| ot_xg_weight | 0.6 (live) | win log loss | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ot_xg_weight | 0.9 | win log loss | -0.0000 | 0.0001 | -0.0001 | 0.0002 |
| starter_prior | 5 | win log loss | 0.0037 | 0.0007 | 0.0044 | 0.0013 |
| starter_prior | 15 (live) | win log loss | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| starter_prior | 30 | win log loss | -0.0017 | 0.0005 | -0.0017 | 0.0008 |
| starter_prior | 60 | win log loss | -0.0023 | 0.0008 | -0.0023 | 0.0015 |
| backup_prior | 10 | win log loss | 0.0012 | 0.0004 | 0.0020 | 0.0007 |
| backup_prior | 20 (live) | win log loss | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| backup_prior | 40 | win log loss | -0.0007 | 0.0003 | -0.0011 | 0.0006 |
| backup_mean | 0 | win log loss | -0.0004 | 0.0004 | 0.0005 | 0.0007 |
| backup_mean | -0.18 (live) | win log loss | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| backup_mean | -0.3 | win log loss | 0.0004 | 0.0003 | 0.0001 | 0.0005 |
| backup_drag | 0 | win log loss | -0.0003 | 0.0004 | 0.0004 | 0.0006 |
| backup_drag | -0.1 (live) | win log loss | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| backup_drag | -0.2 | win log loss | 0.0005 | 0.0004 | -0.0001 | 0.0006 |
| p1_share | 0.27 | 1st-period goals log loss | 0.0094 | 0.0021 | 0.0126 | 0.0036 |
| p1_share | 0.3 (live) | 1st-period goals log loss | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| p1_share | 0.33 | 1st-period goals log loss | 0.0062 | 0.0020 | 0.0025 | 0.0033 |
| f10_share | 0.42 | goal in first 10 log loss | 0.0065 | 0.0016 | 0.0102 | 0.0027 |
| f10_share | 0.48 (live) | goal in first 10 log loss | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| f10_share | 0.54 | goal in first 10 log loss | 0.0020 | 0.0015 | -0.0019 | 0.0025 |

### Recommendation per parameter

| Parameter | Live | Best on tuning | Recommendation |
|---|---|---|---|
| gamma_shape | 30 | Poisson | Evidence for Poisson: better on tuning and on the held-out season by more than 2 SE. Your call. |
| b2b_off | 0.94 | 0.9 | Weak: 0.9 was a little better on both, but within noise. Keep. |
| b2b_def | 1.04 | 1.06 | Weak: 1.06 was a little better on both, but within noise. Keep. |
| home_ice | 0.035 | 0.05 | Keep: 0.05 won on tuning but not on the held-out season. |
| other_goals | 1 | 1.5 | Evidence for 1.5: better on tuning and on the held-out season by more than 2 SE. Your call. |
| ot_xg_weight | 0.6 | 0.9 | Weak: 0.9 was a little better on both, but within noise. Keep. |
| starter_prior | 15 | 60 | Weak: 60 was a little better on both, but within noise. Keep. |
| backup_prior | 20 | 40 | Evidence for 40: better on tuning and on the held-out season by more than 2 SE. Your call. |
| backup_mean | -0.18 | 0 | Keep: 0 won on tuning but not on the held-out season. |
| backup_drag | -0.1 | 0 | Keep: 0 won on tuning but not on the held-out season. |
| p1_share | 0.3 | 0.3 | Keep: nothing beat the live value on the tuning seasons. |
| f10_share | 0.48 | 0.48 | Keep: nothing beat the live value on the tuning seasons. |

Not tested here: the 3-point edge floor and the 35% model floor. They choose bets against prices, and there are no free historical DraftKings prices, so they can only be judged on the nights the site has saved lines (see section 7 and the CLV log).

## 7. Against DraftKings (nights with saved lines)

16 priced outcomes. This is far too few to say anything; it's here so it grows over the season.

| Market | Games | Model log loss | DK no-vig log loss |
|---|---|---|---|
| ML | 8 | 0.6467 | 0.6376 |
| Total 5.5 | 1 | 0.5823 | 0.7603 |
| Total 6.5 | 7 | 0.7446 | 0.6973 |
