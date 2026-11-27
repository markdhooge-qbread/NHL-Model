"""Scoring rules for probability forecasts. Lower is better for every loss here."""
import numpy as np
import pandas as pd

EPS = 1e-4


def log_loss_vec(p, y):
    """Per-game binary log loss."""
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def brier_vec(p, y):
    return (np.asarray(p, float) - np.asarray(y, float)) ** 2


def cat_log_loss_vec(prob_of_actual):
    """Per-game log loss of a categorical forecast, given the probability it put on what happened."""
    return -np.log(np.clip(np.asarray(prob_of_actual, float), EPS, 1.0))


def paired_diff(loss_new, loss_old):
    """Mean difference (new - old) and its standard error, game by game. Negative = new is better."""
    d = np.asarray(loss_new, float) - np.asarray(loss_old, float)
    d = d[~np.isnan(d)]
    if len(d) < 2:
        return float("nan"), float("nan")
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


def calibration_table(p, y, edges):
    """Forecast vs actual rate per probability bucket."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum():
            rows.append({"bucket": f"{lo*100:.0f}-{hi*100:.0f}%", "games": int(m.sum()),
                         "forecast": float(p[m].mean()), "actual": float(y[m].mean())})
    return pd.DataFrame(rows)
