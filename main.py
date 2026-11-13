#!/usr/bin/env python3
"""Minute-by-minute quantile forecaster for Polymarket's "Next CEO of NVIDIA?" event.

For each market the program forecasts the Yes price at every minute of the
next 4 hours as five quantiles (5/25/50/75/95%). It reads one case at a time
from the input directory and writes one forecast file per case.

Model (fitted on the event's own public price history, no network at run time):

* Centre: the price at the checkpoint (persistence). One-minute returns show
  no usable momentum, so no drift term is fitted.
* Spread: per-market realised volatility from the trailing 240 minutes of
  one-minute moves, each move winsorised at +/-0.05 so a market's artificial
  opening quote cannot dominate the estimate, scaled by 0.9 and floored.
  Upward and downward volatility are estimated separately and blended 50/50
  with the pooled estimate, because falls and rises behave differently here.
* Horizon shape: the spread grows like sqrt(T * (1 - exp(-h / T))) with
  T = 30 minutes, saturating instead of the random walk's sqrt(h): realised
  multi-step moves flatten out well before 240 minutes.
* Tails: the outer quantiles use 1.5x the normal z-value and the inner ones
  0.9x, reflecting the jumpy, heavy-tailed moves these prices make.

Standard library only, deterministic: the same case always yields the same
bytes.
"""

import json
import math
import os

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"

# Standard normal quantiles at the scored levels, to 6 decimals.
Z_VALUES = {0.05: -1.644854, 0.25: -0.67449, 0.5: 0.0, 0.75: 0.67449, 0.95: 1.644854}

WINDOW_MINUTES = 240          # trailing minutes used for volatility
MOVE_CAP = 0.05               # winsorisation cap on a one-minute move
SHAPE_TAU = 30.0              # saturation timescale of the spread, minutes
TAIL_OUTER = 1.5              # z multiplier for the 5% / 95% quantiles
TAIL_INNER = 0.9              # z multiplier for the 25% / 75% quantiles
ASYMMETRY = 0.5               # blend weight of the one-sided volatility
SIGMA_MULT = 0.9              # global volatility shrinkage
SIGMA_FLOOR = 0.0008          # minimum per-minute volatility
SIGMA_CAP = 0.03              # maximum per-minute volatility
DEFAULT_SIGMA = 0.0025        # fallback when a market has no usable history


def _clamp01(x):
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _round6(x):
    """Clamp to [0, 1] and round half-up to 6 decimals (never -0)."""
    return math.floor(_clamp01(x) * 1e6 + 0.5) / 1e6 + 0.0


def _norm_ppf(p):
    """Acklam's rational approximation of the standard normal quantile.

    Only used for levels outside the five scored ones; the scored levels use
    the exact constants in Z_VALUES.
    """
    if p <= 0.0:
        return -8.0
    if p >= 1.0:
        return 8.0
    a = (-3.969683028665376e1, 2.209460984245205e2, -2.759285984469534e2,
         1.383577518672690e2, -3.066479806614716e1, 2.506628277459239e0)
    b = (-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
         6.680131188771972e1, -1.328068155288572e1)
    c = (-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838e0,
         -2.549732539343734e0, 4.374664141464968e0, 2.938163982698783e0)
    d = (7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996e0,
         3.754408661907416e0)
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def _z_value(tau):
    z = Z_VALUES.get(tau)
    return z if z is not None else _norm_ppf(tau)


def _side_volatilities(prices):
    """Return (down_sigma, up_sigma) for one market's price history."""
    n = len(prices)
    if n < 2:
        return DEFAULT_SIGMA, DEFAULT_SIGMA
    w = min(WINDOW_MINUTES, n - 1)
    sum_sq = up_sq = down_sq = 0.0
    up_n = down_n = 0
    for i in range(n - w, n):
        move = prices[i] - prices[i - 1]
        if move > MOVE_CAP:
            move = MOVE_CAP
        elif move < -MOVE_CAP:
            move = -MOVE_CAP
        sq = move * move
        sum_sq += sq
        if move > 0.0:
            up_sq += sq
            up_n += 1
        elif move < 0.0:
            down_sq += sq
            down_n += 1
    pooled = math.sqrt(sum_sq / w) * SIGMA_MULT
    pooled = min(SIGMA_CAP, max(SIGMA_FLOOR, pooled))
    if up_n:
        up = math.sqrt(up_sq / w) * SIGMA_MULT
        up = min(SIGMA_CAP, max(SIGMA_FLOOR, up))
    else:
        up = pooled
    if down_n:
        down = math.sqrt(down_sq / w) * SIGMA_MULT
        down = min(SIGMA_CAP, max(SIGMA_FLOOR, down))
    else:
        down = pooled
    up = (1.0 - ASYMMETRY) * pooled + ASYMMETRY * up
    down = (1.0 - ASYMMETRY) * pooled + ASYMMETRY * down
    return down, up


def _tail_factor(tau):
    """z multiplier for a level: fatter tails at the extremes, tighter inside."""
    if tau <= 0.05 or tau >= 0.95:
        return TAIL_OUTER
    return TAIL_INNER


def forecast_market(prices, horizon, quantiles, shape):
    """Forecast rows for one market: horizon rows of len(quantiles) numbers."""
    centre = prices[-1] if prices else 0.5
    down_sigma, up_sigma = _side_volatilities(prices)
    rows = []
    for h in range(horizon):
        width = shape[h]
        row = []
        for tau in quantiles:
            z = _z_value(tau)
            if tau < 0.5:
                q = centre + z * _tail_factor(tau) * down_sigma * width
            elif tau > 0.5:
                q = centre + z * _tail_factor(tau) * up_sigma * width
            else:
                q = centre
            row.append(_round6(q))
        # Defensive: rounding must never break the non-decreasing order.
        for j in range(1, len(row)):
            if row[j] < row[j - 1]:
                row[j] = row[j - 1]
        rows.append(row)
    return rows


def forecast_case(case):
    horizon = int(case["horizon_minutes"])
    levels = case.get("quantiles") or sorted(Z_VALUES)
    quantiles = [float(t) for t in levels]
    shape = [math.sqrt(SHAPE_TAU * (1.0 - math.exp(-(h + 1) / SHAPE_TAU)))
             for h in range(horizon)]
    forecasts = []
    for market in case["markets"]:
        rows = forecast_market(market["prices"], horizon, quantiles, shape)
        forecasts.append({"id": market["id"], "q": rows})
    return {"forecasts": forecasts}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    names = sorted(name for name in os.listdir(INPUT_DIR) if name.endswith(".in"))
    for name in names:
        case_id = name[:-3]
        with open(os.path.join(INPUT_DIR, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        result = forecast_case(case)
        with open(os.path.join(OUTPUT_DIR, case_id + ".out"), "w", encoding="utf-8") as fh:
            fh.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
