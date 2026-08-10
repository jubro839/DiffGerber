"""Data QA: coverage, membership, missing data, extreme returns."""

import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from diffgerber import data

START, END = "2015-01-01", "2026-07-28"

print("== S&P 500 point-in-time membership ==")
mem = data.sp500_membership()
print(f"  intervals: {len(mem)}, distinct symbols: {mem['fmp_symbol'].nunique()}")
for d in ["2016-01-04", "2020-01-02", "2024-01-02", "2026-07-01"]:
    print(f"  members on {d}: {len(data.members_asof(mem, d))}")

print("== full price panel (all S&P members ever) ==")
all_syms = sorted(mem["fmp_symbol"].unique())
px = data.prices_total_return(START, END, symbols=all_syms)
print(f"  shape: {px.shape}  ({px.index.min().date()} .. {px.index.max().date()})")
print(f"  overall missing frac: {px.isna().mean().mean():.4f}")

r = data.log_returns(px)
extreme = (r.abs() > 0.5).sum().sum()
print(f"  |log return| > 50% events: {extreme} "
      f"({extreme / r.notna().sum().sum() * 100:.4f}% of obs)")

print("== point-in-time top-100 universe on 2026-07-01 ==")
uni = data.pit_universe("2026-07-01", 100, px, mem)
print(f"  size: {len(uni)}, first 10: {uni[:10]}")

print("== VIX ==")
vix = data.index_levels("^VIX", START, END)
print(f"  rows: {len(vix)}, last: {vix.iloc[-1, 0]:.2f} on {vix.index[-1].date()}")

print("\nQA done.")
