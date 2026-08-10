"""Read-only data layer over the local FMP Postgres warehouse (marts schema).

Point-in-time S&P 500 membership comes from marts.mst_universe_history,
prices from marts.mv_prices_adjusted (close_total_return), market caps from
marts.security_market_cap, and index levels (e.g. ^VIX) from
marts.index_prices. All queries are cached as parquet under data/cache/.
"""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "data" / "cache"

# Warehouse credentials. Set DIFFGERBER_FMP_ENV to a dotenv file defining
# FMP_DB_{HOST,PORT,NAME,USER,PASSWORD}; otherwise these locations are tried.
ENV_CANDIDATES = [REPO_ROOT / ".env", REPO_ROOT.parent.parent / "FMP" / ".env"]


def _env_path():
    explicit = os.environ.get("DIFFGERBER_FMP_ENV")
    if explicit:
        return Path(explicit)
    for p in ENV_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "No warehouse credentials found. Set DIFFGERBER_FMP_ENV to a dotenv "
        "file defining FMP_DB_HOST/PORT/NAME/USER/PASSWORD, or place one at "
        + " or ".join(str(p) for p in ENV_CANDIDATES)
        + ". The price warehouse is a private vendor database; see README.md "
        "for what the pipeline expects and how to substitute another source."
    )


@contextmanager
def _connect():
    import psycopg
    from dotenv import load_dotenv

    load_dotenv(_env_path(), override=False)
    conn = psycopg.connect(
        host=os.environ["FMP_DB_HOST"],
        port=os.environ["FMP_DB_PORT"],
        dbname=os.environ["FMP_DB_NAME"],
        user=os.environ["FMP_DB_USER"],
        password=os.environ["FMP_DB_PASSWORD"],
        connect_timeout=15,
    )
    try:
        conn.execute("SET TRANSACTION READ ONLY")
        yield conn
    finally:
        conn.rollback()
        conn.close()


def query_df(sql, params=None):
    with _connect() as conn:
        cur = conn.execute(sql, params or {})
        cols = [d.name for d in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)


def _cached(name, key, loader):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:16]
    path = CACHE_DIR / f"{name}_{digest}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    df = loader()
    df.to_parquet(path)
    return df


def sp500_membership():
    """Point-in-time S&P 500 membership intervals."""
    return _cached(
        "sp500_membership",
        {"v": 1},
        lambda: query_df(
            """SELECT fmp_symbol, effective_from, effective_to
               FROM marts.mst_universe_history
               ORDER BY fmp_symbol, effective_from"""
        ),
    )


def members_asof(membership, as_of):
    as_of = pd.Timestamp(as_of).date()
    eff_to = membership["effective_to"].fillna(pd.Timestamp("2999-12-31").date())
    mask = (membership["effective_from"] <= as_of) & (eff_to > as_of)
    return sorted(membership.loc[mask, "fmp_symbol"].unique())


def prices_total_return(start, end, symbols=None):
    """Wide total-return price panel (rows: date, cols: fmp_symbol)."""
    key = {"start": start, "end": end, "symbols": symbols}

    def load():
        sym_clause = "AND s.fmp_symbol = ANY(%(symbols)s)" if symbols else ""
        df = query_df(
            f"""SELECT p.market_date, s.fmp_symbol, p.close_total_return
                FROM marts.mv_prices_adjusted p
                JOIN marts.mst_security s USING (security_id)
                WHERE p.market_date BETWEEN %(start)s AND %(end)s
                  AND p.close_total_return IS NOT NULL {sym_clause}""",
            {"start": start, "end": end, "symbols": list(symbols) if symbols else None},
        )
        wide = df.pivot_table(
            index="market_date", columns="fmp_symbol", values="close_total_return",
            aggfunc="last",
        )
        wide.index = pd.to_datetime(wide.index)
        return wide.astype(float).sort_index()

    return _cached("prices_tr", key, load)


def market_caps_asof(as_of, lookback_days=10):
    """Latest market cap per symbol STRICTLY BEFORE as_of.

    The portfolio is formed at the close of as_of - 1 (returns.iloc[k] is the
    as_of-1 -> as_of close-to-close return and is the first P&L day), so a
    market cap dated as_of is the close of P&L day 1 and must be excluded.
    """
    key = {"as_of": as_of, "lookback_days": lookback_days, "v": 2}

    def load():
        return query_df(
            """SELECT DISTINCT ON (s.fmp_symbol) s.fmp_symbol, m.market_date,
                      m.market_cap_fmp
               FROM marts.security_market_cap m
               JOIN marts.mst_security s USING (security_id)
               WHERE m.market_date BETWEEN %(lo)s AND %(hi)s
                 AND m.market_cap_fmp IS NOT NULL
               ORDER BY s.fmp_symbol, m.market_date DESC""",
            {
                "lo": pd.Timestamp(as_of) - pd.Timedelta(days=lookback_days),
                "hi": pd.Timestamp(as_of) - pd.Timedelta(days=1),
            },
        )

    df = _cached("mcap", key, load)
    return df.set_index("fmp_symbol")["market_cap_fmp"].astype(float)


def index_levels(symbol, start, end):
    """Daily close for an index symbol, e.g. '^VIX', '^GSPC'."""
    key = {"symbol": symbol, "start": start, "end": end}

    def load():
        df = query_df(
            """SELECT market_date, close_price FROM marts.index_prices
               WHERE index_symbol = %(sym)s
                 AND market_date BETWEEN %(start)s AND %(end)s
               ORDER BY market_date""",
            {"sym": symbol, "start": start, "end": end},
        )
        df["market_date"] = pd.to_datetime(df["market_date"])
        return df.set_index("market_date")[["close_price"]].astype(float)

    return _cached("index", key, load)


def symbol_cik_map():
    df = _cached(
        "cik_map",
        {"v": 1},
        lambda: query_df(
            "SELECT fmp_symbol, cik FROM marts.mst_universe WHERE cik IS NOT NULL"
        ),
    )
    return df.set_index("fmp_symbol")["cik"].to_dict()


COVERAGE_LOG = {}   # as_of -> PIT member price-coverage diagnostics


def pit_universe(as_of, n, prices, membership, min_history=252, max_missing=0.05,
                 max_missing_members=0.25):
    """Point-in-time universe: S&P members at as_of, adequate history in
    `prices` up to as_of, one share class per company (highest market cap),
    ranked by market cap, top n."""
    members = members_asof(membership, as_of)
    # strictly before as_of: the portfolio is formed at the as_of-1 close
    hist = prices.loc[: pd.Timestamp(as_of) - pd.Timedelta(days=1)]
    missing = [s for s in members if s not in hist.columns]
    frac = len(missing) / max(len(members), 1)
    # Warehouse price coverage of PIT members is ~88% in 2016 rising to ~99%
    # by 2026; the gap is dominated by acquisition targets (AET, APC, BCR...).
    # Record it per date so the paper can disclose it, and fail loudly only if
    # it degrades far beyond the known level.
    COVERAGE_LOG[pd.Timestamp(as_of)] = {
        "n_members": len(members), "n_missing": len(missing),
        "missing_frac": frac, "missing": sorted(missing),
    }
    if frac > max_missing_members:
        raise ValueError(
            f"pit_universe({as_of}): {len(missing)}/{len(members)} "
            f"({frac:.1%}) members lack prices — far beyond the known coverage "
            f"gap: {sorted(missing)[:10]}"
        )
    ok = []
    for sym in members:
        if sym not in hist.columns:
            continue
        col = hist[sym].tail(min_history)
        if len(col) >= min_history and col.isna().mean() <= max_missing:
            ok.append(sym)
    mcap = market_caps_asof(as_of)
    cik = symbol_cik_map()
    best_per_company = {}
    for sym in ok:
        company = cik.get(sym, sym)
        cur = best_per_company.get(company)
        if cur is None or mcap.get(sym, 0.0) > mcap.get(cur, 0.0):
            best_per_company[company] = sym
    ranked = sorted(best_per_company.values(), key=lambda s: -mcap.get(s, 0.0))
    if len(ranked) < min(n, 400) // 2:
        raise ValueError(
            f"pit_universe({as_of}): only {len(ranked)} eligible symbols "
            f"(min_history={min_history} may exceed available history)"
        )
    return ranked[:n]


def crypto_prices(start, end):
    """Wide daily close panel for all crypto symbols (24/7 calendar)."""
    key = {"start": start, "end": end}

    def load():
        df = query_df(
            """SELECT market_date, crypto_symbol, close_price
               FROM marts.crypto_prices
               WHERE market_date BETWEEN %(start)s AND %(end)s
                 AND close_price > 0""",
            {"start": start, "end": end},
        )
        wide = df.pivot_table(index="market_date", columns="crypto_symbol",
                              values="close_price", aggfunc="last")
        wide.index = pd.to_datetime(wide.index)
        return wide.astype(float).sort_index()

    return _cached("crypto", key, load)


def crypto_universe(as_of, prices, n=20, min_history=252, max_missing=0.02):
    """Cryptos with adequate history at as_of, largest history first, top n.

    Note: the warehouse holds currently-listed symbols only — acknowledge
    potential survivorship in the paper (same caveat as most crypto studies).
    """
    hist = prices.loc[: pd.Timestamp(as_of)]
    ok = []
    for sym in prices.columns:
        col = hist[sym].tail(min_history)
        if len(col) >= min_history and col.isna().mean() <= max_missing:
            ok.append((hist[sym].notna().sum(), sym))
    return [s for _, s in sorted(ok, reverse=True)[:n]]


def sector_map():
    """fmp_symbol -> sector (11 GICS-style sectors, complete in mst_universe)."""
    df = _cached(
        "sector_map",
        {"v": 1},
        lambda: query_df(
            "SELECT fmp_symbol, sector FROM marts.mst_universe "
            "WHERE sector IS NOT NULL"
        ),
    )
    return df.set_index("fmp_symbol")["sector"].to_dict()


def pit_universe_sector(as_of, n, prices, membership, min_history=252,
                        max_missing=0.05):
    """Sector-balanced point-in-time universe.

    Eligibility is identical to pit_universe (PIT membership, price history
    strictly before as_of, one share class per company). Selection allocates
    n slots equally across sectors (floor division), fills each sector with
    its largest-cap eligible names, and hands remaining slots to the sectors
    with the largest aggregate eligible market cap. Falls back to global
    market-cap order if a sector runs out of names.
    """
    # reuse pit_universe's eligibility by asking for every eligible name
    eligible = pit_universe(as_of, 10**6, prices, membership,
                            min_history=min_history, max_missing=max_missing)
    mcap = market_caps_asof(as_of)
    sec = sector_map()
    return _allocate_sector_slots(eligible, lambda s: sec.get(s, "Unknown"),
                                  mcap, n)


def _allocate_sector_slots(eligible, sector_of, mcap, n):
    """Equal sector slots, mcap-ranked within sector, remainder to the
    largest sectors (by aggregate eligible mcap), global-mcap fallback."""
    by_sector = {}
    for sym in eligible:
        by_sector.setdefault(sector_of(sym), []).append(sym)
    for syms in by_sector.values():
        syms.sort(key=lambda s: -mcap.get(s, 0.0))

    sectors = sorted(by_sector,
                     key=lambda k: -sum(mcap.get(s, 0.0) for s in by_sector[k]))
    base = n // len(sectors)
    chosen = []
    for k in sectors:
        chosen.extend(by_sector[k][:base])
    # remaining slots: next-largest names sector by sector (largest sectors first)
    leftover = n - len(chosen)
    depth = base
    while leftover > 0:
        added = 0
        for k in sectors:
            if leftover == 0:
                break
            if len(by_sector[k]) > depth:
                chosen.append(by_sector[k][depth])
                leftover -= 1
                added += 1
        depth += 1
        if added == 0:                      # sectors exhausted
            rest = [s for s in eligible if s not in set(chosen)]
            chosen.extend(rest[:leftover])
            break
    return chosen[:n]


def pit_universe_sector_exsp(as_of, n, prices, membership, min_history=252,
                             max_missing=0.05):
    """Sector-balanced HOLDOUT universe from names that were NEVER S&P
    members over the whole membership history (ex-S&P pool).

    Eligibility mirrors pit_universe: adequate price history strictly before
    as_of, one share class per company, market caps strictly before as_of;
    additionally requires a sector label. `prices` must be the full warehouse
    panel (not the S&P-only panel). Known caveat (disclosed in the paper):
    warehouse coverage of ex-S&P delistings is thin before 2020, so 2017-19
    holdout results carry survivorship risk shared by all methods.
    """
    ever_sp = set(membership["fmp_symbol"].unique())
    sec = sector_map()
    hist = prices.loc[: pd.Timestamp(as_of) - pd.Timedelta(days=1)]
    mcap = market_caps_asof(as_of)
    cik = symbol_cik_map()
    # exclude at the COMPANY level: a share class of an S&P member (e.g.
    # BRK-A vs member BRK-B) must not leak into the ex-S&P holdout
    ever_sp_companies = {cik.get(s, s) for s in ever_sp}
    ok = []
    for sym in hist.columns:
        if sym in ever_sp or sym not in sec:
            continue
        if cik.get(sym, sym) in ever_sp_companies:
            continue
        if mcap.get(sym, 0.0) <= 0.0:
            continue
        col = hist[sym].tail(min_history)
        if len(col) >= min_history and col.isna().mean() <= max_missing:
            ok.append(sym)
    best_per_company = {}
    for sym in ok:
        company = cik.get(sym, sym)
        cur = best_per_company.get(company)
        if cur is None or mcap.get(sym, 0.0) > mcap.get(cur, 0.0):
            best_per_company[company] = sym
    eligible = sorted(best_per_company.values(), key=lambda s: -mcap.get(s, 0.0))
    if len(eligible) < n:
        raise ValueError(
            f"pit_universe_sector_exsp({as_of}): only {len(eligible)} "
            f"eligible ex-S&P symbols for n={n}")
    chosen = _allocate_sector_slots(eligible, lambda s: sec[s], mcap, n)
    overlap = set(chosen) & ever_sp
    assert not overlap, f"S&P names leaked into holdout: {sorted(overlap)}"
    return chosen


def log_returns(prices):
    r = np.log(prices).diff()
    return r.iloc[1:]
