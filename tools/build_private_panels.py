"""Construct auditable, deliberately lagged EXPLORATORY panels from saved snapshots.

Availability timestamps below are assumed lag policies, not observed archival timestamps.
The current historical snapshots can contain corrections/revisions. This is not a certified
point-in-time dataset. All these limitations are carried into the output manifest.

This optional tool writes inside COURSEWORK_PRIVATE_SNAPSHOT. Rebuild in a
working copy rather than overwriting the original frozen evidence.
"""
from pathlib import Path
import hashlib
import json
import os

import numpy as np
import pandas as pd

ROOT = Path(os.environ['COURSEWORK_PRIVATE_SNAPSHOT']).resolve()
ASOF = pd.Timestamp(os.environ.get('COURSEWORK_ASOF', '2026-11-13T12:00:00Z'))
HOLDOUT = '2024-10-25T12:00:00+01:00'
FEATURES = {}
COVERAGE = []


def yahoo(symbol, adjusted=False, first_live=None):
    path = ROOT / 'data/raw' / f'yahoo_{symbol.replace("^", "")}.json'
    payload = json.loads(path.read_text())['chart']['result'][0]
    if payload['meta']['symbol'] != symbol or payload['meta']['dataGranularity'] != '1d':
        raise ValueError(f'Unexpected series metadata for {symbol}')
    dates = pd.to_datetime(payload['timestamp'], unit='s', utc=True).tz_convert('America/New_York').normalize().tz_localize(None)
    data = payload['indicators']['adjclose'][0]['adjclose'] if adjusted else payload['indicators']['quote'][0]['close']
    s = pd.Series(data, index=dates, dtype=float).dropna().sort_index()
    s = s[~s.index.duplicated(keep='last')]
    if first_live:
        s = s[s.index >= pd.Timestamp(first_live)]
    # Midnight NY following the observation date: delayed until well after US close.
    available = pd.DatetimeIndex(s.index + pd.Timedelta(days=1)).tz_localize('America/New_York').tz_convert('UTC')
    return source(s, available, symbol, 'Yahoo historical snapshot; next-day midnight New York availability assumption')


def fred(symbol, lag_business_days=2, lag_calendar_days=None):
    path = ROOT / 'data/raw' / f'fred_{symbol}.csv'
    df = pd.read_csv(path)
    s = pd.Series(pd.to_numeric(df[symbol], errors='coerce').to_numpy(),
                  index=pd.to_datetime(df.iloc[:, 0])).dropna().sort_index()
    if lag_calendar_days is None:
        available = (s.index + pd.offsets.BDay(lag_business_days)).tz_localize('UTC')
        policy = f'{lag_business_days} weekday lag, midnight UTC; US holidays not independently reconstructed'
    else:
        available = (s.index + pd.Timedelta(days=lag_calendar_days)).tz_localize('UTC')
        policy = f'{lag_calendar_days} calendar-day lag, midnight UTC; not a historical release audit'
    return source(s, available, symbol, 'FRED latest-vintage snapshot; ' + policy)


def source(s, available, name, policy):
    frame = pd.DataFrame({'value': s.to_numpy(), 'available': available}, index=s.index)
    frame = frame[frame.available <= ASOF]
    COVERAGE.append(dict(series=name, first=str(frame.index.min().date()), last=str(frame.index.max().date()),
                         observations=len(frame), policy=policy))
    return frame


def add(number, name, values, available, formula, sources, caveat=''):
    frame = pd.DataFrame({'value': values, 'available': available}).replace([np.inf, -np.inf], np.nan).dropna()
    FEATURES[name] = frame
    return dict(number=number, name=name, formula=formula, sources=sources, caveat=caveat,
                first_observation=str(frame.index.min().date()), last_observation=str(frame.index.max().date()))


def build():
    definitions = []
    sp = yahoo('^GSPC')
    p, available = sp.value, sp.available
    logret = np.log(p / p.shift(1))
    definitions += [add(i, f'return_{w}', p.pct_change(w, fill_method=None), available,
                        f'GSPC close / close {w} sessions earlier - 1', ['^GSPC'])
                    for i, w in enumerate([1, 5, 20, 60, 120], 1)]
    for i, w in [(6, 20), (7, 200)]:
        definitions.append(add(i, f'ma_gap_{w}', p / p.rolling(w).mean() - 1,
                               available, f'GSPC / {w}-session moving average - 1', ['^GSPC']))
    delta = p.diff()
    # Exact Wilder seed: first 14 arithmetic-average changes, then recursive smoothing.
    def wilder(series, window=14):
        result = pd.Series(np.nan, index=series.index)
        result.iloc[window] = series.iloc[1:window+1].mean()
        for j in range(window+1, len(series)):
            result.iloc[j] = (result.iloc[j-1] * (window-1) + series.iloc[j]) / window
        return result
    gains, losses = wilder(delta.clip(lower=0)), wilder(-delta.clip(upper=0))
    rsi = 100 - 100 / (1 + gains / losses)
    rsi = rsi.mask((losses == 0) & (gains > 0), 100).mask((losses == 0) & (gains == 0), 50)
    definitions.append(add(8, 'rsi_14', rsi, available, '14-session RSI, Wilder arithmetic seed then smoothing', ['^GSPC']))
    vols = {w: logret.rolling(w).std(ddof=1) * np.sqrt(252) for w in [10, 20, 60]}
    for i, w in [(9, 10), (10, 20)]:
        definitions.append(add(i, f'vol_{w}', vols[w], available, f'SD of {w} daily log returns * sqrt(252)', ['^GSPC']))
    definitions.append(add(11, 'vol_ratio_10_60', vols[10] / vols[60], available,
                           '10-session volatility / 60-session volatility', ['^GSPC']))
    definitions.append(add(12, 'downside_vol_20', np.sqrt(logret.clip(upper=0).pow(2).rolling(20).mean() * 252),
                           available, 'sqrt(252 * mean(min(log return,0)^2)), 20 sessions', ['^GSPC']))
    vix = yahoo('^VIX').reindex(sp.index).dropna()
    definitions.append(add(13, 'vix_level', vix.value, vix.available, 'Latest VIX close', ['^VIX']))
    definitions.append(add(14, 'vix_change_5', vix.value.diff(5), vix.available, 'VIX minus VIX 5 sessions earlier', ['^VIX']))
    definitions.append(add(15, 'vix_ma_gap_20', vix.value / vix.value.rolling(20).mean() - 1,
                           vix.available, 'VIX / 20-session mean - 1', ['^VIX']))
    v3 = yahoo('^VIX3M', first_live='2007-12-04').reindex(vix.index).dropna()
    term = vix.value.reindex(v3.index) / v3.value - 1
    term_available = pd.concat([vix.available.reindex(v3.index), v3.available], axis=1).max(axis=1)
    definitions.append(add(16, 'vix_term', term, term_available, 'VIX / VIX3M - 1', ['^VIX', '^VIX3M'],
                           'Exclude pre-launch reconstructed history; Yahoo provider timing is assumed'))
    vvix = yahoo('^VVIX', first_live='2012-03-14').reindex(sp.index).dropna()
    definitions.append(add(17, 'vvix_level', vvix.value, vvix.available, 'Latest VVIX close', ['^VVIX'],
                           'Pre-2012-03-14 reconstructed history excluded'))
    spy = yahoo('SPY', adjusted=True)
    for i, name, numerator, denominator in [
        (23, 'equal_weight_relative_20', 'RSP', 'SPY'),
        (24, 'small_cap_relative_20', 'IWM', 'SPY'),
        (25, 'technology_relative_20', 'XLK', 'SPY'),
        (26, 'cyclical_defensive_20', 'XLY', 'XLP')]:
        a = yahoo(numerator, adjusted=True)
        b = spy if denominator == 'SPY' else yahoo(denominator, adjusted=True)
        joined = a.join(b, lsuffix='_a', rsuffix='_b', how='inner')
        values = joined.value_a.pct_change(20, fill_method=None) - joined.value_b.pct_change(20, fill_method=None)
        times = joined[['available_a', 'available_b']].max(axis=1)
        definitions.append(add(i, name, values, times,
            f'{numerator} 20-session adjusted-close return minus {denominator} equivalent', [numerator, denominator],
            'ETF total-return proxy, with current historical adjustments; not the forecast target'))
    d2, d10, real = fred('DGS2'), fred('DGS10'), fred('DFII10')
    for i, name, data in [(27, 'yield2_change_20', d2), (28, 'yield10_change_20', d10),
                           (31, 'real_yield_change_20', real)]:
        definitions.append(add(i, name, data.value.diff(20), data.available,
                               'Yield minus yield 20 available observations earlier, percentage points',
                               [{27: 'DGS2', 28: 'DGS10', 31: 'DFII10'}[i]]))
    joined = d10.join(d2, lsuffix='_10', rsuffix='_2', how='inner')
    definitions.append(add(29, 'yield_curve', joined.value_10 - joined.value_2,
        joined[['available_10', 'available_2']].max(axis=1), 'DGS10 minus DGS2, percentage points', ['DGS10', 'DGS2']))
    definitions.append(add(30, 'real_yield', real.value, real.available, 'DFII10 level, percent', ['DFII10']))
    dollar = fred('DTWEXBGS', lag_calendar_days=10)
    definitions.append(add(34, 'dollar_return_20', dollar.value.pct_change(20, fill_method=None), dollar.available,
                           'Broad trade-weighted dollar index 20-observation return', ['DTWEXBGS'],
                           '10-day availability lag; current vintage, not certified historical vintage'))
    oil = fred('DCOILWTICO', lag_calendar_days=10)
    oil_return = oil.value.pct_change(20, fill_method=None)
    # Percent momentum is undefined economically across nonpositive price endpoints.
    oil_return = oil_return.where((oil.value > 0) & (oil.value.shift(20) > 0))
    definitions.append(add(35, 'oil_return_20', oil_return, oil.available,
                           'WTI spot price 20-observation return; nonpositive endpoints invalid', ['DCOILWTICO'],
                           '10-day lag; nonpositive endpoint observations excluded and logged'))
    gold = yahoo('GLD', adjusted=True)
    definitions.append(add(36, 'gold_proxy_return_20', gold.value.pct_change(20, fill_method=None), gold.available,
                           'GLD adjusted-close 20-session return', ['GLD'],
                           'Gold ETF proxy instead of spot gold; explicit implementation refinement'))
    excluded = [dict(number=18, reason='Equity put-call archive not yet reconstructed and checked'),
                *[dict(number=i, reason='Historical S&P membership and constituent prices unavailable in this dataset') for i in range(19,23)],
                *[dict(number=i, reason='Current FRED ICE credit snapshot has only 3 years; insufficient pre-holdout training') for i in [32,33]]]

    # Normal six-session cases only, matching the actual November horizon.
    origins_end = (ASOF - pd.Timedelta(days=7)).tz_convert('Europe/London')
    origins = pd.date_range(pd.Timestamp('2000-01-07 12:00', tz='Europe/London'), origins_end, freq='W-FRI', tz='Europe/London')
    rows, skipped = [], []
    for origin in origins:
        utc = origin.tz_convert('UTC')
        target_date = pd.Timestamp(origin.date()) + pd.Timedelta(days=7)
        target_at = (target_date + pd.Timedelta(hours=16)).tz_localize('America/New_York').tz_convert('UTC')
        target = sp[sp.index <= target_date]
        usable = sp[sp.available <= utc]
        if target.empty or usable.empty or target.iloc[-1].available > ASOF:
            continue
        baseline_date, target_session = usable.index[-1], target.index[-1]
        sessions = len(sp[(sp.index > baseline_date) & (sp.index <= target_session)])
        if baseline_date != pd.Timestamp(origin.date()) - pd.Timedelta(days=1) or target_session != target_date or sessions != 6:
            skipped.append(dict(origin=str(utc), reason='Holiday/missing-session mismatch with normal six-session horizon'))
            continue
        row = dict(origin=utc, baseline=usable.value.iloc[-1], baseline_available_at=usable.available.iloc[-1],
                   baseline_date=str(baseline_date.date()), actual=target.value.iloc[-1], target_at=target_at,
                   target_session_date=str(target_session.date()), target_available_at=target.available.iloc[-1])
        for name, frame in FEATURES.items():
            allowed = frame[frame.available <= utc]
            if len(allowed):
                last = allowed.iloc[-1]
                # A missing recent value must not silently become an indefinitely stale feature.
                if (utc - last.available).days <= 7:
                    row['x_' + name] = last.value
                    row['available_' + name] = last.available
        rows.append(row)
    all_rows = pd.DataFrame(rows)
    out = ROOT / 'data/panels'
    out.mkdir(parents=True, exist_ok=True)
    long_ids = set(range(1,16)) | {24,25,26,27,28,29,35}
    cohorts = {'long_history': [d['name'] for d in definitions if d['number'] in long_ids],
               'expanded': [d['name'] for d in definitions]}
    summaries = {}
    for cohort, names in cohorts.items():
        base = ['origin','baseline','baseline_available_at','baseline_date','actual','target_at','target_session_date','target_available_at']
        columns = base + [c for name in names for c in ['x_' + name, 'available_' + name]]
        panel = all_rows[columns].dropna().copy()
        # Keep the smaller long-history cohort separate from the expanded universe.
        path = out / f'{cohort}.csv'
        panel.to_csv(path, index=False)
        summaries[cohort] = dict(factors=len(names), models=len(names)*(len(names)+1)//2,
            origins=len(panel), first=str(panel.origin.min()), last=str(panel.origin.max()),
            holdout_start=HOLDOUT, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            missing_origins_dropped=len(all_rows)-len(panel))
    manifest = dict(asof=str(ASOF), holdout_start=HOLDOUT, training_window_calendar_weeks=156, min_training=104,
        ridge=1.0, definitions=sorted(definitions,key=lambda d:d['number']), excluded=excluded,
        coverage=COVERAGE, cohorts=summaries, skipped_horizon=skipped,
        limitations=['Current source snapshots may include historical corrections/revisions',
            'Long-history VIX inputs before 2003-09-22 use retrospectively reconstructed current-definition VIX; those fits are not point-in-time valid',
            'Availability timestamps are lag-policy assumptions, not archived publication evidence',
            'ETF adjusted closes are present-day adjusted histories',
            'Holiday cases are excluded, so consecutive eligible weekly origins need not be 7 days apart',
            'Gold uses GLD; oil excludes nonpositive endpoints; thresholds fixed before inspecting model scores'])
    (ROOT/'data/panel_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(summaries,indent=2))


if __name__ == '__main__':
    build()
