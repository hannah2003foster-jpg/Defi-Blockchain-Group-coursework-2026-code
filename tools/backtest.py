"""Chronological single/pair research using supplied availability metadata.

Historical panels approximate publication timing with recorded lag policies;
they are not certified point-in-time archives. The final selected-factor
calculation is run separately through final_forecast.py.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd


TIME_COLUMNS = ['origin', 'baseline_available_at', 'target_at', 'target_available_at']
REQUIRED = TIME_COLUMNS + ['baseline', 'actual', 'baseline_date', 'target_session_date']


def score(error_pct):
    """Unrounded coursework mark; round only for presentation."""
    return np.maximum(0.0, 50.0 - 5.0 * np.asarray(error_pct))


def load_panel(path):
    df = pd.read_csv(path)
    missing = set(REQUIRED) - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    factors = sorted(c[2:] for c in df.columns if c.startswith('x_'))
    if not factors:
        raise ValueError('Need at least one x_FACTOR column')
    for name in factors:
        if f'available_{name}' not in df:
            raise ValueError(f'Missing available_{name}')
    for col in TIME_COLUMNS + [f'available_{f}' for f in factors]:
        if df[col].isna().any():
            raise ValueError(f'Missing timestamp in {col}')
        if not df[col].astype(str).str.contains(r'(?:Z|[+-]\d\d:\d\d)$', regex=True).all():
            raise ValueError(f'{col} must have an explicit timezone offset')
        df[col] = pd.to_datetime(df[col], utc=True, format='mixed')
    if df.origin.duplicated().any() or not df.origin.is_monotonic_increasing:
        raise ValueError('Origins must be unique and sorted')
    london = df.origin.dt.tz_convert('Europe/London')
    ny = df.target_at.dt.tz_convert('America/New_York')
    if not ((london.dt.dayofweek == 4) & (london.dt.hour == 12)
            & (london.dt.minute == 0) & (london.dt.second == 0)).all():
        raise ValueError('Origins must be Friday noon London time')
    next_friday = london.dt.date.map(lambda d: d + pd.Timedelta(days=7))
    if not ((ny.dt.date == next_friday) & (ny.dt.hour == 16)
            & (ny.dt.minute == 0) & (ny.dt.second == 0)).all():
        raise ValueError('Target must be next Friday at 16:00 New York time')
    if (df.baseline_available_at > df.origin).any():
        raise ValueError('Baseline uses future information')
    if (df.target_available_at < df.target_at).any():
        raise ValueError('Target cannot be known before target time')
    baseline_dates = pd.to_datetime(df.baseline_date).dt.date
    target_dates = pd.to_datetime(df.target_session_date).dt.date
    if (baseline_dates >= london.dt.date).any():
        raise ValueError('Daily-close baseline must predate the Friday origin')
    if (target_dates > ny.dt.date).any() or (target_dates < baseline_dates).any():
        raise ValueError('Invalid target session date')
    # Common complete-case panel: every model competes on exactly the same origins.
    for col in ['baseline', 'actual'] + [f'x_{f}' for f in factors]:
        df[col] = pd.to_numeric(df[col], errors='raise')
        if not np.isfinite(df[col]).all():
            raise ValueError(f'Non-finite values in {col}; build a documented common sample')
    if (df[['baseline', 'actual']] <= 0).any().any():
        raise ValueError('Index levels must be positive')
    for f in factors:
        if (df[f'available_{f}'] > df.origin).any():
            raise ValueError(f'{f} uses future information')
    return df, factors


def fit_predict(train, row, factors, ridge):
    """Predict simple horizon return. Standardisation is learned within each fit."""
    y = (train.actual / train.baseline - 1).to_numpy()
    if not factors:
        return float(y.mean()), {'intercept': float(y.mean()), 'weights': {}}
    cols = [f'x_{f}' for f in factors]
    x = train[cols].to_numpy(dtype=float)
    mean, sd = x.mean(axis=0), x.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    z = (x - mean) / sd
    design = np.column_stack([np.ones(len(x)), z])
    penalty = np.diag([0.0] + [ridge] * len(factors))
    coef = np.linalg.lstsq(design.T @ design + penalty, design.T @ y, rcond=None)[0]
    weights = coef[1:] / sd
    intercept = float(coef[0] - weights @ mean)
    prediction = intercept + float(row[cols].to_numpy(dtype=float) @ weights)
    return prediction, {'intercept': intercept, 'weights': dict(zip(factors, weights.tolist()))}


def training_rows(df, i, window):
    """Window is calendar weeks; skipped holiday origins must not lengthen it."""
    origin = df.origin.iloc[i]
    previous = df.iloc[:i]
    return previous.loc[(previous.target_available_at <= origin)
                        & (previous.origin >= origin - pd.Timedelta(weeks=window))]


def run(df, factors, min_train=104, holdout=104, window=156, ridge=1.0,
        evaluate_holdout=False, holdout_start=None):
    if min_train < 2 or holdout < 2 or window < min_train or ridge < 0:
        raise ValueError('Invalid training, holdout, window or ridge configuration')
    # A training outcome must have been published BEFORE this forecast origin.
    eligible = [i for i in range(len(df)) if len(training_rows(df, i, window)) >= min_train]
    if len(eligible) < holdout + 26:
        raise ValueError('Need at least 26 development forecasts plus the requested holdout')
    if holdout_start is None:
        split = eligible[-holdout]
    else:
        boundary = pd.Timestamp(holdout_start)
        if boundary.tzinfo is None:
            raise ValueError('holdout_start requires a timezone')
        matches = [i for i in eligible if df.origin.iloc[i] >= boundary]
        if not matches:
            raise ValueError('No eligible holdout origins after fixed boundary')
        split = matches[0]
    holdout_start = df.origin.iloc[split]
    # Embargo development forecasts whose outcome is unknown at holdout start.
    development = [i for i in eligible if i < split
                   and df.target_available_at.iloc[i] < holdout_start]
    if len(development) < 26:
        raise ValueError('Need 26 development origins after the boundary embargo')
    models = {'unchanged': (), 'historical_mean': ()}
    for size in (1, 2):
        for fs in itertools.combinations(factors, size):
            models[' + '.join(fs)] = fs

    def predict(i, name, fs, phase):
        row = df.iloc[i]
        train = training_rows(df, i, window)
        if name == 'unchanged':
            ret, coefficients = 0.0, {'intercept': 0.0, 'weights': {}}
        else:
            ret, coefficients = fit_predict(train, row, fs, ridge)
        forecast = float(row.baseline * (1 + ret))
        if not np.isfinite(forecast) or forecast <= 0:
            raise ValueError(f'Invalid forecast for {name} at {row.origin}; review model')
        error = 100.0 * abs(forecast - row.actual) / row.actual
        return dict(phase=phase, model=name, origin=str(row.origin), target_at=str(row.target_at),
                    baseline=float(row.baseline), actual=float(row.actual), forecast=forecast,
                    expected_return=ret, error_pct=error, marks=float(score(error)),
                    n_train=len(train), train_last_outcome=str(train.target_available_at.max()),
                    coefficients=json.dumps(coefficients))

    # Compute sufficient statistics once per origin. All candidates share this training set.
    # This avoids hundreds of repeated dataframe operations for each historical week.
    records = []
    factor_index = {f: j for j, f in enumerate(factors)}
    cols = [f'x_{f}' for f in factors]
    for i in development:
        row = df.iloc[i]
        train = training_rows(df, i, window)
        x = train[cols].to_numpy(dtype=float)
        y = (train.actual / train.baseline - 1).to_numpy()
        mean, sd = x.mean(axis=0), x.std(axis=0)
        sd = np.where(sd < 1e-12, 1.0, sd)
        z = (x-mean)/sd
        gram = z.T @ z
        cross = z.T @ (y-y.mean())
        point = row[cols].to_numpy(dtype=float)
        for name, fs in models.items():
            intercept = 0.0 if name == 'unchanged' else float(y.mean())
            weights = {}
            if fs:
                js = [factor_index[f] for f in fs]
                matrix = gram[np.ix_(js, js)] + ridge * np.eye(len(js))
                beta = np.linalg.lstsq(matrix, cross[js], rcond=None)[0] / sd[js]
                intercept -= float(beta @ mean[js])
                weights = dict(zip(fs, beta.tolist()))
                ret = intercept + float(beta @ point[js])
            else:
                ret = intercept
            forecast = float(row.baseline*(1+ret))
            if not np.isfinite(forecast) or forecast <= 0:
                raise ValueError(f'Invalid forecast for {name} at {row.origin}')
            error = 100*abs(forecast-row.actual)/row.actual
            records.append(dict(phase='development', model=name, origin=str(row.origin),
                target_at=str(row.target_at), baseline=float(row.baseline), actual=float(row.actual),
                forecast=forecast, expected_return=ret, error_pct=error, marks=float(score(error)),
                n_train=len(train), train_last_outcome=str(train.target_available_at.max()),
                coefficients=json.dumps({'intercept':intercept,'weights':weights})))
    predictions = pd.DataFrame(records)
    leaderboard = summarise(predictions)
    candidates = leaderboard.loc[~leaderboard.model.isin(['unchanged', 'historical_mean'])]
    winner = candidates.sort_values(['mean_marks', 'mean_error_pct', 'model'],
                                   ascending=[False, True, True]).iloc[0].model
    # Freeze winner from development. Do not generate a holdout ranking of all candidates.
    if evaluate_holdout:
        for i in eligible:
            if i < split:
                continue
            for name in ['unchanged', 'historical_mean', winner]:
                records.append(predict(i, name, models[name], 'holdout'))
    return pd.DataFrame(records), winner, split


def summarise(predictions):
    base = predictions.loc[predictions.model == 'unchanged'].set_index(['phase', 'origin'])
    rows = []
    for (phase, name), group in predictions.groupby(['phase', 'model']):
        reference = np.array([base.loc[(phase, o), 'marks'] for o in group.origin])
        rows.append(dict(phase=phase, model=name, n=len(group),
                         mean_marks=group.marks.mean(), mean_error_pct=group.error_pct.mean(),
                         median_error_pct=group.error_pct.median(),
                         p90_error_pct=group.error_pct.quantile(.9),
                         worst_error_pct=group.error_pct.max(),
                         zero_mark_rate=float((group.marks == 0).mean()),
                         mean_mark_gain=float((group.marks.to_numpy() - reference).mean()),
                         beat_unchanged_rate=float((group.marks.to_numpy() > reference).mean())))
    return pd.DataFrame(rows).sort_values(['phase', 'mean_marks'], ascending=[True, False])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('panel', type=Path)
    parser.add_argument('--output', type=Path, default=Path('results'))
    parser.add_argument('--min-train', type=int, default=104)
    parser.add_argument('--holdout', type=int, default=104)
    parser.add_argument('--holdout-start', help='Fixed offset-aware boundary shared across panels')
    parser.add_argument('--evaluate-holdout', action='store_true',
                        help='Explicitly open the final test; do not use during exploration')
    parser.add_argument('--window', type=int, default=156, help='Rolling training window in calendar weeks')
    parser.add_argument('--ridge', type=float, default=1.0)
    args = parser.parse_args()
    df, factors = load_panel(args.panel)
    predictions, winner, split = run(df, factors, args.min_train, args.holdout, args.window,
                                     args.ridge, args.evaluate_holdout, args.holdout_start)
    args.output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output / 'predictions.csv', index=False)
    summarise(predictions).to_csv(args.output / 'leaderboard.csv', index=False)
    manifest = dict(input_sha256=hashlib.sha256(args.panel.read_bytes()).hexdigest(),
                    factors=factors, selected_on_development=winner,
                    holdout_start=str(df.origin.iloc[split]),
                    settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                    note='Exploratory research engine. Availability metadata requires source verification.')
    (args.output / 'run_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
