"""Reconstruct the already explored trailing year for hindsight diagnostics.

This does not produce a tradable selection rule. Model forecasts are trained historically,
but each weekly winning identity is chosen AFTER the realised close is known.
This does not select or calculate the group's final binding forecast.
"""
from collections import Counter
import hashlib
import itertools
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from backtest import fit_predict, load_panel, score, training_rows

import argparse
REPO = Path(__file__).resolve().parents[1]
_parser = argparse.ArgumentParser(description='Research reproduction requires privately supplied frozen inputs.')
_parser.add_argument('--snapshot', type=Path, required=True, help='Private local frozen snapshot directory; not downloaded by this script')
ROOT = _parser.parse_args().snapshot.resolve()
OUT = REPO / 'outputs/recent_research'
ASOF = pd.Timestamp('2026-11-13T12:00:00Z')
START = ASOF - pd.DateOffset(years=1)
# Legacy frozen comparator, not the selected equal_weight_relative_20 final model.
FROZEN = 'vol_20'

LABELS = {
 'return_1':'1-day return','return_5':'5-day momentum','return_20':'20-day momentum',
 'return_60':'60-day momentum','return_120':'120-day momentum','ma_gap_20':'20-day MA gap',
 'ma_gap_200':'200-day MA gap','rsi_14':'14-day RSI','vol_10':'10-day volatility',
 'vol_20':'20-day volatility','vol_ratio_10_60':'10/60-day volatility ratio',
 'downside_vol_20':'20-day downside volatility','vix_level':'VIX level','vix_change_5':'5-day VIX change',
 'vix_ma_gap_20':'VIX / 20-day average','vix_term':'VIX / VIX3M', 'vvix_level':'VVIX level',
 'equal_weight_relative_20':'20-day equal-weight relative return','small_cap_relative_20':'Small-cap relative return',
 'technology_relative_20':'Technology relative return','cyclical_defensive_20':'Discretionary / staples return',
 'yield2_change_20':'2-year yield change','yield10_change_20':'10-year yield change',
 'yield_curve':'10-year minus 2-year yield','real_yield':'10-year real yield',
 'real_yield_change_20':'Real yield change','dollar_return_20':'Dollar return',
 'oil_return_20':'Oil return','gold_proxy_return_20':'Gold ETF return',
 'unchanged':'Unchanged price','historical_mean':'Historical-average return'}


def label(name):
    return ' + '.join(LABELS[f] for f in name.split(' + '))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    panel_path = ROOT/'data/panels/expanded.csv'
    panel, factors = load_panel(panel_path)
    config = json.loads((ROOT/'data/panel_manifest.json').read_text())
    definitions = {d['name']:d for d in config['definitions']}
    cols = ['x_'+f for f in factors]
    models = {'unchanged':(), 'historical_mean':()}
    models.update({' + '.join(fs):fs for k in [1,2] for fs in itertools.combinations(factors,k)})
    wanted = panel[(panel.target_at > START) & (panel.target_available_at <= ASOF)].index
    records, weekly, inputs, top_three = [], [], [], []
    for i in wanted:
        row = panel.loc[i]
        train = training_rows(panel, i, 156)
        if len(train) < 104:
            raise ValueError('Insufficient historical training')
        assert train.target_available_at.max() <= row.origin
        x = train[cols].to_numpy(dtype=float)
        y = (train.actual/train.baseline-1).to_numpy()
        mean, sd = x.mean(axis=0), x.std(axis=0)
        sd = np.where(sd<1e-12,1,sd)
        z = (x-mean)/sd
        gram, cross = z.T@z, z.T@(y-y.mean())
        point = row[cols].to_numpy(dtype=float)
        model_rows = []
        for name, fs in models.items():
            ids = [factors.index(f) for f in fs]
            beta = np.linalg.lstsq(gram[np.ix_(ids,ids)]+np.eye(len(ids)),cross[ids],rcond=None)[0] if fs else np.array([])
            contributions = beta*((point[ids]-mean[ids])/sd[ids]) if fs else np.array([])
            base_return = 0.0 if name == 'unchanged' else float(y.mean())
            ret = base_return+float(contributions.sum())
            forecast = float(row.baseline*(1+ret))
            if forecast <= 0 or not np.isfinite(forecast):
                raise ValueError('Invalid forecast')
            error = float(abs(forecast-row.actual)/row.actual)
            record = dict(origin=str(row.origin), target_at=str(row.target_at),
                cutoff_date=str(row.origin.tz_convert('Europe/London').date()),
                target_date=str(row.target_at.tz_convert('America/New_York').date()),
                model=name, model_label=label(name), factor_count=len(fs), baseline=float(row.baseline),
                actual=float(row.actual), forecast=forecast, expected_return=ret,
                absolute_percentage_error=error, accuracy_marks=float(score(error*100)),
                n_train=len(train), training_last_outcome=str(train.target_available_at.max()),
                training_mean_return=base_return, contributions=json.dumps(dict(zip(fs,contributions.tolist()))),
                standardised_coefficients=json.dumps(dict(zip(fs,beta.tolist()))))
            model_rows.append(record)
            records.append(record)
        ranked = sorted([r for r in model_rows if r['factor_count']>0],
                        key=lambda r:(r['absolute_percentage_error'],r['factor_count'],r['model']))
        best = ranked[0]
        best_single = next(r for r in ranked if r['factor_count']==1)
        best_pair = next(r for r in ranked if r['factor_count']==2)
        all_best = min(model_rows,key=lambda r:(r['absolute_percentage_error'],r['factor_count'],r['model']))
        baseline = next(r for r in model_rows if r['model']=='unchanged')
        frozen = next(r for r in model_rows if r['model']==FROZEN)
        # Cross-check the legacy comparator with the separate full-design fit.
        direct, _ = fit_predict(train,row,(FROZEN,),1.0)
        assert abs(direct-frozen['expected_return']) < 1e-10
        weekly.append(dict(cutoff_date=best['cutoff_date'], target_date=best['target_date'], status='Included',
            baseline=float(row.baseline),actual=float(row.actual),actual_return=float(row.actual/row.baseline-1),
            hindsight_winner=best['model_label'], winner_id=best['model'], winner_forecast=best['forecast'],
            winner_return=best['expected_return'], winner_error=best['absolute_percentage_error'],
            winner_marks=best['accuracy_marks'], unchanged_error=baseline['absolute_percentage_error'],
            frozen_forecast=frozen['forecast'], frozen_error=frozen['absolute_percentage_error'],
            best_single=best_single['model_label'],single_error=best_single['absolute_percentage_error'],
            best_pair=best_pair['model_label'],pair_error=best_pair['absolute_percentage_error'],
            best_including_benchmarks=all_best['model_label'],
            models_within_0_1_percentage_point=sum(r['absolute_percentage_error']<=best['absolute_percentage_error']+.001 for r in ranked),
            economic_conditions_notes='',evidence_url='',evidence_publication_time=''))
        for rank, record in enumerate(ranked[:3],1):
            top_three.append(dict(cutoff_date=record['cutoff_date'],target_date=record['target_date'],
                rank=rank,model=record['model_label'],forecast=record['forecast'],actual=record['actual'],
                error=record['absolute_percentage_error'],marks=record['accuracy_marks'],
                forecast_return=record['expected_return']))
        fs = models[best['model']]
        for f in fs:
            j = factors.index(f)
            definition = definitions[f]
            inputs.append(dict(cutoff_date=best['cutoff_date'],target_date=best['target_date'],
                winning_model=best['model_label'],factor=LABELS[f],factor_id=f,
                observed_value=float(row['x_'+f]), training_mean=float(mean[j]),training_sd=float(sd[j]),
                standardised_value=float((point[j]-mean[j])/sd[j]),
                standardised_coefficient=json.loads(best['standardised_coefficients'])[f],
                return_contribution=json.loads(best['contributions'])[f],
                historical_mean_return=best['training_mean_return'],
                total_forecast_return=best['expected_return'],
                available_at=str(row['available_'+f]),formula=definition['formula']))
        # Raw contribution sums reconstruct the complete winning return.
        assert abs(sum(json.loads(best['contributions']).values())+best['training_mean_return']-best['expected_return'])<1e-12
    # Show excluded calendar weeks as explicit gaps rather than inventing results.
    included={r['cutoff_date'] for r in weekly}
    skip_origins={pd.Timestamp(r['origin']).tz_convert('Europe/London').date():r['reason'] for r in config['skipped_horizon']}
    for target_date in pd.date_range(START.date(), ASOF.date(), freq='W-FRI'):
        target_time=(target_date+pd.Timedelta(hours=16)).tz_localize('America/New_York').tz_convert('UTC')
        if target_time > ASOF:
            continue
        origin=(target_date-pd.Timedelta(days=7)+pd.Timedelta(hours=12)).tz_localize('Europe/London')
        d=str(origin.date())
        if d not in included:
            reason=skip_origins.get(origin.date(),'Missing common factor data')
            weekly.append(dict(cutoff_date=d,target_date=str(origin.date()+pd.Timedelta(days=7)),
                               status='Excluded: '+reason,economic_conditions_notes='',evidence_url='',evidence_publication_time=''))
    weekly.sort(key=lambda r:r['cutoff_date'])
    winners=Counter(r['winner_id'] for r in weekly if r['status']=='Included')
    counts=[dict(model=label(k),model_id=k,weekly_wins=v) for k,v in winners.most_common()]
    appearances=Counter(f for r in weekly if r['status']=='Included' for f in r['winner_id'].split(' + '))
    appearance_counts=[dict(factor=LABELS[k],factor_id=k,winning_model_appearances=v) for k,v in appearances.most_common()]
    guide=[dict(factor=LABELS[f],factor_id=f,formula=definitions[f]['formula'],
                sources=', '.join(definitions[f]['sources']),caveat=definitions[f]['caveat']) for f in factors]
    summary=dict(asof=str(ASOF),period=f'Targets after {START.date()} and completed by {ASOF.date()}',
        first_target=min(r['target_date'] for r in weekly),last_target=max(r['target_date'] for r in weekly),
        calendar_periods=len(weekly),included_periods=len(wanted),excluded_periods=len(weekly)-len(wanted),
        factor_models=len(models)-2,benchmarks=2,frozen_previously_selected_model=FROZEN,
        weekly_winners_are_hindsight=True,earlier_reserved_year_scored=False,
        unique_winning_models=len(counts),factor_model_beats_unchanged=sum(
            r['winner_error']<r['unchanged_error'] for r in weekly if r['status']=='Included'),
        model_counts=counts,factor_appearances=appearance_counts,
        panel_sha256=hashlib.sha256(panel_path.read_bytes()).hexdigest(),
        warning='The reviewed year is now explored and cannot be called an untouched test after model changes. Source-vintage limitations remain.')
    dataset=dict(summary=summary,weekly=weekly,top_three=top_three,winning_inputs=inputs,
                 factor_guide=guide,model_counts=counts,factor_counts=appearance_counts,
                 sources=json.loads((ROOT/'data/source_manifest.json').read_text())['records'])
    (OUT/'workbook_data.json').write_text(json.dumps(dataset,indent=2,allow_nan=False),encoding='utf-8')
    pd.DataFrame(records).to_csv(OUT/'all_model_forecasts.csv',index=False)
    pd.DataFrame(weekly).to_csv(OUT/'weekly_review.csv',index=False)
    (OUT/'review_manifest.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
