"""Run a labelled synthetic demo, or the unchanged model with private frozen inputs."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,hashlib
import pandas as pd
from model import CUTOFF,FACTOR,training_rows,fit_forecast,factor_and_baseline

ROOT=Path(__file__).resolve().parent

def private_inputs(snapshot):
    """Read frozen inputs after checking raw hashes and retrieval times."""
    manifest=json.loads((snapshot/'data/source_manifest.json').read_text())
    records=manifest['records']
    for record in records:
        if record['status']!='ok':raise ValueError('Source download failed')
        if pd.Timestamp(record['retrieval_completed_at'])>CUTOFF:raise ValueError('Post-cutoff source retrieval rejected')
        p=snapshot/Path(record['file'].replace('\\','/'))
        if hashlib.sha256(p.read_bytes()).hexdigest()!=record['sha256']:raise ValueError('Source hash mismatch')
    def yahoo(symbol,adjusted=False):
        data=json.loads((snapshot/f'data/raw/yahoo_{symbol}.json').read_text())['chart']['result'][0]
        dates=pd.to_datetime(data['timestamp'],unit='s',utc=True).tz_convert('America/New_York').normalize().tz_localize(None)
        values=data['indicators']['adjclose'][0]['adjclose'] if adjusted else data['indicators']['quote'][0]['close']
        return pd.Series(values,index=dates,dtype=float).dropna().sort_index()
    prices=pd.concat([yahoo('GSPC'),yahoo('RSP',True),yahoo('SPY',True)],axis=1,
        keys=['gspc_close','rsp_adjusted_close','spy_adjusted_close'],sort=True).dropna().rename_axis('date').reset_index()
    panel=pd.read_csv(snapshot/'data/panels/expanded.csv')
    return panel,prices,{'data_kind':'PRIVATE_FROZEN_MARKET_INPUTS','source_files_verified':len(records),
        'latest_original_source_retrieval':max(r['retrieval_completed_at'] for r in records)}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--demo',action='store_true',help='Use fabricated sample inputs only (default)')
    group.add_argument('--snapshot',type=Path,help='Local authorised private snapshot directory; never downloaded automatically')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/result.json')
    args=parser.parse_args()
    if args.snapshot:
        panel,prices,metadata=private_inputs(args.snapshot.resolve())
    else:
        panel=pd.read_csv(ROOT/'synthetic/training.csv');prices=pd.read_csv(ROOT/'synthetic/prices.csv')
        for frame in [panel,prices]:
            if not frame.data_kind.eq('SYNTHETIC_NOT_REAL_MARKET_DATA').all():raise ValueError('Demo input is not labelled synthetic')
        metadata={'data_kind':'SYNTHETIC_NOT_REAL_MARKET_DATA','warning':'DEMONSTRATION ONLY: not actual market observations, historical results or the assessed forecast.'}
    current,baseline,observation_date=factor_and_baseline(prices)
    train=training_rows(panel)
    result={**metadata,'information_cutoff':str(CUTOFF),'calculated_at_utc':datetime.now(timezone.utc).isoformat(),
        'factor_name':'20-day equal-weight relative return','factor_value':current,'baseline_observation_date':observation_date,
        **fit_forecast(train,current,baseline)}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(result,indent=2,allow_nan=False))

if __name__=='__main__':main()
