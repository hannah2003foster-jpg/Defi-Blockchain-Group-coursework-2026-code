"""Generate wholly fabricated examples; no source files or network calls are read."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'synthetic';OUT.mkdir(exist_ok=True)
rng=np.random.default_rng(832746)
origins=pd.date_range(end='2026-10-30',periods=130,freq='W-FRI')
rows=[]
for day in origins:
    origin=(day+pd.Timedelta(hours=12)).tz_localize('Europe/London').tz_convert('UTC')
    baseline_day=day-pd.Timedelta(days=1);target_day=day+pd.Timedelta(days=7)
    x=float(rng.normal(0,.022));baseline=float(10000+rng.normal(0,300));ret=float(.001-.08*x+rng.normal(0,.016))
    rows.append({'data_kind':'SYNTHETIC_NOT_REAL_MARKET_DATA','origin':str(origin),
        'baseline':baseline,'baseline_date':str(baseline_day.date()),
        'baseline_available_at':str((baseline_day+pd.Timedelta(days=1)).tz_localize('America/New_York').tz_convert('UTC')),
        'actual':baseline*(1+ret),'target_at':str((target_day+pd.Timedelta(hours=16)).tz_localize('America/New_York').tz_convert('UTC')),
        'target_session_date':str(target_day.date()),'target_available_at':str((target_day+pd.Timedelta(days=1)).tz_localize('America/New_York').tz_convert('UTC')),
        'x_equal_weight_relative_20':x,'available_equal_weight_relative_20':str((baseline_day+pd.Timedelta(days=1)).tz_localize('America/New_York').tz_convert('UTC'))})
pd.DataFrame(rows).to_csv(OUT/'training.csv',index=False)
dates=pd.bdate_range(end='2026-11-12',periods=45)
prices=pd.DataFrame({'date':dates,'gspc_close':10000*np.exp(np.cumsum(rng.normal(0,.005,len(dates)))),
    'rsp_adjusted_close':200*np.exp(np.cumsum(rng.normal(0,.004,len(dates)))),
    'spy_adjusted_close':100*np.exp(np.cumsum(rng.normal(0,.004,len(dates)))),
    'data_kind':'SYNTHETIC_NOT_REAL_MARKET_DATA'})
prices.to_csv(OUT/'prices.csv',index=False)
manifest={'data_kind':'SYNTHETIC_NOT_REAL_MARKET_DATA','seed':832746,'source':'Wholly fabricated by NumPy random generation; no Yahoo, FRED or other market history copied.',
    'purpose':'Demonstrates input schema and software execution only. Not the empirical coursework analysis.',
    'files':{n:hashlib.sha256((OUT/n).read_bytes()).hexdigest() for n in ['training.csv','prices.csv']}}
(OUT/'SYNTHETIC_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print('Generated labelled synthetic inputs only.')
