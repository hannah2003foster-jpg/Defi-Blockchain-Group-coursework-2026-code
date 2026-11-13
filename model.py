"""Original one-factor ridge procedure. No data downloads or embedded market data."""
import numpy as np
import pandas as pd

FACTOR='equal_weight_relative_20'
CUTOFF=pd.Timestamp('2026-11-13T12:00:00Z')
WINDOW_WEEKS=156
MIN_TRAIN=104
RIDGE=1.0

def training_rows(panel,asof=CUTOFF):
    """Keep matured outcomes inside the original calendar-week training window."""
    panel=panel.copy()
    columns=['origin','target_available_at','baseline_available_at','available_'+FACTOR]
    for c in columns:panel[c]=pd.to_datetime(panel[c],utc=True,format='mixed')
    if panel.origin.duplicated().any():raise ValueError('Duplicate origins')
    train=panel[(panel.origin<asof)&(panel.origin>=asof-pd.Timedelta(weeks=WINDOW_WEEKS))&(panel.target_available_at<asof)].copy()
    if len(train)<MIN_TRAIN:raise ValueError('Fewer than 104 matured eligible training outcomes')
    if (train.baseline_available_at>train.origin).any() or (train['available_'+FACTOR]>train.origin).any():
        raise ValueError('A historical training input was unavailable at its origin')
    if (train[['baseline','actual']]<=0).any().any():raise ValueError('Nonpositive index level')
    if not np.isfinite(train[['baseline','actual','x_'+FACTOR]].to_numpy(float)).all():raise ValueError('Nonfinite input')
    return train

def fit_forecast(train,factor_value,baseline):
    """Fit ridge and convert the predicted return into index points.

    Returns and factor values are fractions: 0.01 is one percentage point.
    Only the standardised slope is penalised; the intercept is not.
    """
    x=train['x_'+FACTOR].to_numpy(float)
    y=(train.actual/train.baseline-1).to_numpy(float)
    mean=float(x.mean());sd=float(x.std(ddof=0));sd=1.0 if sd<1e-12 else sd
    z=(x-mean)/sd
    intercept=float(y.mean());slope=float(z@(y-intercept)/(z@z+RIDGE))
    standardised_current=(float(factor_value)-mean)/sd
    predicted_return=intercept+slope*standardised_current
    forecast=float(baseline)*(1+predicted_return)
    if not np.isfinite(forecast) or forecast<=0:raise ValueError('Invalid forecast')
    return {'factor':FACTOR,'training_observations':len(train),'training_calendar_weeks':WINDOW_WEEKS,
        'ridge_penalty':RIDGE,'unpenalised_intercept':True,'training_factor_mean':mean,'training_factor_sd_population':sd,
        'standardised_intercept':intercept,'standardised_slope':slope,'standardised_current_factor':standardised_current,
        'predicted_return':predicted_return,'forecast_unrounded':forecast,'forecast_two_decimals':f'{forecast:.2f}',
        'unchanged_price':float(baseline),'historical_mean_forecast':float(baseline)*(1+intercept),
        'forecast_at_factor_minus_one_training_sd':float(baseline)*(1+predicted_return-slope),
        'forecast_at_factor_plus_one_training_sd':float(baseline)*(1+predicted_return+slope)}

def factor_and_baseline(prices,asof=CUTOFF):
    """Calculate a 20-session lookback from the last 21 admissible price rows.

    This lookback differs from the six-session baseline-to-target return.
    Availability follows the project's next-New-York-midnight assumption.
    """
    prices=prices.copy();prices['date']=pd.to_datetime(prices.date)
    prices=prices.sort_values('date')
    if prices.date.duplicated().any():raise ValueError('Duplicate price date')
    available=pd.DatetimeIndex(prices.date+pd.Timedelta(days=1)).tz_localize('America/New_York').tz_convert('UTC')
    prices=prices.loc[available<=asof].dropna(subset=['gspc_close','rsp_adjusted_close','spy_adjusted_close'])
    if len(prices)<21:raise ValueError('Need at least 21 available price observations')
    if (prices[['gspc_close','rsp_adjusted_close','spy_adjusted_close']]<=0).any().any():raise ValueError('Nonpositive prices')
    first=prices.iloc[-21];last=prices.iloc[-1]
    x=float((last.rsp_adjusted_close/first.rsp_adjusted_close-1)-(last.spy_adjusted_close/first.spy_adjusted_close-1))
    return x,float(last.gspc_close),str(last.date.date())
