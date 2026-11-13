"""Meaningful software checks using fabricated inputs only."""
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from model import CUTOFF,FACTOR,training_rows,fit_forecast,factor_and_baseline
ROOT=Path(__file__).resolve().parent

class ModelChecks(unittest.TestCase):
    def setUp(self):self.panel=pd.read_csv(ROOT/'synthetic/training.csv')
    def test_immature_and_post_origin_rows_do_not_affect_fit(self):
        train=training_rows(self.panel);base=fit_forecast(train,.012,10000)
        future=self.panel.iloc[-1].copy();future['origin']='2026-11-06T12:00:00Z';future['target_available_at']='2026-11-14T05:00:00Z';future['actual']=1e9
        candidate=training_rows(pd.concat([self.panel,pd.DataFrame([future])],ignore_index=True))
        self.assertEqual(base,fit_forecast(candidate,.012,10000))
    def test_unavailable_training_factor_is_rejected(self):
        p=self.panel.copy();p.loc[0,'available_'+FACTOR]='2026-11-13T13:00:00Z'
        with self.assertRaises(ValueError):training_rows(p)
    def test_ridge_matches_full_design_fit(self):
        train=training_rows(self.panel);r=fit_forecast(train,.012,10000)
        x=train['x_'+FACTOR].to_numpy();z=(x-x.mean())/x.std(ddof=0);y=(train.actual/train.baseline-1).to_numpy()
        a=np.column_stack([np.ones(len(x)),z]);coef=np.linalg.lstsq(a.T@a+np.diag([0.,1.]),a.T@y,rcond=None)[0]
        self.assertAlmostEqual(r['standardised_intercept'],coef[0],places=12)
        self.assertAlmostEqual(r['standardised_slope'],coef[1],places=12)
    def test_post_cutoff_price_is_ignored(self):
        prices=pd.read_csv(ROOT/'synthetic/prices.csv');before=factor_and_baseline(prices)
        future=prices.iloc[-1].copy();future['date']='2026-11-13';future['gspc_close']=1e9
        self.assertEqual(before,factor_and_baseline(pd.concat([prices,pd.DataFrame([future])],ignore_index=True)))
    def test_samples_are_explicitly_synthetic(self):
        for file in ['prices.csv','training.csv']:
            self.assertTrue(pd.read_csv(ROOT/'synthetic'/file).data_kind.eq('SYNTHETIC_NOT_REAL_MARKET_DATA').all())

if __name__=='__main__':unittest.main()
