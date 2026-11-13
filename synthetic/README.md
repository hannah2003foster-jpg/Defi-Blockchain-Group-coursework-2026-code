# Synthetic example inputs

Every market-value observation in this directory is **fabricated**. These files contain no downloaded Yahoo/FRED prices and provide no evidence of historical forecast accuracy. Their example output is not the assessed forecast in the report.

| File | Purpose |
| --- | --- |
| `training.csv` | 130 fabricated training rows illustrating the selected-factor panel schema. |
| `prices.csv` | 45 fabricated business-day rows illustrating index Close and adjusted ETF-price inputs. |
| `SYNTHETIC_MANIFEST.json` | Generation seed, provenance statement and SHA-256 fingerprints of both CSV files. |

Every CSV row is labelled `SYNTHETIC_NOT_REAL_MARKET_DATA`. The dates illustrate coursework timing; they are not a verified exchange calendar. The sample's assumed predictor/outcome relationship is artificial and is not fitted to real markets.

From the repository root, run `python -B generate_synthetic.py` to regenerate the same examples with seed `832746`. The generator reads no real market files and makes no network calls.
