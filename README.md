# S&P 500 coursework: code and synthetic examples

This repository contains the forecasting code, running instructions and **clearly labelled synthetic sample data**. The group report contains the assessed forecast and the real historical results.

Downloaded Yahoo/FRED histories, derived historical datasets and private calculation records are excluded from public GitHub. The group retains its frozen evidence privately. This follows the marking clarification reported to the group; it does not grant permission to redistribute provider data.

## Start here

| File or folder | Purpose |
| --- | --- |
| [`final_forecast.py`](final_forecast.py) | Runs the final single-factor procedure using synthetic examples or private frozen inputs. |
| [`model.py`](model.py) | Calculates the selected factor, filters completed training outcomes and fits the ridge model. |
| [`test_model.py`](test_model.py) | Checks key fitting, timing and synthetic-labelling behaviour. |
| [`synthetic/`](synthetic/) | Fabricated inputs, their explanation and file fingerprints. |
| [`generate_synthetic.py`](generate_synthetic.py) | Regenerates the fabricated examples without reading real market data. |
| [`tools/`](tools/) | Supporting code for the earlier 29-factor single/pair research; requires private inputs. |
| [`requirements.txt`](requirements.txt) | Exact NumPy and pandas versions used for the code. |
| [`CODE_MANIFEST.json`](CODE_MANIFEST.json) | SHA-256 fingerprints of the supplied code and example files. |

The Git settings files control file handling and exclude common private-data/output paths from ordinary commits. This repository contains no report draft or raw market-data archive.

## Run the synthetic demonstration

Download using **Code → Download ZIP**, extract it, and open a terminal in the folder containing `final_forecast.py`. Use Python 3.11 or later with the dependencies below:

```text
python -m pip install -r requirements.txt
python -B final_forecast.py --demo
python -B test_model.py
```

The demonstration reads `synthetic/training.csv` and `synthetic/prices.csv`, prints an example calculation and saves `outputs/result.json`. Every input row and the output are labelled `SYNTHETIC_NOT_REAL_MARKET_DATA`. **The example output is not the assessed forecast and does not reproduce the report's real historical results.**

All numerical observations are fabricated using random seed `832746`. The dates illustrate the input schema, not a verified exchange calendar. The scripts make no market-data downloads or live API calls; installing dependencies may access the package registry.

To regenerate the same examples, run `python -B generate_synthetic.py` from the repository root.

## The selected model and its two time windows

The single predictor is **20-day equal-weight relative return**:

```text
(RSP adjusted close / RSP adjusted close 20 trading sessions earlier - 1)
  minus
(SPY adjusted close / SPY adjusted close 20 trading sessions earlier - 1)
```

RSP and SPY supply adjusted ETF prices for the factor. Yahoo `^GSPC` daily **Close** is the sole S&P 500 working series for index baselines and outcomes. The ETFs are not the forecast target.

| Period | Meaning |
| --- | --- |
| 20 trading sessions | Historical lookback used to calculate the predictor. |
| Friday noon London to the following Friday at 16:00 New York | Information cutoff and target for each coursework-style forecast. |
| Six trading sessions | Baseline-to-target return in included historical cases: normally the preceding Thursday's close to the following Friday's close. Holiday or missing-session mismatches are excluded. |

The final cutoff is **13 November 2026 at 12:00 London/GMT**. The target is **20 November 2026 at 16:00 New York / 21:00 London**. The submission extension to 20:00 London on 13 November does not extend the information cutoff.

Fixed fitting settings:

- one predictor and an unpenalised intercept;
- ridge penalty `1` on the standardised slope;
- population mean/standard-deviation scaling learned from training rows only;
- a rolling window of `156` calendar weeks and at least `104` completed eligible outcomes;
- original 29-factor common-sample and calendar exclusions, retained for comparability in the final one-factor fit.

No activation rule, discretionary price adjustment or additional forecast shrinkage is applied.

## Reproduce the real calculation with private inputs

The public checkout alone cannot reproduce the real tables or assessed point forecast. Those calculations require the exact privately retained pre-cutoff snapshot.

With authorised access to a local copy:

```text
python -B final_forecast.py --snapshot /absolute/path/to/private_snapshot --output /private/output/result.json
python -B tools/reproduce_recent_research.py --snapshot /absolute/path/to/private_snapshot
```

On Windows, replace the example paths with quoted Windows paths. The first command runs the selected final model. The second reconstructs the explored trailing-year single/pair comparisons and saves private results under `outputs/recent_research/`; its weekly winners are hindsight diagnostics.

Required private inputs:

| Path | Content |
| --- | --- |
| `data/source_manifest.json` | Records containing `status`, `file`, `sha256` and `retrieval_completed_at`. |
| Raw files referenced by the manifest | Original downloaded bytes, retained privately. |
| `data/raw/yahoo_GSPC.json`, `yahoo_RSP.json`, `yahoo_SPY.json` | Yahoo chart responses for the index and adjusted ETF prices. |
| `data/panels/expanded.csv` | Original eligible common panel. |
| `data/panel_manifest.json` | Factor definitions and `skipped_horizon` records for broader reconstruction. |

The final-model script verifies raw-file hashes and rejects original source retrievals after the cutoff. It does not download replacements or silently substitute synthetic inputs when private data is missing.

The final-model panel requires `origin`, `baseline`, `baseline_date`, `baseline_available_at`, `actual`, `target_at`, `target_session_date`, `target_available_at`, `x_equal_weight_relative_20` and `available_equal_weight_relative_20`. Timestamps carry explicit time-zone offsets. Broader research additionally requires all 29 original `x_FACTOR` and `available_FACTOR` columns.

### Supporting research tools

- `tools/backtest.py`: general chronological single/pair comparison engine. Running a new search is not required for the submitted forecast.
- `tools/reproduce_recent_research.py`: reconstructs the already explored trailing year. `FROZEN = 'vol_20'` identifies the retained 20-day-volatility comparator, **not the final selected factor**. Saved research errors are fractions (`0.01` means 1%); marks convert them to percentage points before scoring.
- `tools/build_private_panels.py`: constructs historical panels from saved sources. It reads `COURSEWORK_PRIVATE_SNAPSHOT` and writes panels and their manifest inside that directory. Rebuild in a separate working copy to preserve original frozen evidence. The optional `COURSEWORK_ASOF` defaults to the noon cutoff. Rebuilding is unnecessary when using the existing panel.

Independent provider downloads can contain revisions and cannot be represented as the group's pre-noon snapshot. Preserve source, field, adjustment, calendar and availability metadata. Do not insert later data into the submitted forecast or commit real provider histories and derived results publicly.

## Checks and limitations

Tests cover unfinished-outcome exclusion, unavailable-input rejection, agreement with a full-design ridge calculation, post-cutoff price exclusion and synthetic labelling.

Historical studies use downloaded provider histories and assumed availability rules, not certified archival publication timestamps. Repeated model search and overlapping reconstructed daily forecasts do not establish an independent predictive advantage. The report states the real results and limitations.

The synthetic demonstration establishes software execution only. Private reproduction audits are retained separately and are not results reproducible from public examples. No earlier reserved-year candidate forecasts are supplied or generated by the demonstration. No third-party market-data licence is granted or implied.
