# S&P 500 coursework code and synthetic demonstration

**PUBLIC SAMPLE DATA IS SYNTHETIC. It contains no downloaded Yahoo/FRED observations and must not be used as evidence of historical accuracy or as the assessed forecast. The assessed final value is stated in the group report.**

This repository contains analysis code, running instructions and fabricated sample inputs. Downloaded market histories, derived historical panels, empirical forecast-by-forecast results, private calculation records and source archives are intentionally excluded. The group retains its original frozen evidence privately. The code-only publication follows the marking clarification reported to the group; that clarification is not a data-provider redistribution licence.

## Run the synthetic demonstration

Use Python 3.11 or later. From this directory:

```text
python -m pip install -r requirements.txt
python -B final_forecast.py --demo
python -B test_model.py
```

The demo reads synthetic/training.csv and synthetic/prices.csv and writes outputs/result.json. Every sample row and the output are labelled SYNTHETIC_NOT_REAL_MARKET_DATA. The dates illustrate the coursework calendar; the numeric observations were wholly fabricated using seed 832746. There are no market-data downloads or live API calls. Package installation may access the package registry.

To regenerate the examples, run python -B generate_synthetic.py. Its generator reads no real market files. Synthetic values and resulting example forecasts are unrelated to the empirical results in the report.

## The selected procedure

The single regression predictor is 20-day equal-weight relative return: RSP's 20-trading-session adjusted-close return minus SPY's equivalent. Yahoo ^GSPC daily Close is the sole historical S&P working series in the actual analysis; RSP/SPY adjusted closes are ancillary factor inputs.

The actual procedure uses Friday noon Europe/London origins and the following Friday 16:00 America/New_York targets. The latest daily baseline is normally the preceding Thursday close, and the original included common-panel cases span six trading-session returns. The private panel retains the original 29-factor common-sample and calendar exclusions even for the final one-factor fit.

The fixed settings are ridge penalty 1, unpenalised intercept, training-only population mean/SD standardisation, rolling 156 calendar weeks and at least 104 completed eligible outcomes. There is no activation rule, discretionary price adjustment or additional forecast shrinkage. The information cutoff remains 13 November 2026 at 12:00 London; the target is 20 November at 16:00 New York/21:00 London. The submission extension to 20:00 London does not extend the information boundary.

## Run with privately held authorised inputs

The public checkout alone does **not** reproduce the real historical tables or the assessed point forecast. Those calculations require the exact privately retained pre-cutoff inputs. The marking clarification allows code, instructions and synthetic examples instead of uploading the source data.

With access to an authorised local frozen snapshot:

```text
python -B final_forecast.py --snapshot /absolute/path/to/private_snapshot --output /private/output/result.json
python -B tools/reproduce_recent_research.py --snapshot /absolute/path/to/private_snapshot
```

On Windows, use a quoted Windows path. The snapshot must contain:

- data/source_manifest.json, with records containing status, file, sha256 and retrieval_completed_at;
- all raw files referenced by that manifest, retained privately;
- data/raw/yahoo_GSPC.json, yahoo_RSP.json and yahoo_SPY.json in Yahoo chart-response format;
- data/panels/expanded.csv, retaining the original eligible common panel;
- data/panel_manifest.json for the optional broader historical reconstruction, including definitions and skipped_horizon.

The final-model script verifies the private manifest hashes and rejects original source retrievals after the cutoff. It calculates the factor from adjusted ETF closes, selects matured rolling training outcomes, fits the unchanged ridge model and saves its arithmetic locally. It does not silently substitute newly downloaded or synthetic data when the private inputs are missing.

The expanded panel schema includes origin, baseline, baseline_date, baseline_available_at, actual, target_at, target_session_date, target_available_at, x_equal_weight_relative_20 and available_equal_weight_relative_20. The broader research code additionally needs all 29 original x_FACTOR and available_FACTOR columns. The optional panel-builder code uses COURSEWORK_PRIVATE_SNAPSHOT to point to a local private source directory. It makes no network calls; the supplied sample illustrates only the selected-factor schema and is not a real exchange-calendar reconstruction.

If obtaining independent inputs from providers under your own access rights, preserve their source, field, adjustment policy, calendar and availability metadata. A later provider download may contain revisions and cannot be represented as the group's pre-noon snapshot. Newly retrieved data must not be inserted into the submitted forecast. Do not commit provider data or derived real-history outputs to this repository.

## Limits and audit scope

Real historical studies used current-vintage provider histories and assumed availability policies, rather than certified archival publication timestamps. The empirical model search and overlapping retrospective daily reconstructions do not establish an independent predictive edge. The report contains the real validation results, final forecast and limitations; the synthetic demo establishes software execution only.

Tests check immature-label exclusion, unavailable-input rejection, agreement with a full-design ridge calculation, exclusion of post-cutoff price rows and explicit synthetic labelling. Prior private reproduction audits remain privately retained. They must not be described as results that can be independently reproduced from this public synthetic sample.

Only the code and wholly generated synthetic examples are supplied publicly. No third-party market-data licence is granted or implied. No earlier reserved-year candidate forecasts are supplied or generated by the demonstration.
