# MFIP Self Support Index combined model prototype

This runnable prototype implements the three proposed components:

1. **Hierarchical risk adjustment:** penalized logistic models with partially pooled service area intercepts, an expected rate, and a simulated predictive range for the three year Self Support Index.
3. **Multiple outcomes:** separate models for sustained cash exit, employment, positive earnings, earnings among earners, and return to assistance. No composite score is constructed.
4. **Absolute goal:** a transparent user supplied percentage is displayed beside the expected range. It is illustrative and has no PIP or funding effect.

The numerical example is **synthetic**. It does not reproduce Minnesota's official Self Support Index methodology or classification. The 2020 model and the 2017 resampling procedure have different specifications; this code is an alternative research prototype.

## Run

```bash
python generate_example.py
python mfip_model.py --input synthetic_mfip.csv --output results --absolute-goal 0.70
```

Dependencies: Python 3.10+, numpy, pandas, scipy, scikit-learn. The outputs are `area_summary.csv`, `validation.csv`, and `model_diagnostics.json`.

## Streamlit Cloud deployment

Put the files in this folder at the root of the GitHub repository. In Streamlit Cloud, set **Main file path** to `streamlit_app.py`. The `requirements.txt` file must be at that same repository root; Streamlit Cloud installs scipy and the other packages from it. Do not set `mfip_model.py` as the main file: that file is the command-line model engine, not the app.

For a local browser session run `streamlit run streamlit_app.py`. Without an upload, the app displays precomputed synthetic results. The uploaded CSV follows `input_schema.csv`.

For real data, provide one row per adult in a baseline quarter, with at least the fields listed in `input_schema.csv`. Use an approved, access-controlled environment. Identifiers can be hashed stable IDs. The model groups annual validation by cohort year; holdout years must occur after training years. Data should include enough service areas and cohorts for estimation. Suppress small cells before sharing outputs.

## Outcome definitions for this prototype

- `ssi_success_3y`: official-compatible binary flag: off cash for the entire three year follow-up quarter or working at least 30 hours per week under approved program rules. Upstream production logic must enforce sanction and time-limit exclusions.
- `sustained_exit_12m`: no cash assistance for 12 consecutive months by follow-up. This window is provisional.
- `employed_3y`: employed in the three year follow-up quarter.
- `earnings_3y`: nonnegative, inflation-adjusted earnings in that quarter.
- `returned_cash_3y`: return to cash within three years after an eligible exit, with missing values for people who never exited. This is a descriptive binary endpoint, not a time-to-event model.

The script does not reconstruct these endpoints from raw MAXIS events. This definition and earnings coverage need review with MFIP program staff before operational use.

## Model and interpretation

Each binary model estimates a pooled logistic regression with a regularized area intercept; the penalty shrinks small-area effects toward zero. The **expected range excludes the fitted area intercept**, so the comparison is based on participant mix and baseline conditions rather than folding the area's observed residual into its own benchmark. A trained probability model predicts each person's outcome; Monte Carlo Bernoulli draws yield a service area distribution of possible observed rates. Its 2.5th and 97.5th percentiles are a *conditional predictive interval* and do not capture coefficient estimation uncertainty. Consequently, the intervals must not be adopted for PIPs without cluster-aware backtesting and recalibration.

The dashboard independently reports (a) observed and expected SSI, (b) companion observed and expected outcomes, and (c) whether observed SSI meets the chosen absolute goal. If observed SSI is below the expected interval, the row is labelled `Below`; above is `Above`; otherwise `Within`. These labels are **prototype comparisons**, not official HSPM decisions.

## Validation and limits

Temporal validation trains on earlier cohort years and scores the latest year. It reports Brier and calibration for binary outcomes and MAE for earnings among earners. Confirm coverage for each area size and population group before any use. Baseline predictors must be measured before the intervention and reviewed for provider influence; child care supply and caseload history may be partly endogenous. Neither a good predictive score nor a county residual establishes causation.

The prototype estimates earnings with a two-part model: probability of positive earnings and log earnings conditional on earnings being positive. It does not yet implement a longitudinal event-history model; returning to assistance is modelled as a three year binary flag. A production model should use dates of exit and return with censoring.

## References

- [Minnesota HSPM MFIP/DWP overview](https://www.dhs.state.mn.us/main/idcplg?IdcService=GET_FILE&allowInterrupt=1&dDocName=MNDHS-072838&dID=160285)
- [Minnesota Self Support Index methodology update](https://edocs.dhs.state.mn.us/lfserver/Public/DHS-6724A-ENG)
- [Minnesota Self Support Index 2020 update](https://www.dhs.state.mn.us/main/idcplg?IdcService=GET_FILE&Rendition=Primary&RevisionSelectionMethod=LatestReleased&allowInterrupt=1&dDocName=DHS-321961&noSaveAs=1)
