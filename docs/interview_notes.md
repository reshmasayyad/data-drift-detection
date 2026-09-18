# Explaining the project

A concise resume description supported by the results:

> Built a chronological drift-monitoring pipeline for 17,379 hourly bike-demand records using KS, PSI, Wasserstein and categorical distribution diagnostics; evaluated known-change simulations and reduced monitoring-year MAE by 43.1% with seven performance-triggered model updates.

The 43.1% reduction compares the triggered policy with a frozen model on 8,686 eligible 2012 hours. Scheduled updates achieve a slightly larger 44.0% reduction with 12 updates. Present both when discussing the policy comparison.

## Questions to be ready for

**What is data drift?** A change in the distribution of inputs. Model error may or may not increase. Conditional-demand changes can also raise error without changing input distributions.

**Why separate monitoring inputs and errors?** Input monitoring is available when labels are delayed. Error monitoring measures actual degradation but needs targets. The paired growth experiment demonstrates their different information.

**How did you avoid leakage?** The rental components that sum to the target are excluded. Data splits are chronological. October selects the model, November–December calibrate thresholds, and 2012 is evaluated afterward. Each weekly prediction precedes using that week's labels for updates.

**Why not just use a KS p-value below 0.05?** Hourly observations are dependent, large samples can flag small differences, and repeated feature tests produce multiple opportunities for an alert. The pipeline uses effect-size scores with empirical calibration instead. Its empirical thresholds still have limitations.

**Why is the calibration season a limitation?** A winter baseline does not represent every future season. Day-block resampling retains intraday structure but breaks dependence across days. Seasonal reference distributions are a reasonable extension.

**What happens during retraining?** After scoring a completed labeled batch, the policy may fit a new model using the last 180 days of available data. The new model predicts the next batch. Four-batch cooldown prevents consecutive redundant updates.

**Is this forecasting?** It estimates demand conditional on observed current-hour weather. Future-hour forecasting would require available weather forecasts or lagged predictors and a different availability audit.

**What can this detector miss?** Changes in feature relationships that leave marginal distributions unchanged, performance shifts while labels are absent, or unobserved outages. Missingness monitoring and multivariate diagnostics would extend it.

**What did the experiments show?** Sensor bias changes input distances; label growth leaves those distances unchanged. Triggered retraining uses fewer updates than a fixed four-week schedule, with slightly higher MAE in this particular dataset. There is no ground-truth drift annotation for the natural stream.
