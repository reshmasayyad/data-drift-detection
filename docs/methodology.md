# Monitoring methodology

## Prediction and chronology

The initial training set has 6,442 observations. Model selection uses 743 October observations. Calibration has 1,460 November–December observations. Monitoring contains 8,734 hourly rows; two boundary days account for 48 excluded hours, leaving 8,686 evaluated hours in 52 full calendar-week intervals. Sparse observations are not interpolated. The 29 October week has lower observed coverage and is retained.

The mean baseline is compared against two boosting models with 15 and 31 leaves. October MAE selects 31. Model parameters and training are unchanged during calibration. An update during 2012 uses all labels available by the end of the current batch within the previous 180 days, including selection/calibration labels if still in the window. No future labels are used. All predictions for the trigger batch use the model that existed before its labels were revealed.

## Distribution score

Numeric features: normalized temperature, normalized feels-like temperature, humidity, wind speed. Categorical features: weather condition, working-day status. Calendar hour, weekday, month, season and holiday remain prediction features but are not distribution-score features.

The score is the mean of four reference-IQR-normalized Wasserstein distances and two total variation distances. This equal weighting is fixed in advance and heuristic; components are not identical statistical quantities. IQR scaling makes numeric distances interpretable relative to historical variation. If IQR is zero, use standard deviation; if that is also zero, use scale one.

PSI uses training quantile bins, deduplicated edges, infinite tails and 0.5 count smoothing. New extremes remain inside the histogram rather than disappearing. KS supplies a descriptive statistic and p-value for numerical features. Those p-values are not production alarm criteria because data are serially dependent and monitoring is repeated. The detector monitors marginal distributions and can miss changes in feature relationships.

## Thresholds and alarms

A 95th percentile of seven observed calibration weeks has little information about the range of plausible weekly mixes. Instead, generate 200 weeks from seven calibration days sampled independently with replacement per week (seed 9001). Full-day blocks retain intraday dependence. Each day has at least 22 observed hours; no individual rows are independently resampled. The empirical 95th-percentile input-score cutoff is 0.327716, and the weekly MAE cutoff is 46.5454 rentals/hour.

These are empirical baselines for the calibration season. Resampling breaks across-day dependence and can create unnatural calendar mixes. A held-out simulation of 20 seeds produces 6.5% no-change input alarms across all 40 batches; it does not validate a nominal 5% deployment guarantee.

Input alarms require one score exceedance. Error alarms require two consecutive adjacent calendar-week MAE exceedances. A missing eligible week resets the error streak. The natural-stream report shows all persistent alerts. The retraining policy resets its streak after an update and enforces four batches between updates.

## Known-change experiments

For each of 20 evaluation seeds, create a 40-week calibration-day-resampled stream. The last 20 weeks are affected by an intervention, except in the no-change control. Sensor corruption shifts reported temperature/feels-like temperature down by 0.20 and humidity up by 0.35, clipped to [0,1], leaving rental labels fixed. Conditional demand growth multiplies rental labels by 1.6 and rounds them, leaving all model features fixed. The derived label is intentionally not constrained to the original casual/registered split because those components are excluded from the model.

The same seeds yield paired baseline inputs. Consequently, demand-growth input-alarm sequences equal no-change input-alarm sequences. A background alert after onset is not evidence that the input detector identified the label intervention.

Report pre-onset and post-onset fractions of batches with alerts, fraction of runs with any post-onset alert, and conditional median delay. Non-alerting runs have missing delay and are retained in the run-alert fraction. Zero delay means an alert at the end of the first affected weekly batch. No-change has a nominal comparison index but no genuine change point.

## Retraining comparison

Every policy starts with the same model. Scheduled updates occur after every four evaluated batches. Error-triggered updates require two high-error weeks and a four-batch cooldown. The final batch is never followed by a charged fit because that update would not affect any evaluated predictions.

Errors are pooled over all hourly predictions. Each policy evaluates exactly the same hours. Counts exclude the initial fit. Hyperparameters, window length and trigger rules are not tuned on 2012 outcomes. Scheduled updates achieve MAE 51.46 with 12 fits, while error-triggered updates achieve 52.32 with 7 fits; neither is claimed to dominate in every application.

The code assumes instantaneous batch-end label access. Delayed labels, outages, cost-weighted errors and operational alert handling are extensions rather than evaluated behavior.
