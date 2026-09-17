"""Fixed-reference weekly monitoring and pre-monitoring threshold calibration."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .data import complete_weeks, TARGET
from .metrics import feature_metrics
from .models import predict, error_metrics


@dataclass(frozen=True)
class Thresholds:
    covariate: float
    error: float
    quantile: float = 0.95


def weekly_report(reference, stream, model):
    records, features = [], []
    for start, batch in complete_weeks(stream):
        distribution = feature_metrics(reference, batch)
        distribution['week'] = start
        features.append(distribution)
        predictions = predict(model, batch)
        records.append({'week': start, 'n': len(batch), 'target_mean': batch[TARGET].mean(),
                        'drift_score': distribution.distance.mean(),
                        **error_metrics(batch[TARGET], predictions)})
    if not records:
        raise ValueError('No eligible weekly batches')
    return pd.DataFrame(records), pd.concat(features, ignore_index=True)


def calibrate(reference, calibration, model, quantile=0.95, bootstrap_weeks=200):
    report, _ = weekly_report(reference, calibration, model)
    # Null resampling uses calibration days only, with seeds disjoint from evaluation.
    # This yields an empirical seasonal baseline, not a guaranteed false-alarm rate.
    from .experiments import controlled_stream
    null = controlled_stream(calibration, seed=9001, scenario='no_change', weeks=bootstrap_weeks)
    null_report, _ = weekly_report(reference, null, model)
    cuts = Thresholds(float(np.quantile(null_report.drift_score, quantile, method='higher')),
                      float(np.quantile(null_report.mae, quantile, method='higher')), quantile)
    report.attrs['bootstrap_weeks'] = bootstrap_weeks
    return cuts, report


def mark_alerts(report, cuts, error_patience=2):
    out = report.copy()
    out['covariate_alert'] = out.drift_score > cuts.covariate
    out['error_exceedance'] = out.mae > cuts.error
    streak, previous, flags = 0, None, []
    for row in out.itertuples():
        if previous is not None and row.week - previous != pd.Timedelta(days=7):
            streak = 0
        streak = streak + 1 if row.error_exceedance else 0
        flags.append(streak >= error_patience)
        previous = row.week
    out['performance_alert'] = flags
    return out
