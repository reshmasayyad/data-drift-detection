"""Distribution diagnostics; p-values are descriptive under temporal dependence."""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

MONITORED_NUMERIC = ['temp', 'atemp', 'hum', 'windspeed']
MONITORED_CATEGORICAL = ['weathersit', 'workingday']


def _values(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or not np.isfinite(arr).all():
        raise ValueError('Metric inputs must be nonempty and finite')
    return arr


def population_stability_index(reference, current, bins=10):
    """Reference-quantile bins with open tails and additive smoothing."""
    ref, cur = _values(reference), _values(current)
    interior = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)[1:-1]))
    edges = np.r_[-np.inf, interior, np.inf]
    p = np.histogram(ref, bins=edges)[0].astype(float) + 0.5
    q = np.histogram(cur, bins=edges)[0].astype(float) + 0.5
    p, q = p / p.sum(), q / q.sum()
    return float(np.sum((q - p) * np.log(q / p)))


def normalized_wasserstein(reference, current):
    ref, cur = _values(reference), _values(current)
    scale = np.quantile(ref, 0.75) - np.quantile(ref, 0.25)
    if scale <= 1e-12:
        scale = np.std(ref)
    if scale <= 1e-12:
        scale = 1.0
    return float(wasserstein_distance(ref, cur) / scale)


def total_variation(reference, current):
    if len(reference) == 0 or len(current) == 0:
        raise ValueError('Empty categorical inputs')
    p, q = pd.Series(reference).value_counts(normalize=True), pd.Series(current).value_counts(normalize=True)
    support = p.index.union(q.index)
    return float(0.5 * np.abs(p.reindex(support, fill_value=0) - q.reindex(support, fill_value=0)).sum())


def feature_metrics(reference, current):
    rows = []
    for feature in MONITORED_NUMERIC:
        test = ks_2samp(reference[feature], current[feature])
        rows.append({'feature': feature, 'distance': normalized_wasserstein(reference[feature], current[feature]),
                     'ks_statistic': float(test.statistic), 'ks_pvalue': float(test.pvalue),
                     'psi': population_stability_index(reference[feature], current[feature]), 'type': 'numeric'})
    for feature in MONITORED_CATEGORICAL:
        rows.append({'feature': feature, 'distance': total_variation(reference[feature], current[feature]),
                     'ks_statistic': np.nan, 'ks_pvalue': np.nan, 'psi': np.nan, 'type': 'categorical'})
    return pd.DataFrame(rows)


def drift_score(reference, current):
    # Equal weights fixed before monitoring. Scale and features are unchanged in 2012.
    return float(feature_metrics(reference, current).distance.mean())
