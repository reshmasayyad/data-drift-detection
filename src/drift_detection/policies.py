"""Predict-before-update evaluation of frozen, periodic and error-triggered models."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from .data import complete_weeks, FEATURES, TARGET
from .models import make_model, predict, error_metrics


def simulate_policy(train, monitoring, initial_model, leaves, cuts, policy,
                    retrain_every=4, cooldown_weeks=4, history_days=180):
    if policy not in {'frozen', 'scheduled', 'error_triggered'}:
        raise ValueError(f'Unknown policy: {policy}')
    model = clone(initial_model)
    model.fit(train[FEATURES], train[TARGET])
    history = train.copy()
    rows, observations = [], []
    streak, last_fit, previous = 0, -cooldown_weeks, None
    batches = list(complete_weeks(monitoring))
    for index, (start, batch) in enumerate(batches):
        # Every prediction precedes exposure of this batch's labels.
        predictions = predict(model, batch)
        metrics = error_metrics(batch[TARGET], predictions)
        observations.append(pd.DataFrame({'timestamp': batch.timestamp, 'actual': batch[TARGET],
                                          'prediction': predictions, 'policy': policy}))
        if previous is not None and start - previous != pd.Timedelta(days=7):
            streak = 0
        streak = streak + 1 if metrics['mae'] > cuts.error else 0
        cutoff = batch.timestamp.max()
        history = pd.concat([history, batch], ignore_index=True)
        window = history.loc[history.timestamp >= cutoff - pd.Timedelta(days=history_days)]
        # All window timestamps are at or before the labels just revealed.
        should_fit = policy == 'scheduled' and (index + 1) % retrain_every == 0
        should_fit |= policy == 'error_triggered' and streak >= 2 and index - last_fit >= cooldown_weeks
        # An update at the end of the final batch cannot affect any evaluated prediction.
        should_fit = bool(should_fit and index < len(batches) - 1)
        if should_fit:
            model = make_model(leaves)
            model.fit(window[FEATURES], window[TARGET])
            last_fit, streak = index, 0
        rows.append({'policy': policy, 'week': start, 'n': len(batch), **metrics,
                     'retrained_after_batch': should_fit,
                     'training_rows_if_updated': len(window) if should_fit else 0,
                     'latest_label_used_if_updated': cutoff if should_fit else pd.NaT})
        previous = start
    return pd.DataFrame(rows), pd.concat(observations, ignore_index=True)


def compare_policies(train, monitoring, initial_model, leaves, cuts):
    reports, observations, summary = [], [], []
    for policy in ('frozen', 'scheduled', 'error_triggered'):
        report, obs = simulate_policy(train, monitoring, initial_model, leaves, cuts, policy)
        reports.append(report); observations.append(obs)
        summary.append({'policy': policy, **error_metrics(obs.actual, obs.prediction),
                        'update_fits': int(report.retrained_after_batch.sum()),
                        'evaluated_hours': len(obs)})
    return pd.DataFrame(summary), pd.concat(reports, ignore_index=True), pd.concat(observations, ignore_index=True)
