"""Known-change stress tests with within-day dependence retained."""
import numpy as np
import pandas as pd
from .data import TARGET
from .monitoring import weekly_report, mark_alerts


def controlled_stream(calibration, seed=42, scenario='no_change', weeks=40, onset=20):
    """Resample full days from calibration; synthetic timestamps define batch order.

    Sensor bias changes observed inputs while holding rentals fixed. Demand growth
    changes rentals while holding inputs fixed. Neither is a causal weather model.
    """
    if scenario not in {'no_change', 'sensor_bias', 'demand_growth'}:
        raise ValueError(f'Unknown scenario: {scenario}')
    rng = np.random.default_rng(seed)
    days = [g.copy() for _, g in calibration.groupby(calibration.timestamp.dt.normalize()) if len(g) >= 22]
    if len(days) < 7:
        raise ValueError('Need at least seven sufficiently observed calibration days')
    chunks = []
    origin = pd.Timestamp('2030-01-07')  # Monday; synthetic sequence, not a forecast date.
    for index in range(weeks):
        week = []
        for day_index in range(7):
            chunk = days[int(rng.integers(len(days)))].copy()
            chunk['timestamp'] = origin + pd.Timedelta(weeks=index, days=day_index) + pd.to_timedelta(chunk.hr, unit='h')
            chunk['synthetic_week'] = index
            chunk['injected'] = scenario != 'no_change' and index >= onset
            week.append(chunk)
        batch = pd.concat(week, ignore_index=True)
        if index >= onset:
            if scenario == 'sensor_bias':
                batch['temp'] = (batch.temp - 0.20).clip(0, 1)
                batch['atemp'] = (batch.atemp - 0.20).clip(0, 1)
                batch['hum'] = (batch.hum + 0.35).clip(0, 1)
            elif scenario == 'demand_growth':
                batch[TARGET] = np.rint(batch[TARGET] * 1.6).astype(int)
        chunks.append(batch)
    return pd.concat(chunks, ignore_index=True)


def evaluate_controlled(reference, calibration, model, cuts, seeds=range(20), weeks=40, onset=20):
    rows, traces = [], {}
    for scenario in ('no_change', 'sensor_bias', 'demand_growth'):
        for seed in seeds:
            stream = controlled_stream(calibration, seed, scenario, weeks, onset)
            report, _ = weekly_report(reference, stream, model)
            report = mark_alerts(report, cuts)
            if seed == 0:
                traces[scenario] = report
            for detector in ('covariate_alert', 'performance_alert'):
                flags = report[detector].to_numpy()
                before = flags[:onset]
                after = np.flatnonzero(flags[onset:])
                rows.append({'scenario': scenario, 'seed': seed, 'detector': detector,
                             'prechange_alert_fraction': before.mean(),
                             'detected_after_onset': bool(after.size),
                             'delay_batches': float(after[0]) if after.size else np.nan,
                             'all_alert_fraction': flags.mean()})
    return pd.DataFrame(rows), traces


def summarize_experiments(results):
    return results.groupby(['scenario', 'detector'], as_index=False).agg(
        prechange_alert_fraction=('prechange_alert_fraction', 'mean'),
        detected_fraction=('detected_after_onset', 'mean'),
        median_delay_batches=('delay_batches', 'median'),
        all_alert_fraction=('all_alert_fraction', 'mean'),
    )
