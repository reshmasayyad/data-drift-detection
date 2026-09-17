import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from drift_detection.data import FEATURES, TARGET, load_hourly, partition
from drift_detection.metrics import population_stability_index, normalized_wasserstein, total_variation
from drift_detection.monitoring import Thresholds, mark_alerts
from drift_detection.experiments import controlled_stream
from drift_detection.policies import simulate_policy
import drift_detection.policies as policies


def make_frame(start, days, count=20):
    times = pd.date_range(start, periods=days * 24, freq='h')
    return pd.DataFrame({'timestamp': times, 'hr': times.hour, 'weekday': times.dayofweek,
                         'mnth': times.month, 'season': 1, 'holiday': 0,
                         'workingday': (times.dayofweek < 5).astype(int), 'weathersit': 1,
                         'temp': .3, 'atemp': .3, 'hum': .6, 'windspeed': .1, TARGET: count})


def test_no_target_components_in_features():
    assert not {'cnt', 'casual', 'registered', 'instant', 'yr', 'timestamp'}.intersection(FEATURES)


def test_chronological_partition_and_exhaustive_rows():
    f = make_frame('2011-01-01', 731)
    parts = partition(f)
    assert sum(map(len, parts.values())) == len(f)
    ordered = list(parts.values())
    assert all(a.timestamp.max() < b.timestamp.min() for a, b in zip(ordered, ordered[1:]))


def test_loader_rejects_duplicate_hours_and_target_mismatch(tmp_path):
    f = make_frame('2011-01-01', 1)
    f['dteday'] = f.timestamp.dt.strftime('%Y-%m-%d'); f['casual'] = 5; f['registered'] = 15
    path = tmp_path / 'hour.csv'; f.to_csv(path, index=False)
    assert len(load_hourly(path)) == 24
    pd.concat([f, f.iloc[:1]]).to_csv(path, index=False)
    with pytest.raises(ValueError, match='Duplicate'):
        load_hourly(path)
    f.loc[0, 'cnt'] = 21; f.to_csv(path, index=False)
    with pytest.raises(ValueError, match='Invalid'):
        load_hourly(path)


def test_distribution_metrics_include_open_tails_and_unseen_categories():
    ref = np.arange(100, dtype=float)
    assert population_stability_index(ref, ref) == pytest.approx(0)
    assert normalized_wasserstein(ref, ref) == pytest.approx(0)
    assert population_stability_index(ref, ref + 1000) > 1
    assert normalized_wasserstein(ref, ref + 1000) > 10
    assert total_variation(['a', 'a'], ['b', 'b']) == pytest.approx(1)
    assert normalized_wasserstein([1, 1], [2, 2]) == pytest.approx(1)
    with pytest.raises(ValueError):
        normalized_wasserstein(ref, [np.nan])


def test_error_patience_resets_over_missing_weeks():
    report = pd.DataFrame({'week': pd.to_datetime(['2012-01-02', '2012-01-09', '2012-01-23', '2012-01-30']),
                           'drift_score': [0]*4, 'mae': [10]*4})
    out = mark_alerts(report, Thresholds(1, 5))
    assert out.performance_alert.tolist() == [False, True, False, True]


def test_controlled_concept_shift_leaves_inputs_unchanged():
    cal = make_frame('2011-11-01', 60)
    null = controlled_stream(cal, 5, 'no_change', weeks=6, onset=3)
    shifted = controlled_stream(cal, 5, 'demand_growth', weeks=6, onset=3)
    pd.testing.assert_frame_equal(null[FEATURES], shifted[FEATURES])
    assert (null.loc[null.synthetic_week < 3, TARGET] == shifted.loc[shifted.synthetic_week < 3, TARGET]).all()
    assert (shifted.loc[shifted.synthetic_week >= 3, TARGET] == 32).all()


def test_updates_use_completed_labels_and_never_change_current_batch(monkeypatch):
    monkeypatch.setattr(policies, 'make_model', lambda leaves: DummyRegressor(strategy='mean'))
    train, stream = make_frame('2011-01-03', 28, 10), make_frame('2012-01-02', 56, 20)
    model = DummyRegressor().fit(train[FEATURES], train[TARGET]); cuts = Thresholds(1, 2)
    report, predictions = simulate_policy(train, stream, model, 15, cuts, 'error_triggered')
    _, frozen = simulate_policy(train, stream, model, 15, cuts, 'frozen')
    np.testing.assert_allclose(predictions.prediction.iloc[:336], frozen.prediction.iloc[:336])
    assert report.retrained_after_batch.iloc[1]
    assert predictions.prediction.iloc[336] == pytest.approx(20)
    assert not report.retrained_after_batch.iloc[-1]
    changed = stream.copy(); changed.loc[changed.timestamp >= '2012-01-30', TARGET] = 2000
    _, changed_predictions = simulate_policy(train, changed, model, 15, cuts, 'error_triggered')
    np.testing.assert_allclose(predictions.prediction.iloc[:672], changed_predictions.prediction.iloc[:672])
    for row in report.loc[report.retrained_after_batch].itertuples():
        assert row.week <= row.latest_label_used_if_updated < row.week + pd.Timedelta(days=7)


def test_initial_training_history_cannot_contain_future_labels():
    train, stream = make_frame('2011-01-03', 28), make_frame('2012-01-02', 14)
    model = DummyRegressor().fit(train[FEATURES], train[TARGET])
    with pytest.raises(ValueError, match='overlaps'):
        simulate_policy(train, stream, model, 15, Thresholds(1, 2), 'scheduled', available_history=stream)
