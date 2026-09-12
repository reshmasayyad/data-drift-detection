"""Demand estimation with training-only fits and separate model selection."""
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from .data import FEATURES, NUMERIC, CATEGORICAL, TARGET


def make_model(max_leaf_nodes: int = 15):
    return HistGradientBoostingRegressor(
        categorical_features=[False] * len(NUMERIC) + [True] * len(CATEGORICAL),
        max_iter=120, max_leaf_nodes=max_leaf_nodes, learning_rate=0.07,
        l2_regularization=2.0, min_samples_leaf=25, random_state=42,
    )


def predict(model, frame):
    return np.maximum(model.predict(frame[FEATURES]), 0.0)


def error_metrics(actual, predictions):
    return {'mae': float(mean_absolute_error(actual, predictions)),
            'rmse': float(np.sqrt(mean_squared_error(actual, predictions)))}


def select_model(train, selection):
    rows, fitted = [], {}
    candidates = {'mean_baseline': DummyRegressor(strategy='mean'),
                  'boosting_15_leaves': make_model(15),
                  'boosting_31_leaves': make_model(31)}
    for name, model in candidates.items():
        model.fit(train[FEATURES], train[TARGET])
        fitted[name] = model
        rows.append({'model': name, **error_metrics(selection[TARGET], predict(model, selection))})
    scores = pd.DataFrame(rows).sort_values('mae').reset_index(drop=True)
    best = scores.iloc[0]['model']
    if best == 'mean_baseline':
        raise RuntimeError('Boosting did not outperform the mean baseline on selection data')
    leaves = 15 if best == 'boosting_15_leaves' else 31
    return fitted[best], leaves, scores
