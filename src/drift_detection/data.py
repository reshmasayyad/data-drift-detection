"""Validated data loading and chronological partitions."""
from pathlib import Path
import pandas as pd

NUMERIC = ['temp', 'atemp', 'hum', 'windspeed']
CATEGORICAL = ['hr', 'weekday', 'mnth', 'season', 'holiday', 'workingday', 'weathersit']
FEATURES = NUMERIC + CATEGORICAL
TARGET = 'cnt'


def load_hourly(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = FEATURES + [TARGET, 'dteday', 'casual', 'registered']
    missing = set(required) - set(frame)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    frame['timestamp'] = pd.to_datetime(frame['dteday']) + pd.to_timedelta(frame['hr'], unit='h')
    frame = frame.sort_values('timestamp').reset_index(drop=True)
    if frame.timestamp.duplicated().any() or frame[FEATURES + [TARGET]].isna().any().any():
        raise ValueError('Duplicate timestamps or missing model values')
    if (frame[TARGET] < 0).any() or not (frame[TARGET] == frame.casual + frame.registered).all():
        raise ValueError('Invalid rental counts')
    return frame


def partition(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    t = frame.timestamp
    return {
        'train': frame.loc[t < '2011-10-01'].copy(),
        'selection': frame.loc[(t >= '2011-10-01') & (t < '2011-11-01')].copy(),
        'calibration': frame.loc[(t >= '2011-11-01') & (t < '2012-01-01')].copy(),
        'monitoring': frame.loc[t >= '2012-01-01'].copy(),
    }


def complete_weeks(frame: pd.DataFrame, minimum_hours: int = 100):
    """Monday-start weeks inside the observed date span, with coverage checks."""
    keys = frame.timestamp.dt.to_period('W-SUN').dt.start_time
    for start, batch in frame.groupby(keys, sort=True):
        contained = (start >= frame.timestamp.min().normalize() and
                     start + pd.Timedelta(days=6) <= frame.timestamp.max().normalize())
        if contained and len(batch) >= minimum_hours:
            yield pd.Timestamp(start), batch.copy()
