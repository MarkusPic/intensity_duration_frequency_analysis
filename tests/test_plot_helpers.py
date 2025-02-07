from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import pytest

from idf_analysis import SERIES, METHOD, HeavyRainfallIndexAnalyse
from idf_analysis.definitions import COL
from idf_analysis.plot_helpers import idf_bar_axes, _set_xlim

PTH_EXAMPLES = Path(__file__).parent.parent / "examples"


@pytest.fixture
def idf():
    """Fixture to initialize an IDF analysis instance and read parameters from a YAML file."""
    idf = HeavyRainfallIndexAnalyse(series_kind=SERIES.PARTIAL, worksheet=METHOD.KOSTRA, extended_durations=False,
                                    method=HeavyRainfallIndexAnalyse.METHODS.SCHMITT)
    idf.read_parameters(PTH_EXAMPLES / 'ehyd_112086_idf_data/idf_parameters.yaml')
    return idf


@pytest.fixture
def rainfall_series():
    """Fixture to create a sample rainfall time series with 5-minute intervals and two rainfall events."""
    freq_min = 5
    start_time = pd.Timestamp("2024-01-01 00:00:00")
    end_time = start_time + pd.Timedelta(hours=10)
    time_index = pd.date_range(start=start_time, end=end_time, freq=f"{freq_min}min")

    rainfall = pd.Series(0., index=time_index)

    events = [
        {"start": start_time + pd.Timedelta(minutes=10), "duration": 20, "intensity": 20.},
        {"start": start_time + pd.Timedelta(hours=5, minutes=10), "duration": 60, "intensity": 50.}
    ]

    for event in events:
        end_time = event["start"] + pd.Timedelta(minutes=event["duration"])
        mask = (rainfall.index >= event["start"]) & (rainfall.index < end_time)
        rainfall.loc[mask] = event["intensity"] / (event["duration"] / freq_min)
    return rainfall


def test_idf_bar_axes(idf, rainfall_series):
    idf.set_series(rainfall_series)

    fig, ax = plt.subplots()
    table = idf.get_return_periods_frame()
    idf_bar_axes(ax, table)
    plt.close(fig)


def test_rain_events(idf, rainfall_series):
    idf.set_series(rainfall_series)
    events = idf.rain_events

    fig, ax = plt.subplots()
    df = idf.event_dataframe(events.iloc[0])

    bars = ax.barh(df.index, df[COL.MAX_OVERLAPPING_SUM], align='center', height=0.6, color='#1E88E5')
    labels = ax.bar_label(bars, df[COL.MAX_OVERLAPPING_SUM].round(1),
                          padding=5, color='black', transform=ax.transAxes)

    _set_xlim(ax, bars, labels)
    plt.close(fig)
