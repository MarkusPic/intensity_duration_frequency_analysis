import pytest
import pandas as pd
import numpy as np
from pandas.tseries.frequencies import to_offset

# Import the functions to be tested (assuming they are in a module named `rain_utils`)
from idf_analysis.sww_utils import (
    guess_freq,
    year_delta,
    rain_events,
    event_number_to_series,
    agg_events,
    event_duration,
    rain_bar_plot,
    resample_rain_series
)

from idf_analysis.definitions import COL


# Fixture for creating a sample datetime index
@pytest.fixture
def sample_datetime_index():
    return pd.date_range(start='2023-01-01', periods=10, freq='min')


# Fixture for creating a sample rain series
@pytest.fixture
def sample_rain_series():
    return pd.Series([0.0, 0.02, 0.03, 0.0, 0.01, 0.05, 0.0, 0.0, 0.02, 0.0],
                     index=pd.date_range(start='2023-01-01', periods=10, freq='min'))


# Fixture for creating sample rain events
@pytest.fixture
def sample_rain_events():
    return pd.DataFrame({
        COL.START: [pd.Timestamp('2023-01-01 00:01'), pd.Timestamp('2023-01-01 00:05')],
        COL.END: [pd.Timestamp('2023-01-01 00:03'), pd.Timestamp('2023-01-01 00:07')]
    })


# Test guess_freq function
@pytest.mark.parametrize("index, expected_freq", [
    (pd.date_range(start='2023-01-01', periods=10, freq='min'), 'min'),  # Minute frequency
    (pd.date_range(start='2023-01-01', periods=10, freq='h'), 'h'),  # Hourly frequency
    (pd.date_range(start='2023-01-01', periods=10, freq='d'), 'd'),  # Daily frequency
    (pd.date_range(start='2023-01-01', periods=3, freq='min'), pd.Timedelta(minutes=1)),  # Small index
    (pd.date_range(start='2023-01-01', periods=20, freq='min').append(pd.date_range(start='2023-01-02', periods=15, freq='2min')), pd.Timedelta(minutes=1)),  # Small index
])
def test_guess_freq(index, expected_freq):
    result = guess_freq(index)
    assert result == to_offset(expected_freq)


# Test year_delta function
@pytest.mark.parametrize("years, expected_timedelta", [
    (1, pd.Timedelta(days=365.25)),
    (2, pd.Timedelta(days=365.25 * 2)),
    (0.5, pd.Timedelta(days=365.25 * 0.5)),
])
def test_year_delta(years, expected_timedelta):
    result = year_delta(years)
    assert result == expected_timedelta


# Test rain_events function
@pytest.mark.parametrize("series, ignore_rain_below, min_gap, expected_events",
                         [
                             (pd.Series([0.0, 0.02, 0.03, 0.0, 0.01, 0.05, 0.0, 0.0, 0.02, 0.0],
                                        index=pd.date_range(start='2023-01-01', periods=10, freq='min')),
                              0.01,
                              pd.Timedelta(hours=4),
                              pd.DataFrame(
                                  {COL.START: [pd.Timestamp('2023-01-01 00:01'), ],
                                   COL.END: [pd.Timestamp('2023-01-01 00:08'), ]})),
                             (pd.Series([0.0, 0.0, 0.0],
                                        index=pd.date_range(start='2023-01-01', periods=3, freq='min')),
                              0.01,
                              pd.Timedelta(hours=4),
                              pd.DataFrame()),  # No rain events
                         ])
def test_rain_events(series, ignore_rain_below, min_gap, expected_events):
    result = rain_events(series, ignore_rain_below, min_gap)
    pd.testing.assert_frame_equal(result, expected_events)


# Test event_number_to_series function
def test_event_number_to_series(sample_rain_events, sample_datetime_index):
    result = event_number_to_series(sample_rain_events, sample_datetime_index)
    expected_series = pd.Series([np.nan, 0, 0, 0, np.nan, 1, 1, 1, np.nan, np.nan],
                                index=sample_datetime_index)
    pd.testing.assert_series_equal(result, expected_series)


# Test agg_events function
@pytest.mark.parametrize("events, series, agg, expected_result", [
    (pd.DataFrame({COL.START: [pd.Timestamp('2023-01-01 00:01')],
                   COL.END: [pd.Timestamp('2023-01-01 00:03')]}),
     pd.Series([0.0, 0.02, 0.03, 0.0], index=pd.date_range(start='2023-01-01', periods=4, freq='min')),
     'sum', np.array([0.05])),
    (pd.DataFrame(), pd.Series([0.0, 0.02, 0.03, 0.0], index=pd.date_range(start='2023-01-01', periods=4, freq='min')),
     'sum', np.array([])),
])
def test_agg_events(events, series, agg, expected_result):
    result = agg_events(events, series, agg)
    np.testing.assert_array_equal(result, expected_result)


# Test event_duration function
def test_event_duration(sample_rain_events):
    result = event_duration(sample_rain_events)
    expected_duration = pd.Series([pd.Timedelta(minutes=2), pd.Timedelta(minutes=2)])
    pd.testing.assert_series_equal(result, expected_duration)


# Test resample_rain_series function
@pytest.mark.parametrize("series, expected_freq", [
    (pd.Series([0.0, 0.02, 0.03, 0.0], index=pd.date_range(start='2023-01-01', periods=4, freq='1min')), 1),
    (pd.Series([0.0, 0.02, 0.03, 0.0], index=pd.date_range(start='2023-01-01', periods=4, freq='5min')), 5),
])
def test_resample_rain_series(series, expected_freq):
    result_series, result_freq = resample_rain_series(series)
    assert result_freq == expected_freq
    assert isinstance(result_series, pd.Series)


# Test rain_bar_plot function (visual test, no assertions)
def test_rain_bar_plot(sample_rain_series):
    import matplotlib.pyplot as plt
    ax = rain_bar_plot(sample_rain_series)
    assert ax is not None
    plt.close()

    ax = rain_bar_plot(sample_rain_series, reverse=True)
    assert ax is not None
    plt.close()

    ax = rain_bar_plot(sample_rain_series.iloc[:1], reverse=True)
    assert ax is not None
    plt.close()
