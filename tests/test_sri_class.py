import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from idf_analysis import HeavyRainfallIndexAnalyse
from idf_analysis.definitions import SERIES, METHOD
from idf_analysis.parameter_formulas import LinearFormula


PTH_EXAMPLES = Path(__file__).parent.parent / "examples"


@pytest.fixture
def idf():
    """Fixture to initialize an IDF analysis instance and read parameters from a YAML file."""
    idf = HeavyRainfallIndexAnalyse(series_kind=SERIES.PARTIAL, worksheet=METHOD.KOSTRA, extended_durations=False,
                                    method=HeavyRainfallIndexAnalyse.METHODS.SCHMITT)
    idf.read_parameters(PTH_EXAMPLES / 'ehyd_112086_idf_data/idf_parameters.yaml')
    return idf


@pytest.fixture
def idf_r():
    """Fixture to initialize an IDF analysis instance and read parameters from a idf table file."""
    idf_table = pd.read_csv(PTH_EXAMPLES / 'ehyd_112086_idf_data/idf_table_UNIX.csv', header=[0, 1], index_col=0)
    idf_table.columns = idf_table.columns.get_level_values(0).astype(int)
    idf = HeavyRainfallIndexAnalyse.from_idf_table(idf_table, linear_interpolation=True)
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


@pytest.mark.parametrize("height_of_rainfall, durations, expected_output", [
    # Scalar duration, scalar return_period
    (60, 2.5, 12),
    (30, 7., 8),
    # List duration, scalar return_period
    (np.array([60, 90]), 2.5, np.array([12, 12])),
    # Scalar duration, list return_period
    # (60, np.array([2.5, 7]), np.array([40.2323786, 51.73796309])),  # TODO
    # List duration, list return_period
    (np.array([60, 90]), np.array([2.5, 7]), np.array([12, 12])),
])
def test_get_sri1(idf, height_of_rainfall, durations, expected_output):
    result = idf.get_sri(height_of_rainfall, durations)
    if isinstance(expected_output, (np.float64, float, int, np.int64)):
        assert np.isclose(result, expected_output, atol=1e-5), f'{result} != {expected_output}'
    else:
        np.testing.assert_allclose(result, expected_output, rtol=1e-6)


@pytest.mark.parametrize("rain, duration, expected", [
    (25, 60, 1),
    (90, 90, 8),
])
def test_get_sri(idf, rain, duration, expected):
    result = idf.get_sri(rain, duration)
    assert np.isclose(result, expected, atol=1e-5)


def test_set_series(idf, rainfall_series):
    idf.set_series(rainfall_series)
    assert idf._freq == pd.Timedelta(minutes=5)


def test_result_sri_table(idf):
    # Call the result_table method
    result = idf.result_sri_table()

    # Check that the result is a pandas DataFrame
    assert isinstance(result, pd.DataFrame), "Result should be a pandas DataFrame"

    # Check the shape of the DataFrame (expected rows and columns)
    expected_shape = (21, 12)  # 21 rows, 12 columns based on the sample data
    assert result.shape == expected_shape, f"Expected shape {expected_shape}, but got {result.shape}"

    # Check that the first few values match the expected ones (you can adjust this based on the actual data)
    expected_sample = [
        [10.991037, 13.603204, 15.579233, 18.191401, 18.711164, 21.323332,
         22.143458, 30.779406, 35.208098, 48.494173, 61.558813, 62.001682],
        [17.468336, 21.292072, 24.184616, 28.008353, 28.769191, 32.592928,
         33.793442, 46.972885, 53.731573, 74.007638, 93.945769, 94.621638],
        [21.963249, 26.741748, 30.356544, 35.135043, 36.085858,
         40.864358, 42.364633, 58.88684, 67.359767, 92.778547,
         117.77368, 118.620973],
    ]

    for i, expected_row in enumerate(expected_sample):
        np.testing.assert_allclose(result.iloc[i, :], expected_row, rtol=1e-6)


@pytest.mark.parametrize("durations, expected_shape", [
    ([20], (16, 1)),
])
def test_get_rainfall_sum_frame(idf, rainfall_series, durations, expected_shape):
    idf.set_series(rainfall_series)
    result = idf.get_rainfall_sum_frame(durations=durations)
    assert result.shape == expected_shape


@pytest.mark.parametrize("durations, expected_shape", [
    ([20], (16, 1)),
])
def test_get_return_periods_frame(idf, rainfall_series, durations, expected_shape):
    idf.set_series(rainfall_series)
    result = idf.get_return_periods_frame(durations=durations)
    assert result.shape == expected_shape


def test_write_and_read_parameters(idf):
    filename = 'temp.yaml'
    idf.write_parameters(filename)
    assert os.path.isfile(filename)
    idf.read_parameters(filename)
    os.remove(filename)


def test_write_and_read_return_periods_frame(idf, rainfall_series):
    filename = 'temp.parq'
    idf.set_series(rainfall_series)
    idf.write_return_periods_frame(filename)
    assert os.path.isfile(filename)
    idf.read_return_periods_frame(filename)
    os.remove(filename)


def test_rain_events(idf, rainfall_series):
    idf.set_series(rainfall_series)
    events = idf.rain_events

    expected_events = pd.DataFrame({
        "start": [pd.Timestamp("2024-01-01 00:10:00"), pd.Timestamp("2024-01-01 05:10:00")],
        "end": [pd.Timestamp("2024-01-01 00:25:00"), pd.Timestamp("2024-01-01 06:05:00")],
        "duration": [pd.Timedelta(minutes=20), pd.Timedelta(minutes=60)],
        "rain_sum": [20.0, 50.0],
        "last_event": [pd.NaT, pd.Timedelta(hours=4, minutes=45)],
    })

    pd.testing.assert_frame_equal(events.reset_index(drop=True), expected_events, check_exact=False, atol=1e-5)


def test_add_max_return_periods_to_events(idf, rainfall_series):
    idf.set_series(rainfall_series)
    events = idf.rain_events
    idf.add_max_return_periods_to_events(events)

    # Expected events DataFrame
    expected_events_df = pd.DataFrame({
        'start': pd.to_datetime(['2024-01-01 00:10:00', '2024-01-01 05:10:00']),
        'end': pd.to_datetime(['2024-01-01 00:25:00', '2024-01-01 06:05:00']),
        'duration': pd.to_timedelta(['0 days 00:20:00', '0 days 01:00:00']),
        'rain_sum': [20.0, 50.0],
        'last_event': pd.to_timedelta([pd.NaT, '0 days 04:45:00']),
        'max_return_period': [0.838494, 7.501639],
        'max_return_period_duration': [20, 360]
    })

    # Compare the actual result with the expected DataFrame
    pd.testing.assert_frame_equal(events, expected_events_df, check_exact=False, atol=1e-5)


def test_write_and_read_rain_events(idf, rainfall_series):
    idf.set_series(rainfall_series)
    filename = 'temp.csv'

    idf.write_rain_events(filename)
    assert os.path.isfile(filename)

    idf.read_rain_events(filename)

    os.remove(filename)


def test_get_max_event_intensities_frame(idf, rainfall_series):
    idf.set_series(rainfall_series)
    events = idf.rain_events
    intensities = idf.get_max_event_intensities_frame(events)

    # Expected intensities DataFrame
    expected_intensities_df = pd.DataFrame({
        5.0: [5.00, 4.17],
        10.0: [10.00, 8.33],
        15.0: [15.0, 12.5],
        20.0: [20.00, 16.67],
        30.0: [20.0, 25.0],
        45.0: [20.0, 37.5],
        60.0: [20.0, 50.0],
        90.0: [20.0, 50.0],
        120.0: [20.0, 50.0],
        180.0: [20.0, 50.0],
        240.0: [20.0, 50.0],
        360.0: [20.0, 70.0],
        540.0: [20.0, 70.0],
        720.0: [20.0, 70.0],
        1080.0: [20.0, 70.0],
        1440.0: [20.0, 70.0],
        2880.0: [20.0, 70.0],
        4320.0: [20.0, 70.0],
        5760.0: [20.0, 70.0],
        7200.0: [20.0, 70.0],
        8640.0: [20.0, 70.0]
    })

    # Compare the actual result with the expected DataFrame
    pd.testing.assert_frame_equal(intensities, expected_intensities_df, check_exact=False, atol=1e-5)


def test_get_max_return_periods_per_durations_frame(idf, rainfall_series):
    idf.set_series(rainfall_series)
    events = idf.rain_events
    return_periods = idf.get_max_return_periods_per_durations_frame(events)

    # Expected return periods DataFrame
    expected_return_periods_df = pd.DataFrame({
        5.0: [0.24, 0.18],
        10.0: [0.33, 0.22],
        15.0: [0.53, 0.33],
        20.0: [0.84, 0.49],
        30.0: [0.54, 1.03],
        45.0: [0.43, 2.70],
        60.0: [0.41, 5.99],
        90.0: [0.35, 4.50],
        120.0: [0.32, 3.69],
        180.0: [0.27, 2.82],
        240.0: [0.25, 2.34],
        360.0: [0.22, 7.50],
        540.0: [0.19, 5.46],
        720.0: [0.17, 4.38],
        1080.0: [0.12, 3.43],
        1440.0: [0.08, 2.69],
        2880.0: [0.05, 1.46],
        4320.0: [0.07, 1.04],
        5760.0: [0.09, 0.81],
        7200.0: [0.08, 0.64],
        8640.0: [0.09, 0.62]
    })

    # Compare the actual result with the expected DataFrame
    pd.testing.assert_frame_equal(return_periods, expected_return_periods_df, check_exact=False, atol=1e-5)


def test_idf_r(idf_r):
    # Check the parameters_final attribute
    parameters_final = idf_r.parameters.parameters_final
    expected_final = {0: {'u': LinearFormula(), 'w': LinearFormula()}}

    for outer_key, outer_value in expected_final.items():
        assert outer_key in parameters_final, f"Key {outer_key} not found in parameters_final"
        for inner_key, inner_value in outer_value.items():
            assert inner_key in parameters_final[outer_key], f"Key {inner_key} not found in parameters_final[{outer_key}]"
            assert type(parameters_final[outer_key][inner_key]) == type(inner_value), f"Expected {inner_key} value {inner_value}, but got {parameters_final[outer_key][inner_key]}"

    # assert parameters_final == expected_final, f"Expected {expected_final}, but got {parameters_final}"

    # Check the parameters_series attribute
    parameters_series = idf_r.parameters.parameters_series
    expected_series = {
        'u': [8.61, 14.17, 18.04, 20.9, 24.84, 28.4, 30.6, 33.16, 34.61, 36.2, 38.34, 41.58, 45.09, 47.76, 51.79, 54.86,
              64.22, 69.23, 74.79, 80.77, 82.42],
        'w': [3.84644007, 4.9513404, 5.7415318, 6.37822869, 7.39341092, 8.57464329, 9.52331163, 11.0432409, 12.26633836,
              14.22061156, 14.32383734, 14.46892822, 14.61715464, 14.72309459, 14.87402415, 14.98043461, 15.20152522,
              18.66198599, 22.67202613, 23.81474637, 26.06492513]
    }
    for key in expected_series:
        assert np.allclose(parameters_series[key], expected_series[key]), f"Expected series for {key} does not match."


def test_result_sri_figure(idf):
    import matplotlib.pyplot as plt
    ax = idf.result_sri_figure()
    assert ax is not None
    plt.close()
