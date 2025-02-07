import pytest
import pandas as pd
import datetime
from datetime import timedelta

from idf_analysis.definitions import COL
from idf_analysis.little_helpers import (
    delta2min, minutes_readable, duration_steps_readable,
    height2rate, rate2height, timedelta_readable, timedelta_readable2, return_period_formatter,
    event_caption)


# Parameterized test for delta2min
@pytest.mark.parametrize("input_value, expected", [
    (pd.Timedelta(minutes=30), 30),
    (pd.DateOffset(minutes=45), 45),
    (pd.Timedelta(hours=1), 60),
    (pd.DateOffset(hours=2), 120)
])
def test_delta2min(input_value, expected):
    assert delta2min(input_value) == expected


# Parameterized test for minutes_readable
@pytest.mark.parametrize("input_value, expected", [
    (30, "30 min"),
    (120, "2 h"),
    (1440, "1 d"),
    (1500, "1.0 d"),
    (2880, "2 d"),
    (43200, "30 d")
])
def test_minutes_readable(input_value, expected):
    assert minutes_readable(input_value) == expected


def test_duration_steps_readable():
    assert duration_steps_readable([30, 120, 1440]) == ["30 min", "2 h", "1 d"]


# Parameterized test for height2rate with scalar values
@pytest.mark.parametrize("height, duration, expected", [
    (10, 60, 10 / 60 * (1000 / 6)),
    (20, 120, 20 / 120 * (1000 / 6)),
    (0, 60, 0.0)
])
def test_height2rate_scalar(height, duration, expected):
    result = height2rate(height, duration)
    assert isinstance(result, float)  # Check type
    assert result == pytest.approx(expected, rel=1e-5)


# Parameterized test for height2rate with pandas Series
@pytest.mark.parametrize("heights, duration, expected", [
    (pd.Series([10, 20, 30]), 60, pd.Series([10, 20, 30]) / 60 * (1000 / 6)),
    (pd.Series([5, 15, 25]), 120, pd.Series([5, 15, 25]) / 120 * (1000 / 6))
])
def test_height2rate_series(heights, duration, expected):
    result = height2rate(heights, duration)
    assert isinstance(result, pd.Series)  # Check type
    pd.testing.assert_series_equal(result, expected, atol=1e-5)  # Compare full series with tolerance


# Parameterized test for rate2height with scalar values
@pytest.mark.parametrize("rate, duration, expected", [
    (2.77778, 60, 2.77778 * 60 / (1000 / 6)),
    (4, 120, 4 * 120 / (1000 / 6)),
    (0, 60, 0.0)
])
def test_rate2height_scalar(rate, duration, expected):
    result = rate2height(rate, duration)
    assert isinstance(result, float)  # Check type
    assert result == pytest.approx(expected, rel=1e-5)


# Parameterized test for rate2height with pandas Series
@pytest.mark.parametrize("rate, duration, expected", [
    (pd.Series([2.77778, 5.55556]), 60, pd.Series([2.77778, 5.55556]) * 60 / (1000 / 6)),
    (pd.Series([1, 3, 7]), 120, pd.Series([1, 3, 7]) * 120 / (1000 / 6))
])
def test_rate2height_series(rate, duration, expected):
    result = rate2height(rate, duration)
    assert isinstance(result, pd.Series)  # Check type
    pd.testing.assert_series_equal(result, expected, atol=1e-5)  # Compare full series with tolerance


@pytest.mark.parametrize("td, min_freq, short, sep, lang, expected", [
    (pd.Timedelta(days=1, hours=2, minutes=30), 'min', False, ', ', 'en', "1 day, 2 hours and 30 minutes"),
    (pd.Timedelta(hours=5, minutes=45), 'min', False, ', ', 'en', "5 hours and 45 minutes"),
    (pd.Timedelta(days=2), 'min', True, ', ', 'en', "2d"),
    (pd.Timedelta(days=0, hours=3, minutes=15), 'min', True, ' - ', 'en', "3h - 15m"),
    (pd.Timedelta(days=1, hours=0, minutes=0), 'min', False, ', ', 'en', "1 day"),
    (pd.Timedelta(days=1, hours=2, minutes=30), 'min', False, ', ', 'de', "1 Tag, 2 Stunden and 30 Minuten"),
    (pd.Timedelta(hours=5, minutes=45), 'min', False, ', ', 'de', "5 Stunden and 45 Minuten"),
    (pd.Timedelta(days=2), 'min', True, ', ', 'de', "2d"),
    (pd.Timedelta(days=0, hours=3, minutes=15), 'min', True, ' - ', 'de', "3h - 15m"),
    (pd.Timedelta(days=1, hours=0, minutes=0), 'min', False, ', ', 'de', "1 Tag"),
])
def test_timedelta_readable(td, min_freq, short, sep, lang, expected):
    assert timedelta_readable(td, min_freq=min_freq, short=short, sep=sep, lang=lang) == expected


@pytest.mark.parametrize(
    "d1, d2, min_freq, short, sep, expected",
    [
        (pd.Timestamp('2020-01-01'), pd.Timestamp('2021-01-01'), 'min', False, ', ', "1 year"),
        (pd.Timestamp('2020-01-01'), pd.Timestamp('2023-06-15'), 'min', False, ', ', "3 years, 23 weeks and 5 days"),
        (pd.Timestamp('2020-01-01'), pd.Timestamp('2020-01-02'), 'min', False, ', ', "1 day"),
        (pd.Timestamp('2020-01-01 12:00:00'), pd.Timestamp('2020-01-01 14:30:00'), 'min', False, ', ',
         "2 hours and 30 minutes"),
        (pd.Timestamp('2020-01-01 00:00:00'), pd.Timestamp('2020-01-01 00:01:00'), 's', True, ', ', "1m"),
        (pd.Timestamp('2020-01-01 00:00:00'), pd.Timestamp('2020-01-01 00:00:30'), 's', True, ', ', "30s"),
        (pd.Timestamp('2020-01-01'), pd.Timestamp('2022-03-01'), 'min', False, ', ', "2 years, 8 weeks and 4 days"),
        (pd.Timestamp('2020-02-29'), pd.Timestamp('2024-02-29'), 'min', False, ', ', "4 years"),
    ]
)
def test_timedelta_readable2(d1, d2, min_freq, short, sep, expected):
    assert timedelta_readable2(d1, d2, min_freq, short, sep) == expected


# Test cases for return_period_formatter
@pytest.mark.parametrize("input_value, expected_output", [
    # Test cases for t < 1
    (0.0, "< 1"),  # Edge case: exactly 0
    (0.5, "< 1"),  # Typical case: less than 1
    (0.999, "< 1"),  # Edge case: just below 1

    # Test cases for t > 200
    (201, "$\\gg$ 100"),  # Edge case: just above 200
    (250, "$\\gg$ 100"),  # Typical case: greater than 200
    (1000, "$\\gg$ 100"),  # Large value

    # Test cases for t > 100 and t <= 200
    (101, "> 100"),  # Edge case: just above 100
    (150, "> 100"),  # Typical case: between 100 and 200
    (200, "> 100"),  # Edge case: exactly 200

    # Test cases for t >= 1 and t <= 100
    (1, "1.0"),  # Edge case: exactly 1
    (50.123, "50.1"),  # Typical case: decimal value
    (100, "100.0"),  # Edge case: exactly 100

    # Test cases for negative values (if applicable)
    (-1, "< 1"),  # Edge case: negative value
    (-0.5, "< 1"),  # Negative value less than 1
])
def test_return_period_formatter(input_value, expected_output):
    """
    Test the return_period_formatter function with various input values.

    Args:
        input_value (float): The input value for the return period.
        expected_output (str): The expected formatted string output.
    """
    result = return_period_formatter(input_value)
    assert result == expected_output, f"Expected {expected_output}, but got {result} for input {input_value}"


# Test cases for event_caption
@pytest.mark.parametrize("event, unit, lang, expected_output", [
    # Test case 1: English, all fields present
    (
            {
                COL.START: pd.Timestamp('2023-01-01 12:00'),
                COL.END: pd.Timestamp('2023-01-01 14:00'),
                COL.LP: 15.5,
                COL.DUR: timedelta(hours=2),
                COL.MAX_PERIOD: 10,
                COL.MAX_PERIOD_DURATION: 30
            },
            'mm',
            'en',
            "rain event\n"
            "between 2023-01-01 12:00 and 14:00\n"
            "with a total sum of 15.5 mm\n"
            "and a duration of 2 hours.\n"
            "The maximum return period was 10.0 years\n"
            "at a duration of 30 min."
    ),
    # Test case 2: German, all fields present
    (
            {
                COL.START: pd.Timestamp('2023-01-01 12:00'),
                COL.END: pd.Timestamp('2023-01-01 14:00'),
                COL.LP: 15.5,
                COL.DUR: timedelta(hours=2),
                COL.MAX_PERIOD: 10,
                COL.MAX_PERIOD_DURATION: 30
            },
            'mm',
            'de',
            "Regenereignis\n"
            "am 01.01.2023 von 12:00 bis 14:00\n"
            "mit einer Regensumme von 15.5 mm\n"
            "und einer Dauer von 2 Stunden.\n"
            "Die maximale Wiederkehrperiode war 10.0 a\n"
            "bei einer Dauerstufe von 30 min."
    ),
    # Test case 3: English, missing optional fields
    (
            {
                COL.START: pd.Timestamp('2023-01-01 12:00'),
                COL.END: pd.Timestamp('2023-01-01 14:00'),
                COL.LP: 15.5
            },
            'mm',
            'en',
            "rain event\n"
            "between 2023-01-01 12:00 and 14:00\n"
            "with a total sum of 15.5 mm\n"
    ),
    # Test case 4: German, missing optional fields
    (
            {
                COL.START: pd.Timestamp('2023-01-01 12:00'),
                COL.END: pd.Timestamp('2023-01-01 14:00'),
                COL.LP: 15.5
            },
            'mm',
            'de',
            "Regenereignis\n"
            "am 01.01.2023 von 12:00 bis 14:00\n"
            "mit einer Regensumme von 15.5 mm\n"
    ),
    # Test case 5: English, same day, same month, different year
    (
            {
                COL.START: pd.Timestamp('2023-01-01 12:00'),
                COL.END: pd.Timestamp('2024-01-01 14:00'),
                COL.LP: 15.5,
                COL.DUR: timedelta(hours=2)
            },
            'mm',
            'en',
            "rain event\n"
            "between 2023-01-01 12:00 and 2024-01-01 14:00\n"
            "with a total sum of 15.5 mm\n"
            "and a duration of 2 hours.\n"
    ),
    # Test case 6: German, same day, same month, different year
    (
            {
                COL.START: pd.Timestamp('2023-01-01 12:00'),
                COL.END: pd.Timestamp('2024-01-01 14:00'),
                COL.LP: 15.5,
                COL.DUR: timedelta(hours=2)
            },
            'mm',
            'de',
            "Regenereignis\n"
            "von 01.01.2023 12:00 bis 01.01.2024 14:00\n"
            "mit einer Regensumme von 15.5 mm\n"
            "und einer Dauer von 2 Stunden.\n"
    ),
])
def test_event_caption(event, unit, lang, expected_output):
    """
    Test the event_caption function with various input parameters.

    Args:
        event (dict): Dictionary containing event details.
        unit (str): Unit for rainfall.
        lang (str): Language for the caption.
        expected_output (str): Expected formatted caption.
    """
    result = event_caption(event, unit=unit, lang=lang)
    assert result.strip() == expected_output.strip(), f"Expected:\n{expected_output}\n\nGot:\n{result}"
