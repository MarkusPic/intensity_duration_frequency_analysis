import pytest
from argparse import ArgumentParser
from idf_analysis.arg_parser import heavy_rain_parser, Borders


def test_borders_str():
    borders = Borders(min_=10, max_=20, unit='mm')
    assert str(borders) == '>= 10 mm and <= 20 mm'

    borders = Borders(min_=10, unit='mm')
    assert str(borders) == '>= 10 mm'

    borders = Borders(max_=20, unit='mm')
    assert str(borders) == '<= 20 mm'

    borders = Borders(unit='mm')
    assert str(borders) == ''


def test_borders_contains():
    borders = Borders(min_=10, max_=20)
    assert 15 in borders
    assert 5 not in borders
    assert 25 not in borders


def test_borders_iter():
    borders = Borders(min_=10, max_=20, unit='mm')
    assert list(borders) == ['>= 10 mm and <= 20 mm']


def test_heavy_rain_parser():
    import sys
    sys.argv = ['script_name', '-i', 'input.csv', '-ws', 'ATV-A_121', '-kind', 'annual', '-t', '50', '-d', '60', '-r', '10', '-h_N', '20', '--r_720_1', '--plot', '--export_table']
    parser = heavy_rain_parser()

    # Test required argument
    sys.argv = ['script_name', '--help']
    with pytest.raises(SystemExit):
        parser = heavy_rain_parser()

    # Test valid input
    sys.argv = ['script_name', '-i', 'input.csv']
    args = heavy_rain_parser()
    assert args.input == 'input.csv'
    assert args.worksheet == 'KOSTRA'
    assert args.series_kind == 'partial'
    assert args.return_period is None
    assert args.duration is None
    assert args.flow_rate_of_rainfall is None
    assert args.height_of_rainfall is None
    assert args.r_720_1 is False
    assert args.plot is False
    assert args.export_table is False

    # Test with additional arguments
    sys.argv = ['script_name', '-i', 'input.csv', '-ws', 'ATV-A_121', '-kind', 'annual', '-t', '50', '-d', '60', '-r', '10', '-h_N', '20',
         '--r_720_1', '--plot', '--export_table']
    args = heavy_rain_parser()
    assert args.input == 'input.csv'
    assert args.worksheet == 'ATV-A_121'
    assert args.series_kind == 'annual'
    assert args.return_period == 50
    assert args.duration == 60
    assert args.flow_rate_of_rainfall == 10
    assert args.height_of_rainfall == 20
    assert args.r_720_1 is True
    assert args.plot is True
    assert args.export_table is True

    # Test invalid choices
    with pytest.raises(SystemExit):
        sys.argv = ['script_name', '-i', 'input.csv', '-ws', 'invalid_method']
        args = heavy_rain_parser()

    with pytest.raises(SystemExit):
        sys.argv = ['-i', 'input.csv', '-kind', 'invalid_series']
        args = heavy_rain_parser()

    with pytest.raises(SystemExit):
        sys.argv = ['-i', 'input.csv', '-t', '0.4']
        args = heavy_rain_parser()

    with pytest.raises(SystemExit):
        sys.argv = ['-i', 'input.csv', '-d', '4']
        args = heavy_rain_parser()

    with pytest.raises(SystemExit):
        sys.argv = ['-i', 'input.csv', '-r', '-1']
        args = heavy_rain_parser()

    with pytest.raises(SystemExit):
        sys.argv = ['-i', 'input.csv', '-h_N', '-1']
        args = heavy_rain_parser()
