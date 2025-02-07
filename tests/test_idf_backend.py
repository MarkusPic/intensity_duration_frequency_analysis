import pytest
import numpy as np
from idf_analysis.idf_backend import IdfParameters
from idf_analysis.definitions import SERIES, METHOD, PARAM, APPROACH
from idf_analysis.parameter_formulas import HyperbolicAuto, DoubleLogNormAuto, LinearFormula


@pytest.mark.parametrize("extended, expected_durations", [
    (False, np.array([5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 360, 540, 720, 1080])),
    (True, np.array([5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 360, 540, 720, 1080,
                     1440, 2880, 4320, 5760, 7200, 8640]))
])
def test_idf_durations(extended, expected_durations):
    idf_params = IdfParameters(series_kind=SERIES.PARTIAL, worksheet=METHOD.KOSTRA, extended_durations=extended)
    np.testing.assert_array_equal(idf_params.durations, expected_durations)


@pytest.mark.parametrize("worksheet, expected_params", [
    (METHOD.KOSTRA, {0: {'u': {'function': 'hyperbolic'}, 'w': {'function': 'double_logarithmic'}},
                     60: {'u': {'function': 'double_logarithmic'}, 'w': {'function': 'double_logarithmic'}},
                     720: {'u': {'function': 'linear'}, 'w': {'function': 'linear'}}}),
    (METHOD.DWA_adv, {0: {'u': {'function': 'hyperbolic'}, 'w': {'function': 'double_logarithmic'}},
                      180: {'u': {'function': 'double_logarithmic'}, 'w': {'function': 'double_logarithmic'}},
                      1440: {'u': {'function': 'linear'}, 'w': {'function': 'linear'}}})
])
def test_idf_parameters_final(worksheet, expected_params):
    idf_params = IdfParameters(series_kind=SERIES.PARTIAL, worksheet=worksheet, extended_durations=True)
    assert idf_params.parameters_final == expected_params


@pytest.mark.parametrize("filter_value, expected_durations", [
    (15, np.array([15, 30, 45, 60, 90, 120, 180, 240, 360, 540, 720, 1080, 1440, 2880, 4320, 5760, 7200, 8640])),
    (120, np.array([120, 240, 360, 720, 1080, 1440, 2880, 4320, 5760, 7200, 8640]))
])
def test_filter_durations(filter_value, expected_durations):
    idf_params = IdfParameters(series_kind=SERIES.PARTIAL, worksheet=METHOD.KOSTRA, extended_durations=True)
    idf_params.filter_durations(filter_value)
    np.testing.assert_array_equal(idf_params.durations, expected_durations)


def test_clear_and_add_parameter_approach():
    idf_params = IdfParameters(series_kind=SERIES.PARTIAL, worksheet=METHOD.KOSTRA, extended_durations=False)
    idf_params.clear_parameter_approaches()
    assert idf_params.parameters_final == {}

    idf_params.add_parameter_approach(60, APPROACH.LIN, APPROACH.LOG1)
    expected = {60: {'u': {'function': 'linear'}, 'w': {'function': 'single_logarithmic'}}}
    assert idf_params.parameters_final == expected


@pytest.mark.parametrize("duration, param, expected", [
    (30, PARAM.U, {'function': 'hyperbolic'}),
    (300, PARAM.W, {'function': 'double_logarithmic'})
])
def test_get_duration_section(duration, param, expected):
    idf_params = IdfParameters(series_kind=SERIES.PARTIAL, worksheet=METHOD.KOSTRA, extended_durations=False)
    assert idf_params.get_duration_section(duration, param) == expected


@pytest.fixture
def idf_params_from_yaml():
    """Fixture to load IdfParameters from a YAML file."""
    return IdfParameters.from_yaml('../examples/ehyd_112086_idf_data/idf_parameters.yaml')


def test_load_from_yaml(idf_params_from_yaml):
    assert isinstance(idf_params_from_yaml.parameters_series, dict)
    assert isinstance(idf_params_from_yaml.parameters_final, dict)

    # Check specific values in parameters_series
    expected_u = np.array([9.2852, 14.4656, 18.2676, 21.223, 24.6312, 27.6473, 29.8823,
                           32.5402, 34.2917, 36.8614, 39.1387, 41.8902, 44.9288, 46.8465,
                           51.6007, 55.8142, 64.2187, 69.2324, 74.7885, 80.7697, 82.4165])
    expected_w = np.array([2.109, 3.3818, 5.3372, 7.022, 8.6112, 9.1709, 10.5036,
                           11.9877, 12.7487, 13.3321, 13.7618, 14.2992, 14.698, 15.0692,
                           14.9139, 14.3355, 15.2022, 18.6616, 22.6725, 23.8148, 26.0658])

    np.testing.assert_allclose(idf_params_from_yaml.parameters_series['u'], expected_u, atol=1e-5)
    np.testing.assert_allclose(idf_params_from_yaml.parameters_series['w'], expected_w, atol=1e-5)

    # Check specific values in parameters_final
    expected_parameters_final = {
        0: {'u': 'HyperbolicAuto(38.04, 16.10)', 'w': 'DoubleLogNormAuto(0.16, 0.55)'},
        60: {'u': 'DoubleLogNormAuto(2.65, 0.18)', 'w': 'DoubleLogNormAuto(1.88, 0.13)'},
        720: {'u': 'Linear()', 'w': 'Linear()'}
    }
    expected_parameters_final_types = {
        0: {'u': HyperbolicAuto, 'w': DoubleLogNormAuto},
        60: {'u': DoubleLogNormAuto, 'w': DoubleLogNormAuto},
        720: {'u': LinearFormula, 'w': LinearFormula}
    }
    assert idf_params_from_yaml.parameters_final.keys() == expected_parameters_final.keys()
    for key in expected_parameters_final:
        assert isinstance(idf_params_from_yaml.parameters_final[key]['u'], expected_parameters_final_types[key]['u'])
        assert isinstance(idf_params_from_yaml.parameters_final[key]['w'], expected_parameters_final_types[key]['w'])
        assert str(idf_params_from_yaml.parameters_final[key]['u']) == expected_parameters_final[key]['u']
        assert str(idf_params_from_yaml.parameters_final[key]['w']) == expected_parameters_final[key]['w']


@pytest.mark.parametrize("duration, expected", [
    (30, [np.float64(24.756091991950807), np.float64(7.633875938389456)]),
    (300, [np.float64(40.24302986233447), np.float64(13.762039748480971)])
])
def test_get_u_w_single(idf_params_from_yaml, duration, expected):
    result = list(idf_params_from_yaml.get_u_w(duration))
    assert np.allclose(result, expected, atol=1e-5)


def test_get_u_w_multiple(idf_params_from_yaml):
    result = list(idf_params_from_yaml.get_u_w([10, 45, 700]))
    expected_u = np.array([14.57579085, 28.01751269, 46.9788282])
    expected_w = np.array([4.17305995, 9.54000732, 15.35681586])

    np.testing.assert_allclose(result[0], expected_u, atol=1e-5)
    np.testing.assert_allclose(result[1], expected_w, atol=1e-5)
