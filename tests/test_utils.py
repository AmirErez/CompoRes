import os
import pickle
import re

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.compores.compores_main import CONFIG_FILE_PATH
from src.compores.utils import save_file, load_configuration, invert_dict, shuffle_samples, shuffle_sample_values, \
    calculate_root_mean_square_error, is_considered_imputed_sample, fetch_synthetic_analysis_input_data, \
    extend_instance, cast_dict_to_array, cast_nested_dict_to_array, deduplicate_synthetic_analysis_input_data, \
    bootstrap_p_value, fit_gev_distribution, calculate_p_value_with_gev, tail_case_gev_bootstrapped_p_value, \
    tail_case_gev_weight_adjusted_p_value, gev_p_value


def is_valid_path(path: str) -> bool:
    # Define your path format validation using a regular expression
    valid_path_regex = r'^(?:[/\\])?(?:[a-zA-Z]:[/\\]|[/\\]|\\.\\|\\.\\{2})?' \
                       r'([^\x00-\x1F<>:"/\\|?*\n]+[/\\])*[^\x00-\x1F<>:"/\\|?*\n]*$'
    path_pattern = re.compile(valid_path_regex)

    # Check if the path matches the defined pattern
    return bool(path_pattern.match(path))


class TestUtils:

    @pytest.fixture(scope="function")
    def setup_teardown_load_configuration(self):
        yield CONFIG_FILE_PATH

    @pytest.fixture(scope="function")
    def setup_teardown_save_dictionary(self, tmp_path):
        # Put a test dictionary in the temporary test directory
        input_dict_data = {'key': 'value'}
        input_file_name = 'test_file.pkl'
        input_path_to_save = str(tmp_path)

        # Yield test_microbiome_df_path and control to the test class
        yield input_dict_data, input_file_name, input_path_to_save

    @pytest.fixture(scope="function")
    def setup_teardown_is_considered_imputed_sample(self):
        sample_list = ["sample_1", "sample_2", "sample_3"]
        input_params = {
            "num": "num_key",
            "den": "den_key",
            "ocu_dict": {"100 OCUs": {'threshold': 0.49941307856337375, 'OCUs': {
                "num_key": {"taxa": ["taxon1", "taxon2"], "imputed_in": ["sample_1", "sample_3"]},
                "den_key": {"taxa": ["taxon_3"], "imputed_in": ["sample_2"]}
            }}},
            "total_ocu_number": 100
        }
        expected_array = [True, False, True]
        yield sample_list, input_params, expected_array

    def test_load_configuration(self, setup_teardown_load_configuration):
        # Replace 'existing_config.yaml' with the path to your existing YAML file
        existing_yaml_path = setup_teardown_load_configuration

        # Load the configuration using load_configuration
        config = load_configuration(existing_yaml_path)

        # Assert that the loaded configuration matches the expected values
        assert isinstance(config["PROJECT"], str)

        assert isinstance(config["GROUP1"], str)
        assert isinstance(config["GROUP2"], str)
        assert isinstance(config["GROUP3"], str)


        assert isinstance(config["PATH_TO_MICROBIOME"], str)
        assert is_valid_path(config["PATH_TO_MICROBIOME"])
        assert isinstance(config["PATH_TO_RESPONSE"], str)
        assert is_valid_path(config["PATH_TO_RESPONSE"])
        assert isinstance(config["PATH_TO_OUTPUTS"], str)
        assert is_valid_path(config["PATH_TO_OUTPUTS"])

    def test_save_dictionary(self, setup_teardown_save_dictionary):
        # Given a dictionary and a file name
        test_dict, file_name, path_to_save = setup_teardown_save_dictionary

        # When saving the dictionary
        save_file(test_dict, file_name, path_to_save)

        # Then check if the file has been created
        assert os.path.exists(os.path.join(path_to_save, file_name))

        # And check if the content of the file matches the original dictionary
        with open(os.path.join(path_to_save, file_name), 'rb') as file_handler:
            loaded_dict = pickle.load(file_handler)

        assert loaded_dict == test_dict

    def test_invert_dict(self):
        # Test case 1: Normal case with multiple values
        input_dict = {
            "key_1": ["value_1", "value_2"],
            "key_2": ["value_1", "value_3"]
        }
        expected_output = {
            "value_1": ["key_1", "key_2"],
            "value_2": ["key_1"],
            "value_3": ["key_2"]
        }
        assert invert_dict(input_dict) == expected_output

        # Test case 2: Empty dictionary
        input_dict = {}
        expected_output = {}
        assert invert_dict(input_dict) == expected_output

        # Test case 3: Single key-value pair
        input_dict = {
            "key_1": ["value_1"]
        }
        expected_output = {
            "value_1": ["key_1"]
        }
        assert invert_dict(input_dict) == expected_output

        # Test case 4: Multiple keys with overlapping values
        input_dict = {
            "key_1": ["value_1", "value_2"],
            "key_2": ["value_2", "value_3"],
            "key_3": ["value_1", "value_3"]
        }
        expected_output = {
            "value_1": ["key_1", "key_3"],
            "value_2": ["key_1", "key_2"],
            "value_3": ["key_2", "key_3"]
        }
        assert invert_dict(input_dict) == expected_output

    def test_cast_nested_dict_to_array(self):
        original_dict = {42: {2: 0.09, 1: 0.18, 0: 0.27, 3: 0.99}}
        expected_dict = {42: np.array([0.27, 0.18, 0.09, 0.99])}
        result = cast_nested_dict_to_array(original_dict)
        for key in original_dict:
            assert np.array_equal(result[key], expected_dict[key])

    def test_cast_dict_to_array(self):
        original_dict = {2: 0.09, 1: 0.18, 0: 0.27, 3: 0.99}
        expected_array = np.array([0.27, 0.18, 0.09, 0.99])
        assert np.array_equal(cast_dict_to_array(original_dict), expected_array)

    def test_extend_instance_with_error(self):
        tests = [
            {
                'd1': np.array([1, 2, 3]),
                'd2': {'a': 1, 'b': 2}
            },
            {
                'd1': {'a': 1, 'b': 2},
                'd2': np.array([1, 2, 3])
            },
            {
                'd1': [1, 2, 3],
                'd2': [1, 2, 3]
            }
        ]
        for test in tests:
            with pytest.raises(ValueError) as e:
                extend_instance(test['d1'], test['d2'])
        assert "No fit between instance types" in str(e.value)


    def test_extend_instance_with_array(self):

        tests = [
            {
                'd1': np.array([1, 2, 3]),
                'd2': np.array([4, 5, 6]),
                'expected': np.array([1, 2, 3, 4, 5, 6])
            }
        ]

        for test in tests:
            result = extend_instance(test['d1'], test['d2'])
            assert np.array_equal(result, test['expected'])

    def test_extend_instance_with_dict(self):

        tests = [
            {
                'd1': {1: {'x': 10, 'y': 20}, 2: {'x': 30, 'y': 40}},
                'd2': {3: {'x': 5, 'y': 6}, 4: {'x': 1, 'y': 2}},
                'expected': {1: {'x': 10, 'y': 20}, 2: {'x': 30, 'y': 40}, 3: {'x': 5, 'y': 6}, 4: {'x': 1, 'y': 2}}
            },
            {
                'd1': {1: 1, 2: 2},
                'd2': {3: 3, 4: 4},
                'expected': {1: 1, 2: 2, 3: 3, 4: 4}
            }
        ]

        for test in tests:
            result = extend_instance(test['d1'], test['d2'])
            assert result == test['expected']


    def test_extend_instance_with_nested_array(self):

        tests = [
            {
                'd1': {'a': 1, 'b': {'x': np.array([10, 20]), 'y': 20}},
                'd2': {'b': {'x': np.array([30, 40, 50]), 'y': 40}, 'c': 3},
                'expected': {'a': 1, 'b': {'x': np.array([30, 40, 50]), 'y': 40}, 'c': 3}
            }
        ]

        for test in tests:
            result = extend_instance(test['d1'], test['d2'])
            # Assert correctly, taking into account that nested values can be ndarray
            for key, value in test['expected'].items():
                if isinstance(value, dict):
                    for nested_key, nested_value in test['expected'][key].items():
                        if isinstance(nested_value, np.ndarray):
                            assert np.array_equal(result[key][nested_key], test['expected'][key][nested_key])
                else:
                    assert result[key] == test['expected'][key]

    def test_shuffle_samples_series(self):
        data = pd.Series([1, 2, 3, 4, 5])
        shuffled_data = shuffle_samples(data)
        assert len(shuffled_data) == 5
        assert np.array_equal(np.sort(data.values), np.sort(shuffled_data.values))

    def test_shuffle_samples_dataframe(self):
        data = pd.DataFrame({
            "A": [1, 2, 3],
            "B": [4, 5, 6]
        })
        shuffled_data = shuffle_samples(data)
        assert data.shape == shuffled_data.shape
        assert np.array_equal(np.sort(data.values.flatten()), np.sort(shuffled_data.values.flatten()))

    def test_shuffle_sample_values(self):
        data = pd.DataFrame({
            "A": [1, 2, 3],
            "B": [4, 5, 6],
            "C": [7, 8, 9],
            "D": [10, 11, 12],
            "E": [13, 14, 15]
        })
        shuffled_data = shuffle_sample_values(data)
        print('\n', data, '\n', shuffled_data)
        assert data.shape == shuffled_data.shape
        assert not np.array_equal(data.values, shuffled_data.values)
        assert np.array_equal(np.sort(data.values.flatten()), np.sort(shuffled_data.values.flatten()))

    def test_calculate_root_mean_square_error_zero(self):
        x = [1.0, 2.0, 3.0, 4.0]
        y = [1.5, 3.0, 4.5, 6.0]
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        expected_rmse = calculate_root_mean_square_error(x, y, slope, intercept)
        assert expected_rmse == 0

    def test_calculate_root_mean_square_error(self):
        x = [1.0, 2.0, 3.0, 4.0]
        y = [1.5, 3.5, 4.5, 3.5]
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        y_pred = [slope * xi + intercept for xi in x]
        actual_rmse = np.sqrt(np.mean((np.array(y) - np.array(y_pred)) ** 2))
        expected_rmse = calculate_root_mean_square_error(x, y, slope, intercept)
        assert actual_rmse == pytest.approx(expected_rmse, rel=1e-5)

    def test_fetch_spa_input_data_with_response_tag_pairs(self, monkeypatch):

        def mock_load_file(file_name, path_to_load_from):
            data = {
                'rmse.pkl': {'test-micro-biome': {'1': [0.1, 0.2]}},
                'slope.pkl': {'test-micro-biome': {'1': [1.1, 1.2]}},
                'intercept.pkl': {'test-micro-biome': {'1': [2.1, 2.2]}},
                'response_index.pkl': ['response1', 'response2'],
                'ocu_dictionary_compores_enriched.pkl': {'1 OCUs': {
                    'NUM_OCU': {'response1': ['otu_1, otu_3']}, 'DEN_OCU': {'response1': ['otu_2']}}
                }
            }
            return data[file_name]

        monkeypatch.setattr('src.compores.utils.load_file', mock_load_file)

        result, response = fetch_synthetic_analysis_input_data(
            path_to_outputs='test/path',
            microbiome_file_name='test-micro-biome',
            balance_method='pairs',
            response_tag='response1'
        )

        assert response == 'response1'
        assert result['rmse'] == [0.1]
        assert result['slope'] == [1.1]
        assert result['intercept'] == [2.1]
        assert result['num_ocu'] == [['otu_1, otu_3']]
        assert result['den_ocu'] == [['otu_2']]

    def test_fetch_spa_input_data_with_response_tag_clr(self, monkeypatch):

        def mock_load_file(file_name, path_to_load_from):
            data = {
                'rmse.pkl': {'test-micro-biome': {2: [0.1, 0.2], 3: [0.3, 0.4]}},
                'slope.pkl': {'test-micro-biome': {2: [1.1, 1.2], 3: [1.3, 1.4]}},
                'intercept.pkl': {'test-micro-biome': {2: [2.1, 2.2], 3: [2.3, 2.4]}},
                'response_index.pkl': ['response1', 'response2'],
                'ocu_dictionary_compores_enriched.pkl': {'2 OCUs': {
                    'NUM_OCU': {'response1': ['otu_1, otu_3'], 'response2': ['otu1', 'otu3']}, 'DEN_OCU': {},
                    'OCUs': {'ocu_1': {'taxa': ['otu_1', 'otu_3']}, 'ocu_2': {'taxa': ['otu_2']}}
                }, '3 OCUs': {
                    'NUM_OCU': {'response1': ['otu_1'], 'response2': ['otu_3']}, 'DEN_OCU': {},
                    'OCUs': {'ocu_1': {'taxa': ['otu_1']}, 'ocu_2': {'taxa': ['otu_2']}, 'ocu_3': {'taxa': ['otu_3']}}
                }
            }}
            return data[file_name]

        monkeypatch.setattr('src.compores.utils.load_file', mock_load_file)

        result, response = fetch_synthetic_analysis_input_data(
            path_to_outputs='test/path',
            microbiome_file_name='test-micro-biome',
            balance_method='CLR',
            response_tag='response1'
        )

        assert response == 'response1'
        assert result['rmse'] == [0.1, 0.3]
        assert result['slope'] == [1.1, 1.3]
        assert result['intercept'] == [2.1, 2.3]
        assert result['num_ocu'] == [['otu_1, otu_3'], ['otu_1']]
        assert result['den_ocu'] == [['otu_1', 'otu_2', 'otu_3'], ['otu_1', 'otu_2', 'otu_3']]

    def test_deduplicate_synthetic_analysis_input_data(self):

        tests = [
            {
                'input_data': {
                    'rmse': [0.1253, 0.12487841, 0.3],
                    'slope': [1.1, 1.2, 1.3],
                    'intercept': [2.1, 2.2, 2.3],
                    'num_ocu': [['otu_1', 'otu_3'], ['otu_1', 'otu_4'], ['otu_2']],
                    'den_ocu': [['otu_2'], ['otu_5'], ['otu_3']],
                    'ocu_number': [1, 2, 3]
                },
                'expected_output': {
                    'rmse': np.array([0.125, 0.3]),
                    'slope': np.array([1.1, 1.3]),
                    'intercept': np.array([2.1, 2.3]),
                    'num_ocu': [['otu_1', 'otu_3'], ['otu_2']],
                    'den_ocu': [['otu_2'], ['otu_3']],
                    'ocu_number': [1, 3]
                }
            },
            {
                'input_data': {
                    'rmse': [2563, 8963, 8962],
                    'slope': [1.1, 1.2, 1.3],
                    'intercept': [2.1, 2.2, 2.3],
                    'num_ocu': [['otu_1', 'otu_3'], ['otu_1', 'otu_4'], ['otu_2']],
                    'den_ocu': [['otu_2'], ['otu_5'], ['otu_3']],
                    'ocu_number': [1, 2, 3]
                },
                'expected_output': {
                    'rmse': np.array([2560, 8960]),
                    'slope': np.array([1.1, 1.2]),
                    'intercept': np.array([2.1, 2.2]),
                    'num_ocu': [['otu_1', 'otu_3'], ['otu_1', 'otu_4']],
                    'den_ocu': [['otu_2'], ['otu_5']],
                    'ocu_number': [1, 2]
                }
            }
        ]

        for test in tests:
            input_data = test['input_data']
            expected_output = test['expected_output']
            result = deduplicate_synthetic_analysis_input_data(input_data)
            for k, v in result.items():
                if isinstance(v[0], list):
                    for i in range(len(v)):
                        assert v[i] == expected_output[k][i]
                else:
                    assert np.array_equal(v, expected_output[k])


    def test_is_considered_imputed_sample(self, setup_teardown_is_considered_imputed_sample):
        sample_list, params, expected_result = setup_teardown_is_considered_imputed_sample
        for i, sample_name in enumerate(sample_list):
            result = is_considered_imputed_sample(sample_name, **params)
            assert result == expected_result[i], f"Failed for sample: {sample_name}"


class TestBootstrapPValue:
    @pytest.fixture
    def shuffled_values(self):
        return {
            "standard_case": {
                "pcc_array": np.array([0.2, 0.5, np.nan, 0.7, 0.9]),
                "observed_value": 0.6,
                "expected_p_value": 0.6
            },
            "all_below": {
                "pcc_array": np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
                "observed_value": 0.6,
                "expected_p_value": 1/6
            },
            "all_above": {
                "pcc_array": np.array([0.6, 0.7, 0.8, 0.9, 1.0]),
                "observed_value": 0.6,
                "expected_p_value": 1.0
            },
            "empty": {
                "pcc_array": np.array([np.nan, np.nan, np.nan]),
                "observed_value": 0.6,
                "expected_p_value": 1.0
            }
        }

    def test_bootstrap_p_value(self, shuffled_values):
        for case, data in shuffled_values.items():
            pcc_array = data["pcc_array"]
            observed_value = data["observed_value"]
            expected_p_value = data["expected_p_value"]
            p_value = bootstrap_p_value(observed_value, pcc_array)
            assert p_value == pytest.approx(expected_p_value, rel=1e-5), f"Failed for case: {case}"


class TestGevCalculations:

    @pytest.fixture(scope="class")
    def pcc_array_for_gev_estimation(self):
        yield np.array(
            [0.90928988, 0.94838389, 0.84483211, 0.92957831, 0.93708107, np.nan,
             0.93088696, 0.92010117, 0.95212665, 0.84599811, 0.9114227,
             0.90928988, 0.94838389, 0.84483211, 0.82386253, 0.87901496,
             0.92101806, 0.88090639, 0.92682886, 0.8729168, 0.88103355,
             0.90928988, 0.94838389, 0.84483211, 0.82386253, 0.8955945,
             0.90227359, 0.92010117, 0.91710954, 0.85497667, 0.9114227,
             0.90928988, 0.94838389, 0.84483211, 0.82386253, 0.84864964,
             0.90227359, 0.95801239, 0.95212665, 0.84599811, 0.9114227,
             0.90928988, 0.94838389, 0.84483211, 0.82386253, 0.87901496,
             0.92101806, 0.91226387, 0.91687404, 0.91997031, 0.9114227]
        )

    def test_fit_gev_distribution(self):
        # Generate random values between -1 and 1 to simulate correlation values
        shuffled_values = np.random.uniform(-1, 1, size=1000)
        shape, loc, scale = fit_gev_distribution(shuffled_values)
        assert shape != 0 and loc != 0 and scale != 0
        assert scale > 0

    def test_fit_gev_distribution_with_nans(self):
        # Generate random values between -1 and 1 to simulate correlation values
        shuffled_values = np.random.uniform(-1, 1, size=1000)
        # Replace some values with NaN
        shuffled_values[::10] = np.nan
        shape, loc, scale = fit_gev_distribution(shuffled_values)
        assert shape != 0 and loc != 0 and scale != 0

    def test_calculate_p_value_with_gev(self):
        tests = [
            {'shape': 0.509356162713149, 'loc': 0.9763952782918282, 'scale': 0.0390383392193931,
             'observed_value': -0.980841181208373},
            {'shape': -0.509356162713149, 'loc': 0.9763952782918282, 'scale': 0.00390383392193931,
             'observed_value': 0.990841181208373},
        ]

        for test in tests:
            resulting_p_value = calculate_p_value_with_gev(
                test['observed_value'], test['shape'], test['loc'], test['scale']
            )
            assert 0 < resulting_p_value <= 1

    def test_calculate_p_value_with_gev_with_negative_shape(self):
        # Make sure observed_value stays in (-1, 1)
        shape = -0.5
        loc = 0.5
        scale = 0.2
        observed_correlation_value = 0.99999
        resulting_p_value = calculate_p_value_with_gev(observed_correlation_value, shape, loc, scale)
        assert 0 < resulting_p_value < 1

    def test_out_of_bootstrap_observed_pcc_value_positive_shape(self):
        bootstrap_pcc_array = np.array([0.2, 0.5, np.nan, 0.7, 0.9])
        shape, loc, scale = fit_gev_distribution(bootstrap_pcc_array)
        observed_pcc_value = 0.9999
        resulting_p_value = calculate_p_value_with_gev(observed_pcc_value, shape, loc, scale)
        assert shape > 0
        assert 0 < resulting_p_value <= 1

    def test_out_of_bootstrap_observed_pcc_value_negative_shape(self):
        bootstrap_pcc_array = np.array([0.2, 0.1, 0.3, 0.4, 0.5, np.nan, 0.7, 0.9])
        shape, loc, scale = fit_gev_distribution(bootstrap_pcc_array)
        observed_pcc_value = 0.9999
        resulting_p_value = calculate_p_value_with_gev(observed_pcc_value, shape, loc, scale)
        assert shape < 0
        assert 0 < resulting_p_value <= 1

    def test_tail_case_gev_bootstrapped_p_value(self, pcc_array_for_gev_estimation):
        shape, loc, scale = fit_gev_distribution(pcc_array_for_gev_estimation)
        observed_pcc_val = max(pcc_array_for_gev_estimation) * 1.014
        resulting_p_val_log = - np.log(tail_case_gev_bootstrapped_p_value(observed_pcc_val, shape, loc, scale))
        assert 11 < resulting_p_val_log <= 19

    def test_tail_case_gev_weight_adjusted_p_value(self, pcc_array_for_gev_estimation):
        observed_pcc_val = max(pcc_array_for_gev_estimation) * 1.043827
        resulting_p_val, g = tail_case_gev_weight_adjusted_p_value(observed_pcc_val, pcc_array_for_gev_estimation)
        assert 0 < resulting_p_val < 1e-9

    def test_gev_p_value(self, pcc_array_for_gev_estimation):
        observed_pcc_val = max(pcc_array_for_gev_estimation) * 1.043827
        for correction_method in ['bootstrap', 'weight', None]:
            resulting_p_val, g = gev_p_value(observed_pcc_val, pcc_array_for_gev_estimation, correction_method)
            assert 1e-11 <= resulting_p_val < 1e-10