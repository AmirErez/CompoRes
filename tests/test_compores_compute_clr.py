import os

import pytest
import pandas as pd
import numpy as np

from src.compores.compores_compute import CompoRes


class TestComputeClr:

    @pytest.fixture(scope="function")
    def basic_input_output_data(self):
        input_counts = pd.DataFrame({
            'A': [.3, .285714, .416667, .444444, .441176],
            'B': [.5, .52381, .083333, .259259, .264706],
            'C': [.2, .190476, .5, .296296, .294118]
        })
        input_target_variable = pd.Series([1.0, 2.0, 3.0, 4.0, 3.0])

        max_ind = 'A'
        expected_Tab_var = {"NUM": max_ind, "DEN": "CLR"}

        # Compute clr transformation
        log_counts_array = np.log(input_counts).to_numpy()
        geometric_mean = np.exp(np.mean(log_counts_array, axis=1))
        clr_transformed = np.log(np.exp(np.log(input_counts)) / geometric_mean[:, np.newaxis])

        expected_B = clr_transformed.iloc[:, 0].to_numpy()

        yield input_counts, input_target_variable, expected_Tab_var, expected_B

        # cleanup .log files in the test directory
        for file in os.listdir('.'):
            if file.endswith('.log'):
                os.remove(file)

    @pytest.fixture(scope="function")
    def non_trivial_input_output_data(self):
        input_counts = pd.DataFrame({
            'A': [.95, .685714, .416667, .444444, .441176],
            'B': [.03, .22381, .083333, .259259, .264706],
            'C': [.02, .090476, .5, .296296, .294118]
        })
        input_target_variable = pd.Series([1.0, 2.0, 3.0, 4.0, 2.5])

        # Compute clr transformation
        log_counts_array = np.log(input_counts).to_numpy()
        geometric_mean = np.exp(np.mean(log_counts_array, axis=1))
        clr_transformed = np.log(np.exp(np.log(input_counts)) / geometric_mean[:, np.newaxis])

        expected_data = {
            'spearman': {
                'expected_Tab_var': {"NUM": 'C', "DEN": "CLR"},
                'expected_B': clr_transformed.iloc[:, 2].to_numpy()
            },
            'pearson': {
                'expected_Tab_var': {"NUM": 'A', "DEN": "CLR"},
                'expected_B': clr_transformed.iloc[:, 0].to_numpy()
            }
        }

        yield input_counts, input_target_variable, expected_data

        # cleanup .log files in the test directory
        for file in os.listdir('.'):
            if file.endswith('.log'):
                os.remove(file)

    def test_compute_clr_general(self, basic_input_output_data):

        input_counts, input_target_variable, expected_dict, expected_balance = basic_input_output_data

        input_log_counts = np.log(input_counts)

        correlation_types_to_test = ['pearson', 'spearman']

        for corr_type in correlation_types_to_test:
            tested_obj = CompoRes(
                input_counts, input_target_variable, 3, '', {}, corr_type=corr_type, balance_method='CLR'
            )
            actual_result = tested_obj.compute_clr(input_log_counts, input_target_variable)
            assert actual_result[0] == expected_dict
            assert np.allclose(actual_result[1], expected_balance)

            tested_obj.cleanup_logger_file_handlers()

    def test_compute_clr_different_corr_types(self, non_trivial_input_output_data):

        input_counts, input_target_variable, expected_result = non_trivial_input_output_data

        input_log_counts = np.log(input_counts)

        correlation_types_to_test = ['pearson', 'spearman']

        for corr_type in correlation_types_to_test:
            tested_obj = CompoRes(
                input_counts, input_target_variable, 3, '', {},
                corr_type=corr_type, balance_method='CLR'
            )
            actual_result = tested_obj.compute_clr(input_log_counts, input_target_variable)
            assert actual_result[0] == expected_result[corr_type]['expected_Tab_var']
            assert np.allclose(actual_result[1], expected_result[corr_type]['expected_B'])

            tested_obj.cleanup_logger_file_handlers()
