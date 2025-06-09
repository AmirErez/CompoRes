import os

import pytest
import pandas as pd
import numpy as np

from src.compores.compores_compute import CompoRes


class TestComputeFirstBalance:

    @pytest.fixture(scope="function")
    def basic_input_output_data(self):
        input_counts = pd.DataFrame({
            'A': [.3, .285714, .416667, .444444, .441176],
            'B': [.5, .52381, .083333, .259259, .264706],
            'C': [.2, .190476, .5, .296296, .294118]
        })
        input_target_variable = pd.Series([1.0, 2.0, 3.0, 4.0, 3.0])

        POS = 'A'
        NEG = 'B'
        expected_Tab_var = {"NUM": POS, "DEN": NEG}

        S1 = np.log(input_counts)[POS]
        S2 = np.log(input_counts)[NEG]
        expected_B = np.sqrt(1 / 2) * (S1 - S2)

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
        input_target_variable = pd.Series([1.0, 2.0, 3.0, 4.0, 3.0])

        expected_data = {
            'spearman': {
                'expected_Tab_var': {"NUM": 'B', "DEN": 'A'},
                'expected_B': np.sqrt(1 / 2) * (np.log(input_counts)['B'] - np.log(input_counts)['A'])
            },
            'pearson': {
                'expected_Tab_var': {"NUM": 'C', "DEN": 'A'},
                'expected_B': np.sqrt(1 / 2) * (np.log(input_counts)['C'] - np.log(input_counts)['A'])
            }
        }

        yield input_counts, input_target_variable, expected_data

        # cleanup .log files in the test directory
        for file in os.listdir('.'):
            if file.endswith('.log'):
                os.remove(file)

    def test_compute_balances_general(self, basic_input_output_data):

        input_counts, input_target_variable, expected_dict, expected_balance = basic_input_output_data

        input_log_counts = np.log(input_counts)

        correlation_types_to_test = ['pearson', 'spearman']
        balance_methods_to_test = ['CLR', 'pairs']

        for corr_type in correlation_types_to_test:
            for balance_method in balance_methods_to_test:
                tested_obj = CompoRes(
                    input_counts, input_target_variable, 3, '', {}, corr_type=corr_type,
                    balance_method=balance_method
                )
                actual_result = tested_obj.compute_balances(input_log_counts, input_target_variable)
                assert actual_result[0] == expected_dict
                assert np.allclose(actual_result[1], expected_balance)

                tested_obj.cleanup_logger_file_handlers()

    def test_compute_balances_different_corr_types(self, non_trivial_input_output_data):

        input_counts, input_target_variable, expected_result = non_trivial_input_output_data

        input_log_counts = np.log(input_counts)

        correlation_types_to_test = ['pearson', 'spearman']
        balance_methods_to_test = ['CLR', 'pairs']

        for corr_type in correlation_types_to_test:
            for balance_method in balance_methods_to_test:
                tested_obj = CompoRes(
                    input_counts, input_target_variable, 3, '', {},
                    corr_type=corr_type, balance_method=balance_method
                )
                actual_result = tested_obj.compute_balances(input_log_counts, input_target_variable)
                assert actual_result[0] == expected_result[corr_type]['expected_Tab_var']
                assert np.allclose(actual_result[1], expected_result[corr_type]['expected_B'])

                tested_obj.cleanup_logger_file_handlers()
