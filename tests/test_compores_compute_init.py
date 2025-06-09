import pandas as pd
import numpy as np
import pytest
import os

from src.compores.compores_compute import CompoRes


class TestCompoResInit:

    @pytest.fixture(scope="function")
    def setup_teardown_norm_non_impute(self):
        data = {'SampleID': ['C10.d4', 'C11.d4', 'C7.d4', 'C8.d4', 'N10.d4'],
                'f_A_1683': [0.145418, 0.145460, 0.145913, .23, 0.560669],
                'f_B_1707': [0.444383, 0.726359, 0.256226, 0.55, 0.163180],
                'f_C_99': [0.410199, 0.128181, 0.597861, 0.22, 0.276151]
                }
        df = pd.DataFrame(data).set_index('SampleID')

        response_series = pd.DataFrame(
            {
                'SampleID': ['C10.d4', 'C11.d4', 'C7.d4', 'C8.d4', 'N10.d4'],
                'GO:0045071_enh': [1.0, 1.005, 1.021461975, 1.0, 1.0]
            }
        ).set_index('SampleID')['GO:0045071_enh']

        yield df, response_series

        # cleanup .log files in the test directory
        for file in os.listdir('.'):
            if file.endswith('.log'):
                os.remove(file)

    def test_cmultrepl_non_normalized_input(self, caplog):
        # Create a sample DataFrame without zeros
        test_data = {'SampleID': ['C10.d4', 'C11.d4', 'C7.d4', 'C8.d4', 'N10.d4'],
                     'f_A_1683': [1, 3, np.nan, 7, 9],
                     'f_B_1707': [5, 4, 3, 3, 1],
                     'f_C_99': [1, 0, 3, 4, 5]
                     }
        test_df = pd.DataFrame(test_data).set_index('SampleID')

        test_response_series = pd.DataFrame(
            {
                'SampleID': ['C10.d4', 'C11.d4', 'C7.d4', 'C8.d4', 'N10.d4'],
                'GO:0045071_enh': [1.0, 1.005, 1.021461975, 1.0, 1.0]
            }
        ).set_index('SampleID')['GO:0045071_enh']

        exc_message_value_err_part = "Not all rows sum up to one, does the input matrix contain normalized proportions?"
        exc_message_system_exit_part = "Error creating CompoRes object: "

        with pytest.raises(SystemExit) as exc_info:
            CompoRes(test_df, test_response_series, 3, '', {}, balance_method='CLR')

        assert exc_info.type == SystemExit
        assert exc_info.value.code == 1
        assert exc_message_value_err_part in caplog.text
        assert exc_message_system_exit_part in caplog.text

        # cleanup .log files in the test directory
        for file in os.listdir('.'):
            if file.endswith('.log'):
                os.remove(file)

    def test_compores_init_wrong_balance_method(self, setup_teardown_norm_non_impute, caplog):
        test_df, test_response_series = setup_teardown_norm_non_impute
        tested_balance_method = 'unknown'
        error_msg = f"Invalid method '{tested_balance_method}'. Supported methods: 'pairs', 'CLR'."
        error_msg = f"Error creating CompoRes object: {error_msg}"

        with pytest.raises(SystemExit) as exc_info:
            CompoRes(
                test_df, test_response_series, 3, '', {},
                balance_method=tested_balance_method
            )

        assert exc_info.type == SystemExit
        assert exc_info.value.code == 1
        assert error_msg in caplog.text

    def test_compores_init_wrong_corr_type(self, setup_teardown_norm_non_impute, caplog):
        test_df, test_response_series = setup_teardown_norm_non_impute
        tested_corr_type = 'unknown'
        error_msg = f"Invalid correlation type '{tested_corr_type}'. Supported types: 'pearson', 'spearman'."
        error_msg = f"Error creating CompoRes object: {error_msg}"

        with pytest.raises(SystemExit) as exc_info:
            CompoRes(
                test_df, test_response_series, 3, '', {},
                corr_type=tested_corr_type, balance_method='CLR'
            )

        assert exc_info.type == SystemExit
        assert exc_info.value.code == 1
        assert error_msg in caplog.text
