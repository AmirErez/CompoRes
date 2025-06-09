import logging
import os

import pandas as pd
import numpy as np
import pytest

from src.compores.preprocessing import Preprocessor


class TestPerformCMultReplImputation:
    @pytest.fixture(scope="function")
    def logger_mock(self):
        logger = logging.getLogger(__name__)

        return logger

    def test_cmultrepl_imputation_performed(self, logger_mock, tmp_path):

        with open(os.path.join(tmp_path, "test_df.csv"), 'w') as f:
            pd.DataFrame([]).to_csv(f, index=False)

        with open(os.path.join(tmp_path, "test_series.csv"), 'w') as f:
            pd.Series([]).to_csv(f, index=False)

        test_data = {'SampleID': ['C10.d4', 'C11.d4', 'C7.d4', 'C8.d4', 'N10.d4'],
                     'f_A_1683': [0, 0, 0, 0.33, .75],
                     'f_B_1707': [0, .15, .7, .55, 0],
                     'f_C_99': [1, .85, .3, .12, .25]
                     }
        test_df = pd.DataFrame(test_data).set_index('SampleID')

        tested_obj = Preprocessor(
            logger_mock, '', '', '',
            os.path.join(tmp_path, "test_df.csv"),
            os.path.join(tmp_path, "test_series.csv"),
            '', '', '', '', '', '', '', '', 1
        )

        expected_imputed_samples_dict = {'f_A_1683': ['C10.d4', 'C11.d4', 'C7.d4'], 'f_B_1707': ['C10.d4', 'N10.d4']}
        expected_imputed_df = pd.DataFrame(
            {
                'SampleID': ['C10.d4', 'C11.d4', 'C7.d4', 'C8.d4', 'N10.d4'],
                'f_A_1683': [0.15610688, 0.1689812, 0.1722931, .33, .6833713],
                'f_B_1707': [0.07497000, .12465282, .57939483, .55, 0.08883827],
                'f_C_99': [0.76892312, .70636598, .24831207, .12, .22779043]
            }
        ).set_index('SampleID')
        # Perform imputation
        tested_obj.perform_cmultrepl_imputation(test_df)

        assert not np.isnan(tested_obj.imputed_x.values).any()  # There should be no NaN values after imputation
        assert tested_obj.imputed_samples_dict == expected_imputed_samples_dict
        assert np.allclose(tested_obj.imputed_x.values, expected_imputed_df.values)

    def test_cmultrepl_no_imputation_performed(self, logger_mock, tmp_path):

        with open(os.path.join(tmp_path, "test_df.csv"), 'w') as f:
            pd.DataFrame([]).to_csv(f, index=False)

        with open(os.path.join(tmp_path, "test_series.csv"), 'w') as f:
            pd.Series([]).to_csv(f, index=False)

        test_data = {
            'SampleID': ['C10.d4', 'C11.d4', 'C7.d4', 'C8.d4', 'N10.d4'],
            'f_A_1683': [0.145418, 0.145460, 0.145913, .23, 0.560669],
            'f_B_1707': [0.444383, 0.726359, 0.256226, 0.55, 0.163180],
            'f_C_99': [0.410199, 0.128181, 0.597861, 0.22, 0.276151]
        }
        test_df = pd.DataFrame(test_data).set_index('SampleID')

        tested_obj = Preprocessor(
            logger_mock, '', '', '',
            os.path.join(tmp_path, "test_df.csv"),
            os.path.join(tmp_path, "test_series.csv"),
            '', '', '', '', '', '', '', '', 1
        )

        # Perform imputation
        tested_obj.perform_cmultrepl_imputation(test_df)

        # Check that the initial DataFrame is unchanged
        assert tested_obj.imputed_x.equals(test_df)

    def test_cmultrepl_wrong_method(self, logger_mock, tmp_path):

        with open(os.path.join(tmp_path, "test_df.csv"), 'w') as f:
            pd.DataFrame([]).to_csv(f, index=False)

        with open(os.path.join(tmp_path, "test_series.csv"), 'w') as f:
            pd.Series([]).to_csv(f, index=False)

        tested_obj = Preprocessor(
            logger_mock, '', '', '',
            os.path.join(tmp_path, "test_df.csv"),
            os.path.join(tmp_path, "test_series.csv"),
            '', '', '', '', '', '', '', '', 1
        )

        tested_method = 'user'
        expected_message = "Invalid method. Supported methods: 'GBM', 'SQ', 'BL'."

        # Perform imputation
        with pytest.raises(ValueError) as exc_info:
            tested_obj.perform_cmultrepl_imputation(pd.DataFrame([]), method=tested_method)

        # Check the exception message
        assert str(exc_info.value) == expected_message

    def test_cmultrepl_unlabeled_zero_counts(self, logger_mock, tmp_path):
        with open(os.path.join(tmp_path, "test_df.csv"), 'w') as f:
            pd.DataFrame([]).to_csv(f, index=False)

        with open(os.path.join(tmp_path, "test_series.csv"), 'w') as f:
            pd.Series([]).to_csv(f, index=False)

        test_data = {
            'SampleID': ['C10.d4', 'C11.d4', 'C7.d4', 'C8.d4', 'N10.d4'],
            'f_A_1683': [0.145418, 0.145460, 0.145913, .23, 0.560669],
            'f_B_1707': [0.444383, 0.726359, 0.256226, np.nan, 0.163180],
            'f_C_99': [0.410199, 0.128181, 0.597861, 0.77, 0.276151]
        }
        test_df = pd.DataFrame(test_data).set_index('SampleID')

        tested_obj = Preprocessor(
            logger_mock, '', '', '',
            os.path.join(tmp_path, "test_df.csv"),
            os.path.join(tmp_path, "test_series.csv"),
            '', '', '', '', '', '', '', '', 1
        )

        expected_message = "NaN values not labelled as count zeros were found in the data set"

        with pytest.raises(ValueError) as exc_info:
            tested_obj.perform_cmultrepl_imputation(test_df, label=None)

        assert str(exc_info.value) == expected_message

    def test_cmultrepl_adjusted(self, logger_mock, tmp_path):
        with open(os.path.join(tmp_path, "test_df.csv"), 'w') as f:
            pd.DataFrame([]).to_csv(f, index=False)

        with open(os.path.join(tmp_path, "test_series.csv"), 'w') as f:
            pd.Series([]).to_csv(f, index=False)

        tested_obj = Preprocessor(
            logger_mock, '', '', '',
            os.path.join(tmp_path, "test_df.csv"),
            os.path.join(tmp_path, "test_series.csv"),
            '', '', '', '', '', '', '', '', 1
        )

        input_frac = 0.65
        test_data = {
            'adjust_true': {
                'expected_data': {'SampleID': ['C10.d4', 'C11.d4', 'C7.d4', 'C8.d4', 'N10.d4'],
                                  'f_A_1683': [0.170163, 0.170220, 0.170841, .23, .67],
                                  'f_B_1707': [0.52, 0.85, 0.3, 0.55, 0.3 * input_frac],
                                  'f_C_99': [0.48, 0.15, 0.7, 0.22, 0.33]
                                  },
                'adjust': True
            },
            'adjust_false': {
                'expected_data': {'SampleID': ['C10.d4', 'C11.d4', 'C7.d4', 'C8.d4', 'N10.d4'],
                                  'f_A_1683': [0.170163, 0.170220, 0.170841, .23, .67],
                                  'f_B_1707': [0.52, 0.85, 0.3, 0.55, 0.45076225],
                                  'f_C_99': [0.48, 0.15, 0.7, 0.22, 0.33]
                                  },
                'adjust': False
            }
        }

        for test_case in test_data:

            # Build expected DataFrame and normalize it
            expected_data = test_data[test_case]['expected_data']
            expected_df = pd.DataFrame(expected_data).set_index('SampleID')
            expected_df = expected_df.div(expected_df.sum(axis=1), axis=0)

            tested_obj.perform_cmultrepl_imputation(expected_df, adjust=test_data[test_case]['adjust'])
            imputed_df = tested_obj.imputed_x

            assert np.allclose(imputed_df, expected_df)
