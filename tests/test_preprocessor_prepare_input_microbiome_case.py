import logging
import shutil
from pathlib import Path

import numpy as np
import pytest
import os
import pandas as pd
from src.compores.preprocessing import Preprocessor
from src.compores.exceptions_module import NonNumericDataFrameError, EmptyDataFrame, MinDataFrame, \
    NegativeValuesDataFrameError


class TestPrepareMicrobiome:

    @pytest.fixture(scope="function")
    def logger_mock(self):
        logger = logging.getLogger(__name__)

        return logger

    @pytest.fixture(scope="function")
    def setup_teardown_response_file(self, tmp_path):
        # Put a test response input file in the temporary test directory
        input_response_file_path = tmp_path / "test_response.tsv"
        input_response_data = {
            'SampleID': ['C10.d4', 'C11.d4', 'C7.d4', 'C8.d4', 'N10.d4'],
            'GO:0045071_enh': [1.0, 0, 1.005, 1.021461975, 1.0],
            'GO:0032823_enh': [1.006384536, 0, 3, 3, 3.006384531],
            'GO:0045887_enh': [2, 2, 2.011461975, 0, 3.005156721],
        }
        input_response_df = pd.DataFrame(input_response_data).set_index('SampleID')
        input_response_df.to_csv(input_response_file_path, sep="\t", index=False)

        yield input_response_file_path

    @pytest.fixture(scope="function")
    def setup_teardown_remove_zero_rows(self, tmp_path):
        # Put a test microbiome input file in the temporary test directory
        input_microbiome_data = {
            'SampleID': ['C10.d4', 'C11.d4', 'C7.d4', 'C8.d4', 'N10.d4'],
            'f_A_1683': [.0, 0, .005, 0.021461975, .0],
            'f_B_1707': [0.006384536, 0, 0, 0, 0.006384531],
            'f_C_1645': [0.006583726, 0, 0, 0, 0.008594536],
            'f_C_181': [0, 0, 0.011461975, 0, 0.005156721],
        }
        input_microbiome_df = pd.DataFrame(input_microbiome_data)
        input_microbiome_file_path = tmp_path / "test.tsv"
        input_microbiome_df.to_csv(input_microbiome_file_path, sep="\t", index=False)

        # Define the expected output file path
        expected_output_microbiome_file_path = os.path.join("preprocessing_results", "microbiome", "test.tsv")

        # Defined the expected microbiome output dataframe
        expected_microbiome_data = {
            'SampleID': ['C10.d4', 'C7.d4', 'C8.d4', 'N10.d4'],
            'f_A_1683': [0.0, 0.30373, 1, .0],
            'f_B_1707': [0.49232, 0, 0, 0.31707],
            'f_C_1645': [0.50768, 0, 0, 0.42683],
            'f_C_181': [0, 0.69627, 0, 0.25610],
        }
        expected_output_microbiome_df = pd.DataFrame(expected_microbiome_data).set_index('SampleID')

        # Yield test_microbiome_df_path and control to the test class
        yield input_microbiome_file_path, expected_output_microbiome_file_path, expected_output_microbiome_df

        if os.path.exists("preprocessing_results"):
            shutil.rmtree("preprocessing_results")

    @pytest.fixture(scope="function")
    def setup_teardown_catch_non_numeric_values(self, tmp_path):
        # Put a test microbiome input file in the temporary test directory
        input_microbiome_data = {
            'SampleID': ['C10.d4', 'C11.d4', 'C7.d4', 'C8.d4', 'N10.d4'],
            'f_A_1683': [.0, .0, .0, .0, .0],
            'f_B_1707': [0.006384536, None, 0, 0, 0],  # Introduce a non-numeric value
            'f_C_1645': [0, 0, 0, 0, 0.008594536],
            'f_C_181': [0, 0, 0.011461975, 0, 0.005156721],
            'f_C_99': [0, "non-numeric", .0, .0, .0]
        }
        input_microbiome_df = pd.DataFrame(input_microbiome_data)
        input_microbiome_file_path = tmp_path / "test_microbiome.tsv"
        input_microbiome_df.to_csv(input_microbiome_file_path, sep="\t", index=False)

        # Define the expected exception message
        expected_exception_message = "The input data contains non-numeric values."

        # Yield test_microbiome_df_path and control to the test class
        yield input_microbiome_file_path, expected_exception_message

    @pytest.fixture(scope="function")
    def setup_teardown_catch_negative_values(self, tmp_path):
        # Put a test microbiome input file in the temporary test directory
        input_microbiome_data = {
            'SampleID': ['C10.d4', 'C11.d4', 'C7.d4', 'C8.d4', 'N10.d4'],
            'f_A_1683': [.0, .0, .0, -2.5634, .0],  # Introduce a negative value
            'f_B_1707': [0.006384536, 0, -1, 0, 0],  # Introduce a negative value
            'f_C_1645': [0, 0, 0, 0, 0.008594536],
            'f_C_181': [0, 0, 0.011461975, 0, 0.005156721],
            'f_C_99': [0, 0.264, .0, .0, .0]
        }
        input_microbiome_df = pd.DataFrame(input_microbiome_data)
        input_microbiome_file_path = tmp_path / "test_microbiome.tsv"
        input_microbiome_df.to_csv(input_microbiome_file_path, sep="\t", index=False)

        # Define the expected exception message
        expected_exception_message = "The input microbiome data contains negative values."

        # Yield test_microbiome_df_path and control to the test class
        yield input_microbiome_file_path, expected_exception_message

    @pytest.fixture(scope="function")
    def setup_teardown_catch_only_zeros_df(self, tmp_path):
        # Put a test microbiome input file in the temporary test directory
        input_microbiome_data = {
            'SampleID': ['C10.d4', 'C11.d4', 'C7.d4', 'C8.d4', 'N10.d4'],
            'f_A_1683': [.0, .0, .0, .0, .0],
            'f_B_1707': [0, 0, 0, 0, 0],
            'f_C_1645': [0, 0, 0, 0, 0],
            'f_C_181': [0, 0, 0, 0, 0.],
            'f_C_99': [0, 0, .0, .0, .0]
        }
        input_microbiome_df = pd.DataFrame(input_microbiome_data)
        input_microbiome_file_path = tmp_path / "test_microbiome.tsv"
        input_microbiome_df.to_csv(input_microbiome_file_path, sep="\t", index=False)

        # Define the expected exception message
        expected_exception_message = "The resulting DataFrame is empty."

        # Yield test_microbiome_df_path and control to the test class
        yield input_microbiome_file_path, expected_exception_message

    @pytest.fixture(scope="function")
    def setup_teardown_check_df_size_validity(self, tmp_path):
        # Put a test microbiome input file in the temporary test directory
        input_microbiome_data = {
            'SampleID': ['C10.d4', 'C11.d4', 'C7.d4', 'C8.d4', 'N10.d4'],
            'f_A_1683': [.0, .0, .0, .0, .0],
            'f_B_1707': [0, 0, 0, 1, 0],
            'f_C_1645': [0, 0, 1, 2, 0],
            'f_C_181': [0, 0, 0, 0, 0.],
            'f_C_99': [0, 0, .3, .0, .0]
        }
        input_microbiome_df = pd.DataFrame(input_microbiome_data)
        input_microbiome_file_path = tmp_path / "test_microbiome.tsv"
        input_microbiome_df.to_csv(input_microbiome_file_path, sep="\t", index=False)

        # Put a test response input file in the temporary test directory
        input_response_file_path = tmp_path / "test_response.tsv"
        input_response_data = {
            'SampleID': ['C10.d4', 'C11.d4', 'C7.d4', 'C8.d4', 'N10.d4'],
            'GO:0045071_enh': [1.0, 0, 1.005, 1.021461975, 1.0],
            'GO:0032823_enh': [1.006384536, 0, 3, 3, 3.006384531],
            'GO:0045887_enh': [2, 2, 2.011461975, 0, 3.005156721],
        }
        input_response_df = pd.DataFrame(input_response_data).set_index('SampleID')
        input_response_df.to_csv(input_response_file_path, sep="\t", index=False)

        # Define the expected exception message
        expected_exception_message = 'The resulting DataFrame should have at least 3 rows and 3 columns for OTU, ' \
                                     'and at least 3 rows and 1 column for response.'

        # Yield test_microbiome_df_path and control to the test class
        yield input_microbiome_file_path, input_response_file_path, expected_exception_message

    @pytest.fixture
    def setup_teardown_check_number_of_non_zero_values_in_cols(self, tmp_path):
        # Create a sample microbiome data frame for testing
        input_microbiome_data = {'SampleID': ['C10.d4', 'C11.d4', 'C7.d4', 'C8.d4', 'N10.d4'],
                                 'f_A_1683': [0, 0, 0, 1, 2],
                                 'f_B_1707': [0, 1, 2, 3, 0],
                                 'f_C_1645': [1, 0, 0, 0, 0],
                                 'f_C_181': [0, 0, 0, 0, 0],
                                 'f_C_99': [1, 2, 3, 4, 5]
                                 }
        input_microbiome_df = pd.DataFrame(input_microbiome_data)
        input_microbiome_file_path = tmp_path / "test.tsv"
        input_microbiome_df.to_csv(input_microbiome_file_path, sep="\t", index=False)
        input_threshold = 0.2
        # Define the expected output file path
        expected_output_microbiome_file_path = os.path.join("preprocessing_results", "microbiome", "test.tsv")
        expected_microbiome_data = {'SampleID': ['C10.d4', 'C11.d4', 'C7.d4', 'C8.d4', 'N10.d4'],
                                    'f_A_1683': [0, 0, 0, 1 / 8, 2 / 7],
                                    'f_B_1707': [0, 1 / 3, 2 / 5, 3 / 8, 0],
                                    'f_C_99': [1, 2 / 3, 3 / 5, 4 / 8, 5 / 7]
                                    }
        expected_df = pd.DataFrame(expected_microbiome_data).set_index('SampleID')
        yield input_microbiome_file_path, input_threshold, expected_output_microbiome_file_path, expected_df

        # Cleanup: Remove the to be created output directory and its content
        if os.path.exists("preprocessing_results"):
            shutil.rmtree("preprocessing_results")

    @pytest.fixture
    def setup_teardown_check_df_normalization(self, tmp_path):
        # Create a sample microbiome data frame for testing
        input_microbiome_data = {'SampleID': ['C10.d4', 'C11.d4', 'C7.d4'],
                                 'f_A_1683': [1, 4, 5],
                                 'f_B_1707': [4, 5, 6],
                                 'f_C_1645': [5, 11, 9],
                                 }
        input_microbiome_df = pd.DataFrame(input_microbiome_data)
        input_microbiome_file_path = tmp_path / "test.tsv"
        input_microbiome_df.to_csv(input_microbiome_file_path, sep="\t", index=False)
        # Define the expected output file path
        expected_output_microbiome_file_path = os.path.join("preprocessing_results", "microbiome", "test.tsv")
        expected_microbiome_data = {'SampleID': ['C10.d4', 'C11.d4', 'C7.d4'],
                                    'f_A_1683': [0.1, 0.2, 0.25],
                                    'f_B_1707': [0.4, 0.25, 0.3],
                                    'f_C_1645': [0.5, 0.55, 0.45]
                                    }
        expected_df = pd.DataFrame(expected_microbiome_data).set_index('SampleID')
        yield input_microbiome_file_path, expected_output_microbiome_file_path, expected_df

        # Cleanup: Remove the to be created output directory and its content
        if os.path.exists("preprocessing_results"):
            shutil.rmtree("preprocessing_results")

    def test_prepare_microbiome_catch_non_existing_file(self, logger_mock, caplog):
        logger = logger_mock
        with pytest.raises(SystemExit) as exc_info:
            preprocess = Preprocessor(
                logger, '', '', '', Path("nonexistent_file_path"), '', '', '', '', '', '', '', '', '', 1
            )
            preprocess.prepare_input("microbiome", Path("nonexistent_file_path"))
        assert exc_info.type == SystemExit
        assert exc_info.value.code == 1
        assert "File `nonexistent_file_path` not found." in caplog.text

    def test_prepare_microbiome_catch_non_numeric_values(
            self, setup_teardown_catch_non_numeric_values, setup_teardown_response_file, logger_mock
    ):
        logger = logger_mock
        input_microbiome_file_path, expected_message = setup_teardown_catch_non_numeric_values
        input_response_file_path = setup_teardown_response_file
        with pytest.raises(NonNumericDataFrameError) as exc_info:
            preprocess = Preprocessor(
                logger, '', '', '', input_microbiome_file_path, input_response_file_path,
                '', '', '', '', '', '', '', '', 1
            )
            preprocess.prepare_input("microbiome", input_microbiome_file_path)
        assert str(exc_info.value) == expected_message

    def test_prepare_microbiome_catch_negative_values(
            self, setup_teardown_catch_negative_values, setup_teardown_response_file, logger_mock
    ):
        logger = logger_mock
        input_microbiome_file_path, expected_message = setup_teardown_catch_negative_values
        input_response_file_path = setup_teardown_response_file
        with pytest.raises(NegativeValuesDataFrameError) as exc_info:
            preprocess = Preprocessor(
                logger, '', '', '', input_microbiome_file_path, input_response_file_path,
                '', '', '', '', '', '', '', '', 1
            )
            preprocess.prepare_input("microbiome", input_microbiome_file_path)
        assert str(exc_info.value) == expected_message

    def test_prepare_microbiome_catch_only_zeros_df(
            self, setup_teardown_catch_only_zeros_df, setup_teardown_response_file, logger_mock
    ):
        logger = logger_mock
        input_microbiome_file_path, expected_message = setup_teardown_catch_only_zeros_df
        input_response_file_path = setup_teardown_response_file
        with pytest.raises(EmptyDataFrame) as exc_info:
            preprocess = Preprocessor(
                logger, '', '', '', input_microbiome_file_path, input_response_file_path,
                '', '', '', '', '', '', '', '', 1
            )
            preprocess.prepare_input("microbiome", input_microbiome_file_path)
        assert str(exc_info.value) == expected_message

    def test_prepare_microbiome_check_df_size_validity(self, setup_teardown_check_df_size_validity, logger_mock):
        logger = logger_mock
        input_microbiome_file_path, input_response_file_path, expected_message = setup_teardown_check_df_size_validity
        with pytest.raises(MinDataFrame) as exc_info:
            preprocess = Preprocessor(
                logger, '', '', '', input_microbiome_file_path, input_response_file_path,
                '', '', '', '', '', '', '', '', 1
            )
            preprocess.prepare_input("microbiome", input_microbiome_file_path)
        assert expected_message in str(exc_info.value)

    def test_prepare_microbiome_remove_zero_rows(
            self, setup_teardown_remove_zero_rows, setup_teardown_response_file, logger_mock
    ):
        logger = logger_mock
        input_microbiome_file_path, expected_output_file_path, expected_output_df = setup_teardown_remove_zero_rows
        input_response_file_path = setup_teardown_response_file

        preprocess = Preprocessor(
            logger, '', '', '', input_microbiome_file_path, input_response_file_path, expected_output_file_path,
            '', '', '', '', '', '', '', 1, imputation_flag=False
        )
        # Call the function on the sample file
        preprocess.prepare_input("microbiome", input_microbiome_file_path)

        # Check if the output file has been created
        assert os.path.exists(expected_output_file_path)
        # Check if the actual output file has the expected content
        assert np.allclose(
            pd.read_csv(expected_output_file_path, sep="\t").set_index('SampleID'),
            expected_output_df,
            rtol=1e-04
        )

    def test_prepare_microbiome_non_zero_values_share_in_columns(
            self, setup_teardown_check_number_of_non_zero_values_in_cols, setup_teardown_response_file, logger_mock
    ):
        logger = logger_mock
        (
            input_microbiome_file_path, filtering_threshold, expected_output_file_path, expected_output_df
        ) = setup_teardown_check_number_of_non_zero_values_in_cols
        input_response_file_path = setup_teardown_response_file
        preprocess = Preprocessor(
            logger, '', '', '', input_microbiome_file_path, input_response_file_path, expected_output_file_path,
            '', '', '', '', '', '', '', 1, imputation_flag=False
        )
        # Call the function on the sample file
        preprocess.prepare_input("microbiome", input_microbiome_file_path)

        processed_data = pd.read_csv(expected_output_file_path, sep="\t", index_col=False)
        processed_data_df = processed_data.set_index(processed_data.columns[0])
        # Check if the actual output file has the expected content
        assert processed_data_df.equals(expected_output_df)

    def test_prepare_microbiome_df_normalization(
            self, setup_teardown_check_df_normalization, setup_teardown_response_file, logger_mock
    ):
        logger = logger_mock
        (
            input_microbiome_file_path,
            expected_output_file_path,
            expected_output_df
        ) = setup_teardown_check_df_normalization
        input_response_file_path = setup_teardown_response_file
        preprocess = Preprocessor(
            logger, '', '', '', input_microbiome_file_path, input_response_file_path, expected_output_file_path,
            '', '', '', '', '', '', '', 1
        )
        # Call the function on the sample file
        preprocess.prepare_input("microbiome", input_microbiome_file_path)

        # Check if the actual output file has the expected content
        assert pd.read_csv(expected_output_file_path, sep="\t", index_col=0).equals(expected_output_df)

# # Run the tests
# if __name__ == "__main__":
#     pytest.main()
