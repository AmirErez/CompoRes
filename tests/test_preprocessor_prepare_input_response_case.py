import logging
import shutil
from pathlib import Path

import numpy as np
import pytest
import os
import pandas as pd
from src.compores.preprocessing import Preprocessor
from src.compores.exceptions_module import NonNumericDataFrameError, EmptyDataFrame, MinDataFrame


class TestPrepareResponse:

    @pytest.fixture(scope="function")
    def logger_mock(self):
        logger = logging.getLogger(__name__)

        return logger

    @pytest.fixture(scope="function")
    def setup_teardown_mock_microbiome_file(self, tmp_path):
        # Put a test microbiome input file in the temporary test directory
        input_microbiome_file_path = tmp_path / "test_microbiome.tsv"
        input_microbiome_data = {'SampleID': ['C10.d4', 'C11.d4', 'C7.d4', 'C8.d4', 'N10.d4'],
                                 'f_A_1683': [0, 0, 0, 1 / 8, 2 / 7],
                                 'f_B_1707': [0, 1 / 3, 2 / 5, 3 / 8, 0],
                                 'f_C_99': [1, 2 / 3, 3 / 5, 4 / 8, 5 / 7]
                                 }
        input_microbiome_df = pd.DataFrame(input_microbiome_data)
        input_microbiome_df.to_csv(input_microbiome_file_path, sep="\t", index=False)

        yield input_microbiome_file_path

    @pytest.fixture
    def setup_teardown_mock_response_file(self, tmp_path):
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
    def setup_teardown_remove_zero_cols(self, tmp_path):
        # Put a test response input file in the temporary test directory
        input_response_data = {
            'SampleID': ['C10.d4', 'C11.d4', 'C7.d4', 'C8.d4', 'N10.d4'],
            'GO:0045071_enh': [1.0, 0, 1.005, 1.021461975, 1.0],
            'GO:0032823_enh': [1.006384536, 0, 3, 3, 3.006384531],
            'GO:2000342_enh': [0, 0, 0, 0, 0],
            'GO:0045887_enh': [2, 2, 2.011461975, 0, 3.005156721],
        }
        input_response_df = pd.DataFrame(input_response_data)
        input_response_file_path = tmp_path / "test_response.tsv"
        input_response_df.to_csv(input_response_file_path, sep="\t", index=False)

        # Define the expected output file path
        expected_output_response_file_path = os.path.join("preprocessing_results", "response", "test-response.tsv")

        # Defined the expected response output dataframe
        expected_response_data = {
            'SampleID': ['C10.d4', 'C11.d4', 'C7.d4', 'C8.d4', 'N10.d4'],
            'GO:0045071_enh': [1.0, 0, 1.005, 1.021461975, 1.0],
            'GO:0032823_enh': [1.006384536, 0, 3, 3, 3.006384531],
            'GO:0045887_enh': [2, 2, 2.011461975, 0, 3.005156721],
        }
        expected_output_response_df = pd.DataFrame(expected_response_data).set_index('SampleID')

        # Yield test_response_df_path and control to the test class
        yield input_response_file_path, expected_output_response_file_path, expected_output_response_df

        # Cleanup: Remove the to be created output directory and its content
        if os.path.exists("preprocessing_results"):
            shutil.rmtree("preprocessing_results")

    @pytest.fixture(scope="function")
    def setup_teardown_catch_non_numeric_values(self, tmp_path):
        # Put a test response input file in the temporary test directory
        input_response_data = {
            'SampleID': ['C10.d4', 'C11.d4', 'C7.d4', 'C8.d4', 'N10.d4'],
            'GO:0045071_enh': [1.0, 1.0, 1.1230, 2.032, 2.032],
            'GO:0032823_enh': [0.006384536, None, 0, 0, 0],  # Introduce a non-numeric value
            'GO:2000342_enh': [1.2354, 1.23, 1.23, 0, 0.008594536],
            'GO:0045887_enh': [1.230, 0, 0.011461975, 0, 0.005156721],
            'GO:0061326_enh': [1.20, "non-numeric", .0, .0, .0]
        }
        input_response_df = pd.DataFrame(input_response_data)
        input_response_file_path = tmp_path / "test_response.tsv"
        input_response_df.to_csv(input_response_file_path, sep="\t", index=False)

        # Define the expected exception message
        expected_exception_message = "The input data contains non-numeric values."

        # Yield test_response_df_path and control to the test class
        yield input_response_file_path, expected_exception_message

    @pytest.fixture(scope="function")
    def setup_teardown_catch_only_zeros_df(self, tmp_path):
        # Put a test response input file in the temporary test directory
        input_response_data = {
            'SampleID': ['C10.d4', 'C11.d4', 'C7.d4', 'C8.d4', 'N10.d4'],
            'GO:0045071_enh': [.0, .0, .0, .0, .0],
            'GO:0032823_enh': [0, 0, 0, 0, 0],
            'GO:2000342_enh': [0, 0, 0, 0, 0],
            'GO:0045887_enh': [0, 0, 0, 0, 0.],
            'GO:0061326_enh': [0, 0, .0, .0, .0]
        }
        input_response_df = pd.DataFrame(input_response_data)
        input_response_file_path = tmp_path / "test_response.tsv"
        input_response_df.to_csv(input_response_file_path, sep="\t", index=False)

        # Define the expected exception message
        expected_exception_message = "The resulting DataFrame is empty."

        # Yield test_response_df_path and control to the test class
        yield input_response_file_path, expected_exception_message

    @pytest.fixture(scope="function")
    def setup_teardown_check_df_min_size_validity(self, tmp_path):
        # Put a test response input file in the temporary test directory
        input_response_data = {
            'SampleID': ['C10.d4', 'C11.d4'],
            'GO:0061326_enh': [.1, .2],
        }
        input_response_df = pd.DataFrame(input_response_data)
        input_response_file_path = tmp_path / "test_response.tsv"
        input_response_df.to_csv(input_response_file_path, sep="\t", index=False)

        # Define the expected exception message
        expected_exception_message = 'The resulting DataFrame should have at least 3 rows and 3 columns for OTU, ' \
                                     'and at least 3 rows and 1 column for response.'

        # Yield test_microbiome_df_path and control to the test class
        yield input_response_file_path, expected_exception_message

    def test_prepare_input_invalid_file_type(
            self, setup_teardown_mock_microbiome_file, setup_teardown_mock_response_file, logger_mock
    ):
        logger = logger_mock
        input_microbiome_file_path = setup_teardown_mock_microbiome_file
        input_response_file_path = setup_teardown_mock_response_file
        with pytest.raises(ValueError) as exc_info:
            preprocess = Preprocessor(
                logger, '', '', '', input_microbiome_file_path, input_response_file_path,
                '', '', '', '', '', '', '', '', 1
            )
            preprocess.prepare_input("invalid_file_type", input_response_file_path)

        assert str(exc_info.value) == "File type `invalid_file_type` is not valid." \
                                      " Please use only 'microbiome' or 'response' values."

    def test_prepare_response_catch_non_existing_file(self, setup_teardown_mock_microbiome_file, logger_mock, caplog):
        logger = logger_mock
        input_microbiome_file_path = setup_teardown_mock_microbiome_file
        with pytest.raises(SystemExit) as exc_info:
            preprocess = Preprocessor(
                logger, '', '', '', input_microbiome_file_path, Path("nonexistent_path"),
                '', '', '', '', '', '', '', '', 1
            )
            preprocess.prepare_input("response", Path("nonexistent_path"))
        assert exc_info.type == SystemExit
        assert exc_info.value.code == 1
        assert "File `nonexistent_path` not found." in caplog.text

    def test_prepare_response_catch_non_numeric_values(
            self, setup_teardown_mock_microbiome_file, setup_teardown_catch_non_numeric_values, logger_mock
    ):
        logger = logger_mock
        input_microbiome_file_path = setup_teardown_mock_microbiome_file
        input_response_file_path, expected_message = setup_teardown_catch_non_numeric_values

        with pytest.raises(NonNumericDataFrameError) as exc_info:
            preprocess = Preprocessor(
                logger, '', '', '', input_microbiome_file_path, input_response_file_path,
                '', '', '', '', '', '', '', '', 1
            )
            preprocess.prepare_input("response", input_response_file_path)
        assert str(exc_info.value) == expected_message

    def test_prepare_response_catch_only_zeros_df(
            self, setup_teardown_mock_microbiome_file, setup_teardown_catch_only_zeros_df, logger_mock
    ):
        logger = logger_mock
        input_microbiome_file_path = setup_teardown_mock_microbiome_file
        input_response_file_path, expected_message = setup_teardown_catch_only_zeros_df

        with pytest.raises(EmptyDataFrame) as exc_info:
            preprocess = Preprocessor(
                logger, '', '', '', input_microbiome_file_path, input_response_file_path,
                '', '', '', '', '', '', '', '', 1
            )
            preprocess.prepare_input("response", input_response_file_path)
        assert str(exc_info.value) == expected_message

    def test_prepare_otu_check_df_size_validity(
            self, setup_teardown_mock_microbiome_file, setup_teardown_check_df_min_size_validity, logger_mock
    ):
        logger = logger_mock
        input_microbiome_file_path = setup_teardown_mock_microbiome_file
        input_response_file_path, expected_message = setup_teardown_check_df_min_size_validity

        with pytest.raises(MinDataFrame) as exc_info:
            preprocess = Preprocessor(
                logger, '', '', '', input_microbiome_file_path, input_response_file_path,
                '', '', '', '', '', '', '', '', 1
            )
            preprocess.prepare_input("response", input_response_file_path)
        assert expected_message in str(exc_info.value)

    def test_prepare_otu_remove_zero_cols(
            self, setup_teardown_mock_microbiome_file, setup_teardown_remove_zero_cols, logger_mock
    ):
        logger = logger_mock
        input_microbiome_file_path = setup_teardown_mock_microbiome_file
        (
            input_response_file_path,
            expected_output_file_path,
            expected_output_df
        ) = setup_teardown_remove_zero_cols

        # Call the function on the sample file
        preprocess = Preprocessor(
            logger, '', '', '', input_microbiome_file_path, input_response_file_path, '', expected_output_file_path,
            '', '', '', '', '', '', 1
        )
        # Call the function on the sample file
        preprocess.prepare_input("response", input_response_file_path)

        # Check if the output file has been created
        assert os.path.exists(expected_output_file_path)
        # Check if the actual output file has the expected content
        assert np.allclose(
            pd.read_csv(expected_output_file_path, sep="\t").set_index('SampleID'),
            expected_output_df,
            rtol=1e-04
        )


