import logging
import shutil
import pytest
import os
import pandas as pd
from src.compores.preprocessing import Preprocessor
from src.compores.exceptions_module import DuplicatedIndices

class TestDuplicatedInputFilesIndices:

    @pytest.fixture(scope="function")
    def logger(self):
        logger = logging.getLogger(__name__)

        return logger

    @pytest.fixture(scope="function")
    def setup_teardown_input_file_microbiome(self, tmp_path):
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

        # Put a test microbiome input file in the temporary test directory
        input_microbiome_data = {
            'SampleID': ['C10.d4', 'C11.d4', 'C11.d4', 'C8.d4', 'N10.d4'],
            'f_A_1683': [.0, 0, .005, 0.021461975, .0],
            'f_B_1707': [0.006384536, 0, 0, 0, 0.006384531],
            'f_C_1645': [0.006583726, 0, 0, 0, 0.008594536],
            'f_C_181': [0, 0, 0.011461975, 0, 0.005156721],
        }
        input_microbiome_df = pd.DataFrame(input_microbiome_data)
        expected_file_name = "test_microbiome.tsv"
        input_microbiome_file_path = tmp_path / expected_file_name
        input_microbiome_df.to_csv(input_microbiome_file_path, sep="\t", index=False)

        expected_exception_message = "Duplicated sample tag names: provide unique sample identifiers"

        yield input_microbiome_file_path, input_response_file_path, expected_exception_message, expected_file_name

        if os.path.exists("preprocessing_results"):
            shutil.rmtree("preprocessing_results")

    @pytest.fixture(scope="function")
    def setup_teardown_input_file_response(self, tmp_path):
        # Put a test response input file in the temporary test directory
        expected_file_name = "test_response.tsv"
        input_response_file_path = tmp_path / expected_file_name
        input_response_data = {
            'SampleID': ['C10.d4', 'C11.d4', 'C11.d4', 'C8.d4', 'N10.d4'],
            'GO:0045071_enh': [1.0, 0, 1.005, 1.021461975, 1.0],
            'GO:0032823_enh': [1.006384536, 0, 3, 3, 3.006384531],
            'GO:0045887_enh': [2, 2, 2.011461975, 0, 3.005156721],
        }
        input_response_df = pd.DataFrame(input_response_data).set_index('SampleID')
        input_response_df.to_csv(input_response_file_path, sep="\t", index=False)

        # Put a test microbiome input file in the temporary test directory
        input_microbiome_data = {
            'SampleID': ['C10.d4', 'C11.d4', 'C7.d4', 'C8.d4', 'N10.d4'],
            'f_A_1683': [.0, 0, .005, 0.021461975, .0],
            'f_B_1707': [0.006384536, 0, 0, 0, 0.006384531],
            'f_C_1645': [0.006583726, 0, 0, 0, 0.008594536],
            'f_C_181': [0, 0, 0.011461975, 0, 0.005156721],
        }
        input_microbiome_df = pd.DataFrame(input_microbiome_data)
        input_microbiome_file_path = tmp_path / "test_microbiome.tsv"
        input_microbiome_df.to_csv(input_microbiome_file_path, sep="\t", index=False)

        expected_exception_message = "Duplicated sample tag names: provide unique sample identifiers"

        yield input_microbiome_file_path, input_response_file_path, expected_exception_message, expected_file_name

        if os.path.exists("preprocessing_results"):
            shutil.rmtree("preprocessing_results")

    def test_check_input_files_for_duplicated_indices_microbiome_ex(self, setup_teardown_input_file_microbiome, logger):
        input_m_file_path, input_r_file_path, expected_message, expected_file = setup_teardown_input_file_microbiome
        with pytest.raises(DuplicatedIndices) as exc_info:
            preprocess = Preprocessor(
                logger, '', '', '', input_m_file_path, input_r_file_path,
                '', '', '', '', '', '', '', '', ['CLR'], 1000, 1
            )
            preprocess.check_input_files_for_duplicated_indices(input_m_file_path, input_r_file_path)

        assert exc_info.type is DuplicatedIndices
        assert expected_message in str(exc_info.value)
        assert expected_file in str(exc_info.value)

    def test_check_input_files_for_duplicated_indices_response_ex(self, setup_teardown_input_file_response, logger):
        input_m_file_path, input_r_file_path, expected_message, expected_file = setup_teardown_input_file_response
        with pytest.raises(DuplicatedIndices) as exc_info:
            preprocess = Preprocessor(
                logger, '', '', '', input_m_file_path, input_r_file_path,
                '', '', '', '', '', '', '', '', ['CLR'], 1000, 1
            )
            preprocess.check_input_files_for_duplicated_indices(input_m_file_path, input_r_file_path)

        assert exc_info.type is DuplicatedIndices
        assert expected_message in str(exc_info.value)
        assert expected_file in str(exc_info.value)

    def test_check_input_files_for_duplicated_indices_microbiome_t(self, setup_teardown_input_file_microbiome, logger):
        input_m_file_path, input_r_file_path, expected_message, expected_file = setup_teardown_input_file_microbiome
        preprocess = Preprocessor(
            logger, '', '', '', input_m_file_path, input_r_file_path,
            '', '', '', '', '', '', '', '', ['CLR'], 1000, 1, deduplicate_flag=True
        )

        preprocess.check_input_files_for_duplicated_indices(input_m_file_path, input_r_file_path)

        m_file = pd.read_csv(input_m_file_path, sep="\t")

        assert m_file.index.duplicated().shape[0] == 5, "The resulting file should have 5 sample tags kept."

    def test_check_input_files_for_duplicated_indices_response_t(self, setup_teardown_input_file_microbiome, logger):
        input_m_file_path, input_r_file_path, expected_message, expected_file = setup_teardown_input_file_microbiome
        preprocess = Preprocessor(
            logger, '', '', '', input_m_file_path, input_r_file_path,
            '', '', '', '', '', '', '', '', ['CLR'], 1000, 1, deduplicate_flag=True
        )

        preprocess.check_input_files_for_duplicated_indices(input_m_file_path, input_r_file_path)

        r_file = pd.read_csv(input_r_file_path, sep="\t")

        assert r_file.index.duplicated().shape[0] == 5, "The resulting file should have 5 sample tags kept."