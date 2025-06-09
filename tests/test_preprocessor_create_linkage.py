import logging

import numpy as np
import pandas as pd
import pytest

from src.compores.preprocessing import Preprocessor


class TestCreateLinkage:
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
    def setup_teardown_mock_corr_file(self, tmp_path):
        # Mocking a correlation matrix for testing
        mock_corr_data = {
            'otu_1': [1.0, 0.8, 0.3, 0.5, 0.5, 0.3],
            'otu_2': [0.8, 1.0, 0.5, 0.3, 0.3, 0.5],
            'otu_3': [0.3, 0.5, 1.0, 0.8, 0.8, 0.8],
            'otu_4': [0.5, 0.3, 0.8, 1.0, 0.8, 0.8],
            'otu_5': [0.5, 0.3, 0.8, 0.8, 1.0, 0.8],
            'otu_6': [0.3, 0.5, 0.8, 0.8, 0.8, 1.0]
        }
        mock_corr_df = pd.DataFrame(mock_corr_data, index=['otu_1', 'otu_2', 'otu_3', 'otu_4', 'otu_5', 'otu_6'])
        expected_result = {
            'expected_linkage_matrix': np.array(
                [
                    [0.,  1.,  0.3, 2.],
                    [2., 3.,  0.3, 2.],
                    [4.,  7.,  0.3, 3.],
                    [5.,  8.,  0.3, 4.],
                    [6.,  9.,  0.45499115, 6.]
                ]
            )
        }

        # Save the mock correlation matrix to a temporary file
        mock_corr_file_path = tmp_path / 'mock_corr_file.csv'
        mock_corr_df.to_csv(mock_corr_file_path, sep='\t')
        yield mock_corr_file_path, expected_result

    def test_create_linkage(
            self, tmp_path,
            setup_teardown_mock_microbiome_file, setup_teardown_mock_response_file, setup_teardown_mock_corr_file,
            logger_mock
    ):

        # Test the create_linkage function
        fastspar_corr_file, expected_result = setup_teardown_mock_corr_file
        preprocessor = Preprocessor(
            logger_mock, '', '', '', setup_teardown_mock_microbiome_file, setup_teardown_mock_response_file, '', '', '',
            fastspar_corr_file, '', '', '','', 1
        )
        preprocessor.path_to_fastspar_corr = fastspar_corr_file
        preprocessor.create_linkage()
        actual_linkage_matrix = preprocessor.linkage_matrix

        assert isinstance(expected_result['expected_linkage_matrix'], np.ndarray)
        assert np.allclose(actual_linkage_matrix, expected_result['expected_linkage_matrix'])

