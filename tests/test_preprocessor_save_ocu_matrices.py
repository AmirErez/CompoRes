import logging
import os
from pathlib import Path

import pytest

import numpy as np
import pandas as pd

from scipy.cluster.hierarchy import linkage
from src.compores.preprocessing import Preprocessor


class TestSaveOcuMatrices:
    @pytest.fixture(scope="function")
    def logger_mock(self):
        logger = logging.getLogger(__name__)

        return logger

    @pytest.fixture(scope="function")
    def setup_teardown_mock_params(self, tmp_path):

        input_microbiome_data = {
            'SampleID': ['C10.d4', 'C11.d4', 'C7.d4', 'C8.d4', 'N10.d4'],
            'f_A_1683': [.0, .0, .005, .021461975, .0],
            'f_B_1707': [.006384536, .0, .0, .0, .006384531],
            'f_C_1645': [.006583726, .0, .0, .0, .008594536],
            'f_C_181': [.0, .0, .011461975, .0, .005156721],
        }
        input_microbiome_df = pd.DataFrame(input_microbiome_data)
        input_microbiome_df.set_index('SampleID', inplace=True)

        # Create a mock microbiome input file
        input_microbiome_file_path = tmp_path / "test_microbiome.tsv"
        input_microbiome_df.to_csv(input_microbiome_file_path, sep="\t")

        # Save a mock 'preprocessed' microbiome file
        preprocessed_microbiome_path = tmp_path / 'preprocessing_results/microbiome'
        os.makedirs(preprocessed_microbiome_path, exist_ok=True)
        preprocessed_microbiome_file_path = tmp_path / 'preprocessing_results/microbiome/test_microbiome.tsv'
        input_microbiome_df.to_csv(preprocessed_microbiome_file_path, sep="\t")

        # Create a mock response input file
        input_response_file_path = tmp_path / "test_response.tsv"
        input_response_data = {
            'SampleID': ['C10.d4', 'C11.d4', 'C7.d4', 'C8.d4', 'N10.d4'],
            'GO:0045071_enh': [1.0, 0, 1.005, 1.021461975, 1.0],
            'GO:0032823_enh': [1.006384536, 0, 3, 3, 3.006384531],
            'GO:0045887_enh': [2, 2, 2.011461975, 0, 3.005156721],
        }
        input_response_df = pd.DataFrame(input_response_data).set_index('SampleID')
        input_response_df.to_csv(input_response_file_path, sep="\t", index=False)

        # Create a mock temporary linkage matrix
        linkage_array = linkage(input_microbiome_df.T.values, method='ward')

        yield linkage_array

    @pytest.fixture(scope="function")
    def setup_teardown_save_ocu_matrices(self, tmp_path):

        # Create one of the expected OCU matrix (for the first threshold) to test
        expected_ocu_matrix = {
            'SampleID': ['C10.d4', 'C11.d4', 'C7.d4', 'C8.d4', 'N10.d4'],
            'ocu_1': [.012968, 0, .0, .0, .014979067],
            'ocu_2': [.0, .0, .011461975, .0, .005156721],
            'ocu_3': [.0, .0, .005, .021461975, .0]
        }
        expected_ocu_matrix_df = pd.DataFrame(expected_ocu_matrix).set_index('SampleID')

        expected_dict_str = """{
    "4 OCUs": {
        "threshold": 0,
        "OCUs": {
            "ocu_1": {
                "taxa": [
                    "f_A_1683"
                ],
                "imputed_in": []
            },
            "ocu_2": {
                "taxa": [
                    "f_B_1707"
                ],
                "imputed_in": []
            },
            "ocu_3": {
                "taxa": [
                    "f_C_1645"
                ],
                "imputed_in": []
            },
            "ocu_4": {
                "taxa": [
                    "f_C_181"
                ],
                "imputed_in": []
            }
        }
    },
    "3 OCUs": {
        "threshold": 0.0022189634418180487,
        "OCUs": {
            "ocu_1": {
                "taxa": [
                    "f_B_1707",
                    "f_C_1645"
                ],
                "imputed_in": []
            },
            "ocu_2": {
                "taxa": [
                    "f_C_181"
                ],
                "imputed_in": []
            },
            "ocu_3": {
                "taxa": [
                    "f_A_1683"
                ],
                "imputed_in": []
            }
        }
    }
}"""
        expected_output_example = {
            'output_dir': tmp_path / 'preprocessing_results/microbiome/OCUs/Van-IP-feces',
            'expected_dict_str': expected_dict_str,
            'expected_ocu_matrix_df': expected_ocu_matrix_df,
            'expected_ocu_matrix_df_columns': ['ocu_1', 'ocu_2', 'ocu_3']
        }
        yield expected_output_example

    def test_save_ocu_matrices(
            self, setup_teardown_mock_params, setup_teardown_save_ocu_matrices, logger_mock, tmp_path
    ):
        logger = logger_mock
        plot_output_path = f"{tmp_path}/CLR"
        preprocessor = Preprocessor(
            logger, 'Van', 'IP', 'feces',
            tmp_path / "test_microbiome.tsv", tmp_path / "test_response.tsv",
            tmp_path / 'preprocessing_results/microbiome/test_microbiome.tsv',
            '', '', '', '',
            tmp_path,
            tmp_path / 'preprocessing_results/microbiome/OCUs',
            plot_output_path, 1
        )

        linkage_mock_input = setup_teardown_mock_params
        preprocessor.linkage_matrix = linkage_mock_input

        expected_output = setup_teardown_save_ocu_matrices

        # Call your function with test data
        preprocessor.save_ocu_matrices()

        output_path = expected_output['output_dir']
        # Check if the output file has been created
        assert os.path.exists(output_path)

        # Check if there are exactly three directories inside "mock_results_path/..."
        sub_dirs = [f for f in os.listdir(output_path) if os.path.isdir(os.path.join(output_path, f))]
        assert len(sub_dirs) == 2

        # Check if the dictionary file has been created
        output_file = os.path.join(output_path, f"{preprocessor.file_name}_ocu_clustering_dictionary.json")
        assert (os.path.exists(output_file))

        content = Path(output_file).read_text()
        assert content == expected_output['expected_dict_str']

        # Check the first OCU matrix
        first_clustering_example = str(max([int(f) for f in sub_dirs]) - 1)
        assert os.path.exists(os.path.join(output_path, first_clustering_example))
        actual_ocu_matrix_example = pd.read_csv(
            os.path.join(
                output_path,
                first_clustering_example,
                f"{preprocessor.file_name}_{first_clustering_example}_OCUs.tsv"
            ), sep="\t"
        ).set_index('SampleID')
        assert np.allclose(actual_ocu_matrix_example.values, expected_output['expected_ocu_matrix_df'], rtol=1e-04)
        assert actual_ocu_matrix_example.columns.tolist() == expected_output['expected_ocu_matrix_df_columns']

        # Validate all OCU matrix files (CSV/TSV outputs)
        for sub_dir in ["3"]:
            ocu_dir = os.path.join(plot_output_path, sub_dir)
            csv_files = [f for f in os.listdir(ocu_dir) if f.endswith('.csv')]
            assert len(csv_files) == 1, f"Expected 1 TSV file in {ocu_dir}, found {len(csv_files)}"


    def test_save_ocu_taxa_map_csv(self, setup_teardown_mock_params, logger_mock, tmp_path):
        """Test to validate the structure and content of OCU CSV files generated in each clustering directory."""
        logger = logger_mock
        plot_output_path = tmp_path / "CLR"
        preprocessor = Preprocessor(
            logger, 'Van', 'IP', 'feces',
            tmp_path / "test_microbiome.tsv", tmp_path / "test_response.tsv",
            tmp_path / 'preprocessing_results/microbiome/test_microbiome.tsv',
            '', '', '', '',
            tmp_path,
            tmp_path / 'preprocessing_results/microbiome/OCUs',
            plot_output_path,
            1
        )

        linkage_mock_input = setup_teardown_mock_params
        preprocessor.linkage_matrix = linkage_mock_input
        preprocessor.save_ocu_matrices()

        for sub_dir in ["3"]:
            ocu_dir = os.path.join(plot_output_path,sub_dir)
            csv_files = [f for f in os.listdir(ocu_dir) if f.endswith('.csv')]

            for csv_file in csv_files:
                df_path = os.path.join(ocu_dir, csv_file)
                assert os.path.exists(df_path), f"{df_path} does not exist"

                df = pd.read_csv(df_path, sep=",")
                assert 'OCU' in df.columns, f"'OCU' column missing in {csv_file}"
                assert df.shape[0] == int(sub_dir), f"Unexpected number of rows in {csv_file}"

                # Ensure all OCU columns are float and no NaNs
                for col in df.columns:
                    if col != 'OCU':
                        assert df[col].apply(lambda x: isinstance(x, str)).all(), \
                            f"Non-string values found in column {col} of {csv_file}"
                        assert not df[col].isna().any(), f"NaNs found in column {col} of {csv_file}"
