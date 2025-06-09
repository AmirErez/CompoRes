import logging
import os
import shutil

import numpy as np
import pandas as pd
import pytest
import yaml

from src.compores.compores_main import OneCaseCombination, ComporesMain
from src.compores.utils import save_file


class TestComporesMain:
    @pytest.fixture(scope="function")
    def logger_mock(self):
        logger = logging.getLogger(__name__)

        yield logger

        if os.path.exists("artificial_experiment_output"):
            shutil.rmtree("artificial_experiment_output")

    @pytest.fixture(scope="function")
    def setup_teardown_set_paths(self, tmp_path):
        cfg_file = {
            "PATH_TO_MICROBIOME": os.path.join("path", "to", "microbiome"),
            "PATH_TO_RESPONSE": os.path.join("path", "to", "response"),
            "PATH_TO_METADATA": os.path.join("path", "to", "metadata"),
            "PATH_TO_OUTPUTS": os.path.join("path", "to", "outputs"),
            "OCU_SAMPLING_RATE": 10,
            "CODA_METHOD": None,
            "CORR": 'pearson',
            "SHUFFLE": 'response',
            "N_SHUFFLES": 10,
            "SHUFFLE_CYCLES": 3,
            "N_WORKERS": 4,
        }
        yield cfg_file, "suffix1", "suffix2", "suffix3"

    @pytest.fixture(scope="function")
    def setup_teardown_set_config(self, tmp_path):
        cfg_file = {
            "GROUP1": "50otu",
            "GROUP2": "200samples",
            "GROUP3": "synthetic",
            "PATH_TO_MICROBIOME": os.path.join(
                os.path.dirname(__file__), "data", "artificial_experiment_data", "microbiome"
            ),
            "PATH_TO_RESPONSE": os.path.join(
                os.path.dirname(__file__), "data", "artificial_experiment_data", "response"
            ),
            "PATH_TO_METADATA": os.path.join(
                os.path.dirname(__file__), "data", "artificial_experiment_data", "metadata"
            ),
            "PATH_TO_OUTPUTS": os.path.join(tmp_path, "artificial_experiment_output"),
            'OCU_SAMPLING_RATE': 1,
            "CODA_METHOD": "pairs",
            "CORR": "pearson",
            "SHUFFLE": "microbiome",
            "N_SHUFFLES": 10,
            "SHUFFLE_CYCLES": 3,
            "N_WORKERS": None,
            "METAFILE": os.path.join(
                os.path.dirname(__file__), "data", "artificial_experiment_data", "metafile.tsv"
            ),
        }

        temp_path = os.path.join(tmp_path, "config.yml")
        with open(temp_path, 'w') as file:
            yaml.dump(cfg_file, file, default_flow_style=False)

        yield temp_path


    @pytest.fixture(scope="function")
    def setup_teardown_fetch_full_target_response_label(self, tmp_path):
        path = tmp_path / 'path/to/output/s1-s2-s3/pairs/'
        partial_tag = 'response_2', 'response_3'
        expected_tag = 'response_2_full'

        response_tags = pd.Series(['response_1_full', 'response_2_full'])
        os.makedirs(path, exist_ok=True)
        response_tags.to_pickle(os.path.join(path, 'response_index.pkl'))

        yield path, partial_tag, expected_tag

    @pytest.fixture(scope="function")
    def setup_teardown_fetch_full_target_response_label_exit(self, tmp_path):
        path = tmp_path / 'path/to/output/s1-s2-s3/pairs/'
        partial_tag = 'response_3'

        response_tags = pd.Series(['response_1_full', 'response_2_full'])
        os.makedirs(path, exist_ok=True)
        response_tags.to_pickle(os.path.join(path, 'response_index.pkl'))

        yield path, partial_tag


    def test_set_paths(self, logger_mock, setup_teardown_set_paths):
        # Given parameters for set_paths
        logger = logger_mock
        cfg_file, s1, s2, s3 = setup_teardown_set_paths

        tested_class = OneCaseCombination(logger, cfg_file, True, s1, s2, s3)
        # When calling set_paths
        tested_class.set_paths()

        # Expected paths based on the sample parameters
        expected_fastspar_path = os.path.join("path", "to", "outputs", "preprocessing_results", "fastspar")
        expected_result = (
            os.path.join("path", "to", "microbiome", "suffix1-suffix2-suffix3.tsv"),
            os.path.join("path", "to", "response", "suffix1-suffix2.tsv"),
            os.path.join("path", "to", "metadata", "suffix1-suffix2-metadata.tsv"),
            expected_fastspar_path,
            os.path.join(expected_fastspar_path, "taxa_correlation_suffix1-suffix2-suffix3.tsv"),
            os.path.join(expected_fastspar_path, "taxa_covariance_suffix1-suffix2-suffix3.tsv"),
            os.path.join("path", "to", "outputs", "preprocessing_results", "microbiome", "suffix1-suffix2-suffix3.tsv"),
            os.path.join("path", "to", "outputs", "preprocessing_results", "response", "suffix1-suffix2.tsv"),
        )

        # Then compare the actual result with the expected result
        assert (
                   tested_class.path_to_microbiome, tested_class.path_to_response, tested_class.meta_data,
                   tested_class.path_to_fastspar_res, tested_class.path_to_fastspar_corr,
                   tested_class.path_to_fastspar_cov, tested_class.path_to_microbiome_clustering,
                   tested_class.path_to_prepared_response
               ) == expected_result

    def test_set_paths_output_path_key_error(self, tmp_path):
        # Given parameters for set_paths
        cfg_file = {
            "PATH_TO_MICROBIOME": os.path.join("path", "to", "microbiome"),
            "PATH_TO_RESPONSE": os.path.join("path", "to", "response"),
            "PATH_TO_METADATA": os.path.join("path", "to", "metadata"),
            "CODA_METHOD": 'pairs',
            "CORR": 'pearson',
        }
        with open(tmp_path / 'config.yml', 'w') as file:
            yaml.dump(cfg_file, file, default_flow_style=False)

        with pytest.raises(KeyError) as exc_info:
            ComporesMain(str(tmp_path / 'config.yml'))

        assert "Check the `PATH_TO_OUTPUTS` parameter value in the config file" in str(exc_info.value)

    def test_set_paths_output_path_type_error(self, tmp_path):
        # Given parameters for set_paths
        cfg_file = {
            "PATH_TO_MICROBIOME": os.path.join("path", "to", "microbiome"),
            "PATH_TO_RESPONSE": os.path.join("path", "to", "response"),
            "PATH_TO_METADATA": os.path.join("path", "to", "metadata"),
            "PATH_TO_OUTPUTS": None,
            "CODA_METHOD": 'pairs',
            "CORR": 'pearson',
        }
        with open(tmp_path / 'config.yml', 'w') as file:
            yaml.dump(cfg_file, file, default_flow_style=False)

        with pytest.raises(TypeError) as exc_info:
            ComporesMain(str(tmp_path / 'config.yml'))

        assert "Check the `PATH_TO_OUTPUTS` parameter value in the config file" in str(exc_info.value)

    def test_close(self, logger_mock, setup_teardown_set_config):
        logger = logger_mock
        temp_path = setup_teardown_set_config
        tested_class = ComporesMain(temp_path)
        tested_class.logger = logger
        mock_handler1 = logging.StreamHandler()
        mock_handler2 = logging.FileHandler("test.log")
        tested_class.logger.handlers = [mock_handler1, mock_handler2]
        tested_class.close()
        assert len(tested_class.logger.handlers) == 0

        # clean up
        file = os.path.join(os.path.dirname(__file__), "test.log")
        if os.path.exists(file):
            os.remove(file)

    def test_set_n_workers_from_none(self, logger_mock, setup_teardown_set_config):
        logger = logger_mock
        temp_path = setup_teardown_set_config

        tested_class = ComporesMain(temp_path)
        tested_class.logger = logger
        tested_class.set_n_workers()

        # Expected paths based on the sample parameters
        cpu_count = tested_class._get_total_cpu_count()
        expected_n_workers = max(2, (cpu_count // 2 // 4) * 4) if cpu_count > 2 else 1

        # Clean up
        tested_class.logger_instance.cleanup_logger_file_handlers()

        # Then compare the actual result with the expected result
        assert tested_class.n_workers == expected_n_workers

    def test_mean_log_score_over_otu_clustering(self):
        input_dict = {
            "50otu_200samples_synthetic": {
                3: np.array([0.05, 0.15, 0.25]),
                4: np.array([0.15, 0.25, 0.35])
            }
        }

        expected_dict = {
            "50otu_200samples_synthetic": np.array([2.44642613, 1.64170717, 1.21805824])
        }

        actual_dict = OneCaseCombination.mean_log_score_over_otu_clustering(input_dict)

        are_equal_with_tolerance = np.allclose(
            actual_dict["50otu_200samples_synthetic"], expected_dict["50otu_200samples_synthetic"], rtol=1e-5, atol=1e-8
        )

        assert are_equal_with_tolerance

    def test_fetch_full_target_response_label(
            self, setup_teardown_fetch_full_target_response_label, setup_teardown_set_config
    ):
        config_path = setup_teardown_set_config
        response_index_path, partial_tag, expected_tag = setup_teardown_fetch_full_target_response_label
        compores_main = ComporesMain(config_path)
        result = compores_main.fetch_full_target_response_label(response_index_path, partial_tag)
        assert result == expected_tag

    def test_fetch_full_target_response_label_exit(
            self, setup_teardown_fetch_full_target_response_label_exit, setup_teardown_set_config
    ):
        config_path = setup_teardown_set_config
        response_index_path, partial_tag = setup_teardown_fetch_full_target_response_label_exit
        compores_main = ComporesMain(config_path)

        # Catch the exit call
        with pytest.raises(SystemExit) as exc_info:
            result = compores_main.fetch_full_target_response_label(response_index_path, partial_tag)
            assert result is None
        assert "1" in str(exc_info.value)


    def test_fetch_first_response_tag(self, monkeypatch, tmp_path):

        path_to_ranking = tmp_path / "ranking"
        n_shuffle_cycles = 5

        def mock_listdir(path):
            return [f"{n_shuffle_cycles}.csv"]

        def mock_read_csv(filepath_or_buffer):
            return pd.DataFrame(
                [2.2, 2.0, 1.8], index = ["response_2_e", "response_1_b", "response_3_x"]
            ).reset_index()

        monkeypatch.setattr(os, "listdir", mock_listdir)
        monkeypatch.setattr(pd, "read_csv", mock_read_csv)

        response_tag = ComporesMain.fetch_first_response_tag(path_to_ranking, n_shuffle_cycles)
        assert response_tag == "response_2_e"

    def test_extract_response_tags(self, logger_mock, setup_teardown_set_paths, monkeypatch, tmp_path):

        logger = logger_mock
        cfg_file, s1, s2, s3 = setup_teardown_set_paths

        def mock_read_parquet(filepath_or_buffer):
            return pd.DataFrame({
                'feature1': [1, 2, 3],
                'feature2': [4, 5, 6],
                'feature3\nwith_newline': [7, 8, 9]
            })

        monkeypatch.setattr(pd, "read_parquet", mock_read_parquet)
        expected_response_tags = ['response_1_feature1', 'response_2_feature2', 'response_3_feature3']

        tested_class = OneCaseCombination(logger, cfg_file, True, s1, s2, s3)
        tested_class.path_to_prepared_response = tmp_path / "response.parquet"
        tested_class.extract_response_tags()

        assert tested_class.response_index == expected_response_tags

    def test_combine_batch_dictionaries_to_ocu(self, logger_mock, setup_teardown_set_paths, tmp_path):

        logger = logger_mock
        cfg_file, s1, s2, s3 = setup_teardown_set_paths

        mock_batch_structure = [[1, 2, 3], [4, 5, 6]]
        mock_batch_dictionaries = {
            "1-3": {"V-I-P": {22: {3: 0.7, 2: 0.6, 1: 0.5}, 37: {1: 0.8, 2: 0.9, 3: 1.0}}},
            "4-6": {"V-I-P": {22: {4: 0.8, 5: 0.9, 6: 1.0}, 37: {5: 1.2, 4: 1.1, 6: 1.3}}},
        }
        # Save the mock dictionaries according to the mock batch structure in the temp directory
        for key, value in mock_batch_dictionaries.items():
            save_file(mock_batch_dictionaries[key], "mock_dict_name.pkl", tmp_path / key)

        expected_combined_dictionary = {
            "V-I-P": {
                22: {1: 0.5, 2: 0.6, 3: 0.7, 4: 0.8, 5: 0.9, 6: 1.0},
                37: {1: 0.8, 2: 0.9, 3: 1.0, 4: 1.1, 5: 1.2, 6: 1.3}
            }
        }

        tested_class = OneCaseCombination(logger, cfg_file, True, s1, s2, s3)
        tested_class.intermediate_results_path = tmp_path

        tested_class.combine_batch_dictionaries_to_ocu("mock_dict_name", mock_batch_structure)

        # Check if the combined dictionary is saved correctly
        combined_dict_path = tmp_path / "mock_dict_name.pkl"
        assert os.path.exists(combined_dict_path)

        # Read the combined dictionary and verify its structure
        saved_dict = pd.read_pickle(combined_dict_path)
        assert "V-I-P" in saved_dict
        assert 22 in saved_dict["V-I-P"]
        assert 37 in saved_dict["V-I-P"]
        assert [1, 2, 3, 4, 5, 6] == list(saved_dict["V-I-P"][22].keys())
        assert [1, 2, 3, 4, 5, 6] == list(saved_dict["V-I-P"][37].keys())
        assert saved_dict["V-I-P"][22] == expected_combined_dictionary["V-I-P"][22]
        assert saved_dict["V-I-P"][37] == expected_combined_dictionary["V-I-P"][37]

    def test_combine_batch_dictionaries_to_ocu_nested(self, logger_mock, setup_teardown_set_paths, tmp_path):

        logger = logger_mock
        cfg_file, s1, s2, s3 = setup_teardown_set_paths

        mock_batch_structure = [[1, 2], [3]]
        mock_batch_dictionaries = {
            "1-2": {"V-I-P": {
                22: {2: {"s": 0.1, "p": 0.2}, 1: {"s": 0.3, "p": 0.4}},
                37: {1: {"s": 0.1, "p": 0.2}, 2: {"s": 0.3, "p": 0.9}}}},
            "3-3": {"V-I-P": {22: {3: {"s": 0.5, "p": 0.05}}, 37: {3: {"s": 0.3, "p": 0.9}}}},
        }
        # Save the mock dictionaries according to the mock batch structure in the temp directory
        for key, value in mock_batch_dictionaries.items():
            save_file(mock_batch_dictionaries[key], "mock_dict_name.pkl", tmp_path / key)

        expected_combined_dictionary = {
            "V-I-P": {
                22: {1: {"s": 0.3, "p": 0.4}, 2: {"s": 0.1, "p": 0.2}, 3: {"s": 0.5, "p": 0.05}},
                37: {1: {"s": 0.1, "p": 0.2}, 2: {"s": 0.3, "p": 0.9}, 3: {"s": 0.3, "p": 0.9}}
            }
        }

        tested_class = OneCaseCombination(logger, cfg_file, True, s1, s2, s3)
        tested_class.intermediate_results_path = tmp_path

        tested_class.combine_batch_dictionaries_to_ocu("mock_dict_name", mock_batch_structure)

        # Check if the combined dictionary is saved correctly
        combined_dict_path = tmp_path / "mock_dict_name.pkl"
        assert os.path.exists(combined_dict_path)

        # Read the combined dictionary and verify its structure
        saved_dict = pd.read_pickle(combined_dict_path)
        assert "V-I-P" in saved_dict
        assert 22 in saved_dict["V-I-P"]
        assert 37 in saved_dict["V-I-P"]
        assert [1, 2, 3] == list(saved_dict["V-I-P"][22].keys())
        assert [1, 2, 3] == list(saved_dict["V-I-P"][37].keys())
        assert saved_dict["V-I-P"][22] == expected_combined_dictionary["V-I-P"][22]
        assert saved_dict["V-I-P"][37] == expected_combined_dictionary["V-I-P"][37]

