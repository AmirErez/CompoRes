import os.path
import pickle
import shutil

import pytest
import yaml

from src.compores.compores_main import ComporesMain

MIN_OCU_NUM = 3


class TestCompoResResultDictionary:

    @pytest.fixture(scope="function")
    def setup_teardown_compores(self, tmp_path):
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
            "PATH_TO_OUTPUTS": os.path.join(os.path.dirname(__file__), "artificial_experiment_output"),
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
        with open(tmp_path / 'config.yaml', 'w') as file:
            yaml.dump(cfg_file, file)

        path_to_config = tmp_path / "config.yaml"

        yield path_to_config

        # cleanup .log files in the test directory
        if os.path.exists("artificial_experiment_output"):
            shutil.rmtree("artificial_experiment_output")

    def test_compores_output(self, setup_teardown_compores):

        input_path = setup_teardown_compores

        expected_exp_name = "50otu-200samples-synthetic"
        expected_dir_name_keys = set(range(3, 51))
        # expected_dir_name_keys.discard(46)

        runner = ComporesMain(input_path)
        runner.run()
        # Read one of the dictionary pickles in the compores_basic_result/exp_name subfolder of the output directory
        with open(os.path.join(
                runner.config_dict["PATH_TO_OUTPUTS"], "compores_basic_results",
                expected_exp_name, "pairs", "p_values.pkl"
        ), "rb") as f:
            p_values_dict = pickle.load(f)

        # Check that the `exp_name` key is in the dictionary
        assert expected_exp_name in p_values_dict
        # Check that the dictionary has keys in accordance to the OTU clustering folder names (3, ..., 50) without 46
        assert set(p_values_dict["50otu-200samples-synthetic"].keys()) == expected_dir_name_keys
        # Check that the chosen dictionary has numpy arrays as values of the length 2 (number of response variables)
        for key, value in p_values_dict["50otu-200samples-synthetic"].items():
            assert isinstance(value, dict), f"Value at key {key} is not a dictionary"
            assert len(value) == 2, f"Dictionary at key {key} is not of length 2"
