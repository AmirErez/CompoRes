import os.path
import pickle
import shutil

import pandas as pd
import pytest
import yaml

from src.compores.compores_main import ComporesMain, SYNTHETIC_ANALYSIS_RESULTS_PATH, P_VALUES_FILE_NAME
from src.compores.preprocessing import MIN_OCU_NUM

TEST_DATA_PATH = "artificial_experiment_data"
TEST_OUTPUT_PATH = "artificial_experiment_output"
TEST_SAMPLING_RATE = 15


class TestCompoResIntegration:

    @pytest.fixture(scope="class", autouse=True)
    def cleanup_test_output(self):
        yield
        if os.path.exists(TEST_OUTPUT_PATH):
            shutil.rmtree(TEST_OUTPUT_PATH)

    @pytest.fixture(scope="function")
    def setup_teardown_compores(self, tmp_path):
        cfg_file = {
            "GROUP1": "50otu",
            "GROUP2": "200samples",
            "GROUP3": "synthetic",
            "PATH_TO_MICROBIOME": os.path.join(os.path.dirname(__file__), "data", TEST_DATA_PATH, "microbiome"),
            "PATH_TO_RESPONSE": os.path.join(os.path.dirname(__file__), "data", TEST_DATA_PATH, "response"),
            "PATH_TO_METADATA": os.path.join(os.path.dirname(__file__), "data", TEST_DATA_PATH, "metadata"),
            "PATH_TO_OUTPUTS": os.path.join(os.path.dirname(__file__), TEST_OUTPUT_PATH),
            'TAXA_FILTER': 0.2,
            'OCU_SAMPLING_RATE': TEST_SAMPLING_RATE,
            'MAX_OCU': 800,
            "CODA_METHOD": "pairs",
            "SPARCCKIT_MAX_ITER": 3,
            "OUTLIERS_REMOVAL": False,
            "CORR": "pearson",
            "SHUFFLE": "microbiome",
            "N_SHUFFLES": 4,
            "SHUFFLE_CYCLES": 3,
            "N_WORKERS": None,
            "METAFILE": os.path.join(os.path.dirname(__file__), "data", TEST_DATA_PATH, "metafile.tsv"),
        }
        with open(tmp_path / 'config.yaml', 'w') as file:
            yaml.dump(cfg_file, file)

        path_to_config = tmp_path / "config.yaml"

        expected_exp_name = "50otu-200samples-synthetic"
        expected_dir_name_keys = list(range(MIN_OCU_NUM, 51))
        expected_dir_name_keys = expected_dir_name_keys[::-1]
        expected_dir_name_keys = expected_dir_name_keys[::TEST_SAMPLING_RATE]

        yield path_to_config, expected_exp_name, expected_dir_name_keys

        if hasattr(self, "runner"):
            self.runner.close()

    def test_compores_output_only_preprocessing_step(self, setup_teardown_compores):

        input_path, expected_exp_name, expected_dir_name_keys = setup_teardown_compores

        self.runner = ComporesMain(input_path)
        self.runner.run(only_preprocess_step_flag=True)

        for coda_method in ["CLR", "pairs"]:
            path_to_plot_dir = os.path.join(
                self.runner.config_dict["PATH_TO_OUTPUTS"],
                "plots",
                "response_vs_best_balance",
                expected_exp_name,
                coda_method,
            )
            for i in expected_dir_name_keys[:-1]:
                assert os.path.exists(os.path.join(path_to_plot_dir, str(i)))
                assert any(
                    "mapping.csv" in filename
                    for filename in os.listdir(os.path.join(path_to_plot_dir, str(i)))
                ), f"mapping.csv not found in {os.path.join(path_to_plot_dir, str(i))}"

        log_file_path = os.path.join(
            self.runner.config_dict["PATH_TO_OUTPUTS"],
            "logs",
            expected_exp_name,
            "compores_main.log",
        )
        with open(log_file_path, "r") as log_file:
            log_contents = log_file.read()
            assert (
                "Preprocessing step only: exiting before running CoDa analysis."
                in log_contents
            )

    def test_compores_output(self, setup_teardown_compores):

        input_path, expected_exp_name, expected_dir_name_keys = setup_teardown_compores

        self.runner = ComporesMain(input_path)
        self.runner.run()
        # Read one of the dictionary pickles in the compores_basic_result/exp_name subfolder of the output directory
        with open(os.path.join(
                self.runner.config_dict["PATH_TO_OUTPUTS"], "compores_basic_results",
                expected_exp_name, "pairs", P_VALUES_FILE_NAME
        ), "rb") as f:
            p_values_dict = pickle.load(f)

        # Check that the `exp_name` key is in the dictionary
        assert expected_exp_name in p_values_dict
        # Check that the dictionary has keys in accordance to the processed OTU clustering folder names
        assert list(p_values_dict["50otu-200samples-synthetic"].keys()) == expected_dir_name_keys
        # Check that the chosen dictionary has numpy arrays as values of length 2 (number of response variables)
        for key, value in p_values_dict["50otu-200samples-synthetic"].items():
            assert isinstance(value, dict), f"Value at key {key} is not a dictionary"
            assert len(value) == 2, f"Dictionary at key {key} is not of length 2"

    @pytest.mark.slow
    def test_include_classification_power_analysis_after_compores(
        self, setup_teardown_compores
    ):
        # NOTE: SPA uses a shell code; the test may fail if no Linux-like shell is available
        input_path, expected_exp_name, expected_dir_name_keys = setup_teardown_compores

        self.runner = ComporesMain(input_path)

        # match the GitHub behavior to running compores from src
        os.chdir(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
        self.runner.run()
        self.runner.add_synthetic_data_analysis(response_label=None)
        # change back to the test directory
        os.chdir(os.path.dirname(__file__))

        # Verify the classification analysis outputs
        classification_output_path = os.path.join(
            self.runner.config_dict["PATH_TO_OUTPUTS"],
            SYNTHETIC_ANALYSIS_RESULTS_PATH,
            "response_1_RSP1",
        )

        assert os.path.exists(classification_output_path)
        assert os.path.exists(
            os.path.join(
                classification_output_path,
                f"{expected_exp_name}_mean_auroc_vs_noise_pairs_balance_with_noise_analysis.png"
            )
        )


    def test_generate_otu_p_value_summary_data(self, setup_teardown_compores):

        input_path, expected_exp_name, expected_dir_name_keys = setup_teardown_compores

        self.runner = ComporesMain(input_path)

        self.runner.run()
        self.runner.generate_otu_p_value_summary_data()

        # Check that every method folder contains the expected number of summary data files
        for method in ["CLR", "pairs"]:
            summary_data_path = os.path.join(
                self.runner.config_dict["PATH_TO_OUTPUTS"],
                "otu_significance_tracing",
                expected_exp_name,
                method
            )
            files = os.listdir(summary_data_path)
            if method == "CLR":
                assert len(files) == 6  # 3 files for the single response variable, 2 tested response variables
            elif method == "pairs":
                assert len(files) == 12  # 6 files per response variable, 2 tested response variables
            for file_name in files:
                df = pd.read_csv(os.path.join(summary_data_path, file_name), sep="\t")
                assert not df.empty, f"DataFrame in {file_name} is empty"
                assert df.shape[0] == 50
