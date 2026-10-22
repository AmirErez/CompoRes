import os
import subprocess
import sys

import pandas as pd
import pytest
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUICK_TEST_CONFIG = os.path.join(REPO_ROOT, "configs", "config_synthetic_data_quick.yaml")
EXPERIMENT_NAME = "50otu-100samples-synthetic"
CORRELATED_RESPONSES = {f"response_{i}_correlated_response_{i}" for i in range(1, 6)}


class TestQuickTestRun:
    """Runs the quick test of the README end to end through the command line, and checks the results it documents."""

    @pytest.fixture(scope="class")
    def quick_test_outputs(self, tmp_path_factory):
        tmp_path = tmp_path_factory.mktemp("quick_test")
        with open(QUICK_TEST_CONFIG, "r") as config_file:
            config = yaml.safe_load(config_file)
        # Keep the bundled input data, but write the outputs to the temporary folder
        for key in ("PATH_TO_MICROBIOME", "PATH_TO_RESPONSE", "PATH_TO_METADATA"):
            config[key] = os.path.normpath(os.path.join(os.path.dirname(QUICK_TEST_CONFIG), config[key]))
        config["PATH_TO_OUTPUTS"] = str(tmp_path / "outputs")
        with open(tmp_path / "config.yaml", "w") as config_file:
            yaml.dump(config, config_file)

        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [os.path.join(REPO_ROOT, "src"), env.get("PYTHONPATH")]))
        result = subprocess.run(
            [sys.executable, "-m", "compores", "--config", str(tmp_path / "config.yaml")],
            cwd=tmp_path, env=env, capture_output=True, text=True
        )
        assert result.returncode == 0, f"The quick test run failed:\n{result.stderr[-3000:]}"
        return tmp_path / "outputs"

    @pytest.mark.parametrize("balance_method", ["CLR", "pairs"])
    def test_correlated_responses_rank_on_top(self, quick_test_outputs, balance_method):
        ranking = pd.read_csv(
            quick_test_outputs / "compores_response_ranking" / EXPERIMENT_NAME / balance_method /
            "minus_mean_log_p_value_sorted_responses_at_shuffling_cycle_2.csv",
            index_col=0
        )
        assert set(ranking.index[:5]) == CORRELATED_RESPONSES

    def test_best_balance_is_built_from_true_otus(self, quick_test_outputs):
        # At 50 OCUs every OCU is a single OTU; the observed best balance does not depend on the random shuffles
        best_balances = pd.read_csv(
            quick_test_outputs / "balance_calculation_results" / EXPERIMENT_NAME / "pairs" / "ocu_level_results" /
            "regular" / "response_5_correlated_response_5_combined_ocu_level_results.csv",
            index_col=0
        )
        assert best_balances.loc["50 OCUs", "NUM_Taxa_List"] == "['OTU_5']"
        assert best_balances.loc["50 OCUs", "DEN_Taxa_List"] == "['OTU_35']"
