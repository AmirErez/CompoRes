import logging
import os

import pytest

import numpy as np
import pandas as pd

from src.compores.preprocessing import Preprocessor
from src.compores.compores_main import OneCaseCombination
from src.compores.utils import extract_response_tags


class TestPreprocessorOutlierRemoval:

    @pytest.fixture(scope="function")
    def logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter("%(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        logger.info("This will be printed to the console")

        yield logger

    @pytest.fixture(scope="function")
    def setup_teardown_set_config(self, tmp_path):

        path_microbiome = tmp_path / "microbiome.csv"
        path_response = tmp_path / "response.csv"

        pd.DataFrame().to_csv(path_microbiome, sep="\t")
        pd.DataFrame().to_csv(path_response, sep="\t")

        cfg_file = {
            "GROUP1": "s1",
            "GROUP2": "s2",
            "GROUP3": "s3",
            "PATH_TO_MICROBIOME": path_microbiome,
            "PATH_TO_RESPONSE": path_response,
            "PATH_TO_METADATA": '',
            "PATH_TO_OUTPUTS": os.path.join(tmp_path, "test_output"),
            # "PATH_TO_OUTPUTS": "test_output",
            "OCU_SAMPLING_RATE": 1,
            "CODA_METHOD": "CLR",
            "MAX_OCU": 800,
            "OUTLIERS_REMOVAL": True,
            "SPARCCKIT_MAX_ITER": 1000,
            "CORR": "pearson",
            "SHUFFLE": "microbiome",
            "N_SHUFFLES": 10,
            "SHUFFLE_CYCLES": 3,
            "N_WORKERS": None,
        }

        yield cfg_file

    @pytest.fixture
    def test_preprocessor(self, logger, tmp_path):
        # Create minimal required paths for Preprocessor initialization
        path_microbiome = tmp_path / "microbiome.csv"
        path_response = tmp_path / "response.csv"
        path_to_outlier_detection = tmp_path / "outputs/preprocessing_results/outlier_detection"
        os.makedirs(path_to_outlier_detection, exist_ok=True)

        pd.DataFrame().to_csv(path_microbiome, sep="\t")
        pd.DataFrame().to_csv(path_response, sep="\t")

        preprocessor = Preprocessor(
            logger=logger,
            s1="s1",
            s2="s2",
            s3="s3",
            path_to_microbiome=path_microbiome,
            path_to_response=path_response,
            path_to_microbiome_clustering=tmp_path / "clustering",
            path_to_prepared_response=tmp_path / "prepared_response",
            path_to_sparcckit_results=tmp_path / "sparcckit_results",
            path_to_sparcckit_corr=tmp_path / "sparcckit_corr",
            path_to_sparcckit_cov=tmp_path / "sparcckit_cov",
            path_to_outputs=tmp_path / "outputs",
            path_to_clustered_ocu=tmp_path / "clustered_ocu",
            path_to_response_vs_balance_plots=tmp_path / "balance_plots",
            balance_methods_list=["CLR"],
            sparcckit_iter=100,
            clustering_sampling_rate=1,
            outlier_removal_tag=True,
        )

        yield preprocessor

    @pytest.fixture(scope='function')
    def test_series_without_groups(self):
        sample_tags = ["A", "B", "C", "D", "E", "F"]
        response_tag = "response"
        response_series = pd.Series([1, 2, 3, 100, 4, 5], index=sample_tags, name=response_tag)
        expected_mask = pd.Series([True, True, True, False, True, True], index=sample_tags, name=response_tag)
        yield response_series, expected_mask

    @pytest.fixture(scope='function')
    def test_series_without_groups_min_samples(self):
        sample_tags = ["A", "B", "C"]
        response_tag = "response"
        response_series = pd.Series([1, 2, 3], index=sample_tags, name=response_tag)
        expected_mask = pd.Series([True, True, True], index=sample_tags, name=response_tag)
        yield response_series, expected_mask

    @pytest.fixture(scope='function')
    def test_series_with_groups(self, tmp_path):
        sample_tags = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        response_tag = "response"
        group_tags = pd.Series(["X", "X", "X", "X", "X", "Y", "Y", "Y", "Y", "Y"], index=sample_tags, name="Category")
        group_tags.to_csv(tmp_path / "s1-s2-metadata.csv", sep="\t", header=True)
        response_series = pd.Series([1, 2, 3, 100, 4, 5, 6, 200, 7, 8], index=sample_tags, name=response_tag)
        expected_mask = pd.Series(
            [True, True, True, False, True, True, True, False, True, True],
            index=sample_tags, name=response_tag, dtype=bool
        )
        yield response_series, expected_mask

    @pytest.fixture(scope='function')
    def test_series_with_groups_min_samples(self, tmp_path):
        sample_tags = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
        response_tag = "response"
        group_tags = pd.Series(["X", "X", "X", "X", "X", "Y", "Y", "Y", "Y"], index=sample_tags, name="Category")
        group_tags.to_csv(tmp_path / "s1-s2-metadata.csv", sep="\t", header=True)
        response_series = pd.Series([1, 2, 3, 100, 4, 5, 6, 200, 7], index=sample_tags, name=response_tag)
        expected_mask = pd.Series(
            [True, True, True, False, True, True, True, True, True],
            index=sample_tags, name=response_tag
        )
        yield response_series, expected_mask

    @pytest.fixture(scope='function')
    def setup_compare_slope_arrays(self):
        tested_arrays = {
            'slope_array_before': np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            'se_array_before': np.array([0.01, 0.02, 0.03, 0.04, 0.05]),
            'slope_array_after': np.array([0.11, 0.25, 0.28, 0.6, 0.55]),
            'se_array_after': np.array([0.01, 0.025, 0.035, 0.04, 0.045]),
            'response_index': ['tag1', 'tag2', 'tag3', 'tag4', 'tag5']
        }
        expected_results = np.array([2.4e-01, 0.6e-01, 3.3e-01, 2.0e-04, 2.3e-01])
        yield tested_arrays, expected_results

    def test_prepare_response_outlier_mask_without_groups(self, test_preprocessor, test_series_without_groups):
        test_response, expected_mask = test_series_without_groups
        mask, _ = test_preprocessor.prepare_response_outlier_mask(test_response)
        pd.testing.assert_series_equal(mask, expected_mask)

    def test_prepare_response_outlier_mask_without_groups_min_samples(
            self, test_preprocessor, test_series_without_groups_min_samples
    ):
        test_response, expected_mask = test_series_without_groups_min_samples
        mask, _ = test_preprocessor.prepare_response_outlier_mask(test_response)
        pd.testing.assert_series_equal(mask, expected_mask)

    def test_prepare_response_outlier_mask_with_groups(self, test_preprocessor, test_series_with_groups, tmp_path):
        test_preprocessor.meta_data = tmp_path / "s1-s2-metadata.csv"
        test_response, expected_mask = test_series_with_groups
        mask, _ = test_preprocessor.prepare_response_outlier_mask(test_response)
        pd.testing.assert_series_equal(mask, expected_mask)

    def test_prepare_response_outlier_mask_with_groups_min_samples(
            self, test_preprocessor, test_series_with_groups_min_samples, tmp_path
    ):
        test_preprocessor.meta_data = tmp_path / "s1-s2-metadata.csv"
        test_response, expected_mask = test_series_with_groups_min_samples
        mask, _ = test_preprocessor.prepare_response_outlier_mask(test_response)
        pd.testing.assert_series_equal(mask, expected_mask)

    def test_find_common_outlier_mask_remove(self, test_preprocessor, tmp_path):
        df = pd.DataFrame(
            {
                "Data1": [1, 2, 3, 4, 50, 6],
                "Data2": [4, 5, 6, 7, 60, 9],
                "Data3": [7, 8, 9, 10, 70, 12],
            }, index=["sample4", "sample2", "sample1", "sample3", "sample5", "sample6"]
        )
        df.to_csv(tmp_path / "prepared_response", sep="\t")
        expected_mask = pd.Series(
            [True, True, True, True, False, True],
            index=["sample4", "sample2", "sample1", "sample3", "sample5", "sample6"],
        )
        expected_mask.name = "CommonMask"
        mask = test_preprocessor.find_common_outlier_mask()
        pd.testing.assert_series_equal(mask, expected_mask)

    def test_find_common_outlier_mask_remove_by_share(self, test_preprocessor, tmp_path):
        df = pd.DataFrame({
            "Data1": [1, 2, 3, 4, 50, 6],
            "Data2": [4, 5, 6, 7, 60, 9],
            "Data3": [7, 8, 9, 10, 70, 12],
            "Data4": [10, 11, 12, 13, 14, 15]},
            index=['sample4', 'sample2', 'sample1', 'sample3', 'sample5', 'sample6']
        )
        df.to_csv(tmp_path / "prepared_response", sep="\t")
        expected_mask = pd.Series([True, True, True, True, False, True],
                                  index=['sample4', 'sample2', 'sample1', 'sample3', 'sample5', 'sample6']).astype(bool)
        expected_mask.name = "CommonMask"
        mask = test_preprocessor.find_common_outlier_mask()
        pd.testing.assert_series_equal(mask, expected_mask)

    def test_find_common_outlier_mask_no_removal(self, test_preprocessor, tmp_path):

        df = pd.DataFrame({
            "Data1": [1, 2, 3, 4, 50, 6],
            "Data2": [4, 5, 6, 7, 15, 9],
            "Data3": [7, 8, 9, 10, 11, 12],
            "Data4": [10, 11, 12, 13, 14, 15]},
            index=['sample4', 'sample2', 'sample1', 'sample3', 'sample5', 'sample6']
        )
        df.to_csv(tmp_path / "prepared_response", sep="\t")
        expected_mask = pd.Series([True, True, True, True, False, True],
                                  index=['sample4', 'sample2', 'sample1', 'sample3', 'sample5', 'sample6'], dtype=bool)
        expected_mask.name = "CommonMask"
        mask = test_preprocessor.find_common_outlier_mask()
        pd.testing.assert_series_equal(mask, expected_mask)

    def test_compare_slope_arrays(self, test_preprocessor, setup_compare_slope_arrays):

        tested_arrays, expected_array = setup_compare_slope_arrays
        result = test_preprocessor.calculate_slope_change_p_values(**tested_arrays)

        assert os.path.exists(os.path.join(
            test_preprocessor.outputs_path, 'preprocessing_results', 'outlier_detection', "slope_change_stats.tsv"
        ))
        assert np.allclose(result, expected_array, atol=0.1)

    def test_calculate_slopes_and_ses(self, logger, setup_teardown_set_config, test_preprocessor):
        cfg_dict = setup_teardown_set_config
        logger=logger
        combination = OneCaseCombination(
            logger,
            cfg_dict,
            True,
            "s1",
            "s2",
            "s3",
            ocu_case="s1-s2-s3",
            deduplicate=True,
        )
        input_microbiome_data = {
            "SampleID": ["C10.d4", "C11.d4", "C7.d4", "C8.d4", "N10.d4"],
            "f_A_1683": [0.1, 0.2, 0.25, 0.3, 0.35],
            "f_B_1707": [0.4, 0.25, 0.3, 0.35, 0.45],
            "f_C_1645": [0.5, 0.55, 0.45, 0.35, 0.2],
        }
        microbiome_df = pd.DataFrame(input_microbiome_data).set_index("SampleID")
        input_response_data = {
            "SampleID": ["C10.d4", "C11.d4", "C7.d4", "C8.d4", "N10.d4"],
            "GO_0045071_enh": [1.0, 0, 1.005, 1.021461975, 1.0],
            "GO_0032823_enh": [1.006384536, 0, 3, 3, 3.006384531],
            "GO_0045887_enh": [2, 2, 2.011461975, 0, 3.005156721],
        }
        response_df = pd.DataFrame(input_response_data).set_index("SampleID")
        response_df.to_csv(test_preprocessor.path_to_prepared_response, sep="\t")
        combination.response_index = extract_response_tags(
            test_preprocessor.path_to_prepared_response, combination.intermediate_results_path
        )

        expected_slope_array = np.array([1.11, -2.37, 1.36])
        expected_se_array = np.array([0.78, 1.22, 2.34])

        all_sample_slope_array, all_sample_se_array, _ = test_preprocessor.calculate_slopes_and_ses(
            combination, microbiome_df, response_df, True, True, False
        )
        assert np.allclose(all_sample_slope_array, expected_slope_array, atol=0.01)
        assert np.allclose(all_sample_se_array, expected_se_array, atol=0.01)
