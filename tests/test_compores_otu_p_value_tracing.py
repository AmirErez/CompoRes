import os
import pickle
import shutil

import numpy as np
import pandas as pd
import pytest

from src.compores.compores_otu_p_value_tracing import ComporesClusteredPValueCalculations
from src.compores.exceptions_module import NoResponseLabelFound


class TestComporesClusteredPValueCalculations:
    @pytest.fixture(scope="function")
    def setup_teardown_set_paths(self):
        cfg_file = {
            "PATH_TO_MICROBIOME": os.path.join("path", "to", "microbiome"),
            "PATH_TO_RESPONSE": os.path.join("path", "to", "response"),
            "PATH_TO_METADATA": os.path.join("path", "to", "metadata"),
            "PATH_TO_OUTPUTS": os.path.join("path", "to", "outputs"),
            "OCU_SAMPLING_RATE": 10,
            "CODA_METHOD": 'pairs',
            "CORR": 'pearson',
            "SHUFFLE": 'response',
            "N_SHUFFLES": 10,
            "SHUFFLE_CYCLES": 3,
            "N_WORKERS": 4,
        }

        # Write a microbiome file to the mock source folder
        taxa = ['taxa_1', 'taxa_2', 'taxa_3', 'taxa_4', 'taxa_5', 'taxa_6']
        microbiome_df = pd.DataFrame(np.random.rand(5, 6), columns=taxa)
        preprocessed_microbiome_path = os.path.join(cfg_file["PATH_TO_OUTPUTS"], "preprocessing_results", "microbiome")
        os.makedirs(preprocessed_microbiome_path, exist_ok=True)
        microbiome_df.to_csv(os.path.join(preprocessed_microbiome_path, 's1-s2-s3.tsv'), sep='\t')

        # Write response index pkl file to the mock source folder
        response_tags = ['response_1', 'response_2', 'response_3']
        for balance_method in ['CLR', 'pairs']:
            response_tags_path = os.path.join(
                 cfg_file["PATH_TO_OUTPUTS"], 'compores_basic_results', 's1-s2-s3', balance_method
             )
            os.makedirs(response_tags_path, exist_ok=True)
            with open(os.path.join(response_tags_path, "response_index.pkl"), "wb") as f:
                pickle.dump(response_tags, f)

        # Write mock pairs_minus_mean_log_p_values file for r_label
        empty_df = pd.DataFrame(np.zeros((6, 6)), index=taxa, columns=taxa)
        otu_pairs_tracing_path = os.path.join(
            cfg_file["PATH_TO_OUTPUTS"], 'otu_significance_tracing', 's1-s2-s3', 'pairs'
        )
        os.makedirs(otu_pairs_tracing_path, exist_ok=True)
        pd.DataFrame(empty_df.astype(int)).to_parquet(
            os.path.join(otu_pairs_tracing_path, 's1-s2-s3_otu_pairs_minus_mean_p_values_r_label.parquet')
        )

        yield cfg_file, "s1", "s2", "s3"

        if os.path.exists("path"):
            shutil.rmtree("path")

    @pytest.fixture(scope="function")
    def setup_teardown_update_all_pairs_sum_p_values(self):

        correlation_df = pd.DataFrame(
            np.array([[np.nan, .4, .5], [0.02, np.nan, np.nan], [np.nan, .3, np.nan]]),
            columns=['ocu_1', 'ocu_2', 'ocu_3'],
            index=['ocu_1', 'ocu_2', 'ocu_3']
        )

        partial_ocu_dict = {
            'ocu_1': {'taxa': ['taxa_1', 'taxa_2', 'taxa_3'], 'imputed_in': []},
            'ocu_2': {'taxa': ['taxa_4', 'taxa_5'], 'imputed_in': []},
            'ocu_3': {'taxa': ['taxa_6'], 'imputed_in': []}
        }

        taxa = ['taxa_1', 'taxa_2', 'taxa_3', 'taxa_4', 'taxa_5', 'taxa_6']
        expected_df = pd.DataFrame(
            {
                'taxa_1': [0.000000, 0.000000, 0.000000, 0.652004, 0.652004, 0.000000],
                'taxa_2': [0.000000, 0.000000, 0.000000, 0.652004, 0.652004, 0.000000],
                'taxa_3': [0.000000, 0.000000, 0.000000, 0.652004, 0.652004, 0.000000],
                'taxa_4': [0.152715, 0.152715, 0.152715, 0.000000, 0.000000, 0.601986],
                'taxa_5': [0.152715, 0.152715, 0.152715, 0.000000, 0.000000, 0.601986],
                'taxa_6': [0.231049, 0.231049, 0.231049, 0.000000, 0.000000, 0.000000]
            },
            index=taxa, columns=taxa
        )

        yield correlation_df, partial_ocu_dict, expected_df

    @pytest.fixture(scope="function")
    def setup_teardown_update_all_clr_sum_p_values(self):
        correlation_df = pd.DataFrame(
            np.array([.4, 0.02, .3]),
            index=['ocu_1', 'ocu_2', 'ocu_3']
        )

        partial_ocu_dict = {
            'ocu_1': {'taxa': ['taxa_1', 'taxa_2', 'taxa_3'], 'imputed_in': []},
            'ocu_2': {'taxa': ['taxa_4', 'taxa_5'], 'imputed_in': []},
            'ocu_3': {'taxa': ['taxa_6'], 'imputed_in': []}
        }

        taxa = ['taxa_1', 'taxa_2', 'taxa_3', 'taxa_4', 'taxa_5', 'taxa_6']
        expected_df = pd.DataFrame(
            np.array([[0.305430, 0], [0.305430, 0], [0.305430, 0], [1.956012 , 0], [1.956012 , 0], [1.203973, 0]]),
            index=taxa, columns=['minus_mean_log_p_value_positive', 'minus_mean_log_p_value_negative']
        )

        yield correlation_df, partial_ocu_dict, expected_df

    @pytest.fixture(scope="function")
    def setup_teardown_final_output(self):

        correlation_df = pd.DataFrame(
            np.array([[np.nan, .4, .5], [0.02, np.nan, np.nan], [np.nan, .3, np.nan]]),
            columns=['ocu_1', 'ocu_2', 'ocu_3'],
            index=['ocu_1', 'ocu_2', 'ocu_3']
        )

        partial_ocu_dict = {
            'ocu_1': {'taxa': ['taxa_1', 'taxa_2', 'taxa_3'], 'imputed_in': []},
            'ocu_2': {'taxa': ['taxa_4', 'taxa_5'], 'imputed_in': []},
            'ocu_3': {'taxa': ['taxa_6'], 'imputed_in': []}
        }

        taxa = ['taxa_1', 'taxa_2', 'taxa_3', 'taxa_4', 'taxa_5', 'taxa_6']
        expected_df = pd.DataFrame(
            {
                'p_value_estimate_positive': [0.836251, 0.836251, 0.836251, 0.521001, 0.521001, 0.547723],
                'p_value_estimate_negative': [0.521001, 0.521001, 0.521001, 0.767181, 0.767181, 0.793701],
            },
            index=taxa,
        )

        yield correlation_df, partial_ocu_dict, expected_df

    def test_create_instance(self, setup_teardown_set_paths):
        cfg_file, s1, s2, s3 = setup_teardown_set_paths

        tested_class = ComporesClusteredPValueCalculations(cfg_file, s1, s2, s3, init_flag=True)
        assert tested_class.config_dict == cfg_file
        assert tested_class.g1 == s1
        assert tested_class.g2 == s2
        assert tested_class.g3 == s3
        assert tested_class.balance_methods == ['CLR', 'pairs']
        assert tested_class.exp_name == f'{s1}-{s2}-{s3}'
        assert tested_class.outputs_path == cfg_file["PATH_TO_OUTPUTS"]
        assert tested_class.otu_p_value_tracing_path == os.path.join(
            cfg_file["PATH_TO_OUTPUTS"], 'otu_significance_tracing', tested_class.exp_name,
        )
        assert tested_class.ocu_sampling_rate == cfg_file["OCU_SAMPLING_RATE"]
        assert tested_class.path_to_preprocessed_microbiome == os.path.join(
            cfg_file["PATH_TO_OUTPUTS"], "preprocessing_results", "microbiome"
        )
        assert tested_class.balance_results_path == os.path.join(
            cfg_file["PATH_TO_OUTPUTS"], 'balance_calculation_results', tested_class.exp_name
        )
        assert tested_class.compores_basic_results_path == os.path.join(
            cfg_file["PATH_TO_OUTPUTS"], 'compores_basic_results', tested_class.exp_name
        )
        assert tested_class.response_index is None
        assert tested_class.response_label is None
        assert tested_class.current_otu_tracing_file is None
        assert tested_class.current_normalization_matrix is None

        # Check that the required parquet files have been created
        for balance_method in tested_class.balance_methods:
            response_tags_path = os.path.join(
                tested_class.compores_basic_results_path, balance_method
            )
            with open(os.path.join(response_tags_path, "response_index.pkl"), "rb") as f:
                response_tags = pickle.load(f)
            for r_tag in response_tags:
                otu_tracing_path = os.path.join(
                    tested_class.otu_p_value_tracing_path, balance_method,
                    f'{tested_class.exp_name}_otu_{balance_method.lower()}_minus_mean_log_p_values_{r_tag}.parquet'
                )
                normalize_matrix_path = os.path.join(
                    tested_class.otu_p_value_tracing_path, balance_method,
                    f'{tested_class.exp_name}_otu_{balance_method.lower()}_normalization_count_{r_tag}.parquet'
                )
                assert os.path.exists(otu_tracing_path)
                assert os.path.exists(normalize_matrix_path)

    def test_set_current_response(self, setup_teardown_set_paths):
        cfg_file, s1, s2, s3 = setup_teardown_set_paths
        tested_class = ComporesClusteredPValueCalculations(cfg_file, s1, s2, s3, init_flag=True)
        tested_class.set_current_response(1, 'r_label', 'pairs')
        assert tested_class.response_index == 1
        assert tested_class.response_label == 'r_label'
        assert tested_class.current_otu_tracing_file == os.path.join(
            tested_class.otu_p_value_tracing_path, 'pairs',
            f'{tested_class.exp_name}_otu_pairs_minus_mean_log_p_values_r_label.parquet'
        )

    def test_update_all_pair_items_sum_p_values(
            self , setup_teardown_set_paths, setup_teardown_update_all_pairs_sum_p_values
    ):
        cfg_file, s1, s2, s3 = setup_teardown_set_paths

        correlation_df, partial_ocu_dict, expected_df = setup_teardown_update_all_pairs_sum_p_values

        tested_class = ComporesClusteredPValueCalculations(cfg_file, 's1', 's2', 's3', init_flag=True)
        tested_class.set_current_response(1, 'response_2', 'pairs')

        tested_class.update_all_pair_items_sum_p_values(
            correlation_df, partial_ocu_dict
        )

        # Assert the csv file was created in the correct path
        assert os.path.exists(tested_class.current_otu_tracing_file)

        # Read the resulted values are equal to the expected_df values
        result_df = pd.read_parquet(tested_class.current_otu_tracing_file)
        assert np.allclose(result_df, expected_df)
        assert [c_r == c_e for (c_r, c_e) in zip(result_df.columns, expected_df.columns)]
        assert [i_r == i_e for (i_r, i_e) in zip(result_df.index, expected_df.index)]

        result_norm_df = pd.read_parquet(tested_class.current_normalization_matrix)
        expected_norm_df_mask = expected_df != 0
        assert np.allclose(result_norm_df.values, expected_norm_df_mask.values)

    def test_update_all_clr_items_sum_p_values(
            self , setup_teardown_set_paths, setup_teardown_update_all_clr_sum_p_values, monkeypatch
    ):
        cfg_file, s1, s2, s3 = setup_teardown_set_paths

        correlation_df, partial_ocu_dict, expected_df = setup_teardown_update_all_clr_sum_p_values

        tested_class = ComporesClusteredPValueCalculations(cfg_file, 's1', 's2', 's3', init_flag=True)
        tested_class.set_current_response(1, 'response_2', 'CLR')

        tested_class.update_all_clr_items_sum_p_values(correlation_df, correlation_df, partial_ocu_dict)

        # Assert the csv file was created in the correct path
        assert os.path.exists(tested_class.current_otu_tracing_file)

        # Read the resulted values are equal to the expected_df values
        result_df = pd.read_parquet(tested_class.current_otu_tracing_file)
        assert np.allclose(result_df, expected_df)
        assert result_df.columns[0] == 'minus_mean_log_p_value_positive'

    def test_update_all_clr_items_sum_p_values_like(
            self , setup_teardown_set_paths, setup_teardown_update_all_clr_sum_p_values, monkeypatch
    ):
        cfg_file, s1, s2, s3 = setup_teardown_set_paths

        correlation_df, partial_ocu_dict, expected_df = setup_teardown_update_all_clr_sum_p_values

        tested_class = ComporesClusteredPValueCalculations(cfg_file, 's1', 's2', 's3', init_flag=True)
        tested_class.set_current_response(1, 'response_2', 'CLR')

        tested_class.update_all_clr_items_sum_p_values(correlation_df, correlation_df, partial_ocu_dict)

        # Assert the csv file was created in the correct path
        assert os.path.exists(tested_class.current_otu_tracing_file)

        # Read the resulted values are equal to the expected_df values
        result_df = pd.read_parquet(tested_class.current_otu_tracing_file)
        assert result_df.columns[0] == 'minus_mean_log_p_value_positive'
        assert [i_r == i_e for (i_r, i_e) in zip(result_df.index, expected_df.index)]

    def test_prepare_response_list_to_trace(self, setup_teardown_set_paths):
        test_cases = [
            {
                "name": "with_valid_response_tag",
                "response_tag": "resp",
                "expected_result": ["response_1"],
                "should_raise": False,
            },
            {
                "name": "with_no_response_tag",
                "response_tag": None,
                "expected_result": ["response_1", "response_2", "response_3"],
                "should_raise": False,
            },
            {
                "name": "with_invalid_response_tag",
                "response_tag": "invalid_tag",
                "should_raise": True,
            }
        ]

        for test_case in test_cases:

            cfg_file, s1, s2, s3 = setup_teardown_set_paths
            test_instance = ComporesClusteredPValueCalculations(cfg_file, s1, s2, s3)
            test_path = os.path.join(cfg_file["PATH_TO_OUTPUTS"], "compores_basic_results", "s1-s2-s3", 'CLR')

            if test_case["should_raise"]:
                with pytest.raises(NoResponseLabelFound):
                    result = test_instance.prepare_response_list_to_trace(
                            test_path,
                            test_case["response_tag"]
                        )
            else:
                result = test_instance.prepare_response_list_to_trace(
                        test_path,
                        test_case["response_tag"]
                    )
                assert result == test_case["expected_result"]

                if test_case["response_tag"]:
                    # Check if response_label was set correctly
                    assert hasattr(test_instance, 'response_label')
                    assert test_instance.response_label == test_case["expected_result"][0]

    def test_normalize_df_basic(self):

        norm = pd.DataFrame(
            [[2, 0], [0, 4]],
            index=["a", "b"],
            columns=["x", "y"],
        ).astype(float)

        traces = pd.DataFrame(
            [[2.0, 8.0], [6.0, 16.0]],
            index=["c", "d"],
            columns=["z", "w"],
        )

        result = ComporesClusteredPValueCalculations.normalize_df(norm, traces)

        expected = pd.DataFrame(
            [[1.0, 8.0], [6.0, 4.0]],
            index=["a", "b"],
            columns=["x", "y"],
        )


        assert result is traces
        assert np.allclose(result.values, expected.values)

    def test_prepare_final_otu_tracing_output(
            self , setup_teardown_set_paths, setup_teardown_final_output
    ):
        cfg_file, s1, s2, s3 = setup_teardown_set_paths

        correlation_df, partial_ocu_dict, expected_df = setup_teardown_final_output

        tested_class = ComporesClusteredPValueCalculations(cfg_file, 's1', 's2', 's3', init_flag=True)
        tested_class.set_current_response(1, 'response_2', 'pairs')

        tested_class.update_all_pair_items_sum_p_values(
            correlation_df, partial_ocu_dict
        )
        tested_class.prepare_final_otu_tracing_output()

        # Assert the final csv file was created in the correct path
        result_df = pd.read_csv(tested_class.current_estimated_p_value_file.replace(
            "pairs_estimated", "pairs_condensed_estimated"
        ), sep="\t", index_col=0)

        assert np.allclose(result_df.values, expected_df.values)