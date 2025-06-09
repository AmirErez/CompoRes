import os
import shutil

import numpy as np
import pandas as pd
import pytest

from src.compores.compores_otu_heatmaps import ComporesClusteredHeatmapCalculations


class TestComporesClusteredHeatmapCalculations:
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

        # Write microbiome file to the mock source folder
        taxa = ['taxa_1', 'taxa_2', 'taxa_3', 'taxa_4', 'taxa_5', 'taxa_6']
        microbiome_df = pd.DataFrame(np.random.rand(5, 6), columns=taxa)
        preprocessed_microbiome_path = os.path.join(cfg_file["PATH_TO_OUTPUTS"], "preprocessing_results", "microbiome")
        os.makedirs(preprocessed_microbiome_path, exist_ok=True)
        microbiome_df.to_csv(os.path.join(preprocessed_microbiome_path, 's1-s2-s3.tsv'), sep='\t')

        # Write response index pkl file to the mock source folder
        response_tags = pd.Series(['response_1', 'response_2', 'response_3'])
        response_tags_path = os.path.join(cfg_file["PATH_TO_OUTPUTS"], "compores_basic_results",
                                          "s1-s2-s3", 'CLR'
                                          )
        os.makedirs(response_tags_path, exist_ok=True)
        response_tags.to_pickle(os.path.join(response_tags_path, 'response_index.pkl'))

        # Write mock pairs_minus_sum_log_p_values file
        empty_df = pd.DataFrame(np.zeros((6, 6)), index=taxa, columns=taxa)
        otu_pairs_tracing_path = os.path.join(
            cfg_file["PATH_TO_OUTPUTS"], 'otu_significance_tracing', 's1-s2-s3', 'pairs'
        )
        os.makedirs(otu_pairs_tracing_path, exist_ok=True)
        pd.DataFrame(empty_df).to_parquet(
            os.path.join(otu_pairs_tracing_path, 's1-s2-s3_otu_pairs_minus_sum_log_p_values_r_label.parquet')
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
                'taxa_1': [0.000000, 0.000000, 0.000000, 3.912023, 3.912023, 0.000000],
                'taxa_2': [0.000000, 0.000000, 0.000000, 3.912023, 3.912023, 0.000000],
                'taxa_3': [0.000000, 0.000000, 0.000000, 3.912023, 3.912023, 0.000000],
                'taxa_4': [0.916291, 0.916291, 0.916291, 0.000000, 0.000000, 1.203973],
                'taxa_5': [0.916291, 0.916291, 0.916291, 0.000000, 0.000000, 1.203973],
                'taxa_6': [0.693147, 0.693147, 0.693147, 0.000000, 0.000000, 0.000000]
            },
            index=taxa, columns=taxa
        )

        yield correlation_df, partial_ocu_dict, expected_df

    @pytest.fixture(scope="function")
    def setup_teardown_update_all_diagonal_sum_p_values(self):
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
        expected_df = {
            'pairs_like_false': pd.DataFrame(
                np.array([0.91629073, 0.91629073, 0.91629073, 3.91202301, 3.91202301, 1.2039728]),
                index=taxa, columns=['0']
            ),
            'pairs_like_true': pd.DataFrame(
                np.array([6.94857727, 6.94857727, 6.94857727, 9.94430955, 9.94430955, 7.23625935]),
                index=taxa, columns=['0']
            )
        }


        yield correlation_df, partial_ocu_dict, expected_df


    def test_create_instance(self, setup_teardown_set_paths):
        cfg_file, s1, s2, s3 = setup_teardown_set_paths

        tested_class = ComporesClusteredHeatmapCalculations(cfg_file, s1, s2, s3)
        assert tested_class.config_dict == cfg_file
        assert tested_class.g1 == s1
        assert tested_class.g2 == s2
        assert tested_class.g3 == s3
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
        assert tested_class.all_pairs_file is None

    def test_set_current_response(self, setup_teardown_set_paths):
        cfg_file, s1, s2, s3 = setup_teardown_set_paths
        tested_class = ComporesClusteredHeatmapCalculations(cfg_file, s1, s2, s3)
        tested_class.set_current_response(1, 'r_label')
        assert tested_class.response_index == 1
        assert tested_class.response_label == 'r_label'
        assert tested_class.all_pairs_file == os.path.join(
            tested_class.otu_p_value_tracing_path, 'pairs',
            f'{tested_class.exp_name}_otu_pairs_minus_sum_log_p_values_r_label.parquet'
        )

    def test_update_all_pairs_sum_p_values(
            self , setup_teardown_set_paths, setup_teardown_update_all_pairs_sum_p_values
    ):
        cfg_file, s1, s2, s3 = setup_teardown_set_paths

        correlation_df, partial_ocu_dict, expected_df = setup_teardown_update_all_pairs_sum_p_values

        tested_class = ComporesClusteredHeatmapCalculations(cfg_file, 's1', 's2', 's3')
        tested_class.set_current_response(1, 'response_2')

        tested_class.update_all_pairs_sum_p_values(
            correlation_df, partial_ocu_dict
        )

        # Assert the csv file was created in the correct path
        assert os.path.exists(tested_class.all_pairs_file)

        # Read the resulted values are equal to the expected_df values
        result_df = pd.read_parquet(tested_class.all_pairs_file)
        assert np.allclose(result_df, expected_df)
        assert [c_r == c_e for (c_r, c_e) in zip(result_df.columns, expected_df.columns)]
        assert [i_r == i_e for (i_r, i_e) in zip(result_df.index, expected_df.index)]

    def test_update_all_diagonal_sum_p_values(
            self , setup_teardown_set_paths, setup_teardown_update_all_diagonal_sum_p_values, monkeypatch
    ):
        cfg_file, s1, s2, s3 = setup_teardown_set_paths

        correlation_df, partial_ocu_dict, expected_df = setup_teardown_update_all_diagonal_sum_p_values

        tested_class = ComporesClusteredHeatmapCalculations(cfg_file, 's1', 's2', 's3')
        tested_class.set_current_response(1, 'response_2')

        def mock_update_otu_cumulative_p_value_analysis_state(self, coda_method: str, value: bool):
            pass
        monkeypatch.setattr(
            ComporesClusteredHeatmapCalculations,
            "update_otu_cumulative_p_value_analysis_state",
            mock_update_otu_cumulative_p_value_analysis_state
        )

        tested_class.update_all_diagonal_sum_p_values(correlation_df, partial_ocu_dict, pairs_like=False)

        # Assert the csv file was created in the correct path
        assert os.path.exists(tested_class.diagonal_file)

        # Read the resulted values are equal to the expected_df values
        result_df = pd.read_parquet(tested_class.diagonal_file)
        print(result_df)
        print(expected_df['pairs_like_false'])
        assert np.allclose(result_df, expected_df['pairs_like_false'])
        assert result_df.columns == 'minus_sum_log_p_values'
        assert [i_r == i_e for (i_r, i_e) in zip(result_df.index, expected_df['pairs_like_false'].index)]

    def test_update_all_diagonal_sum_p_values_pairs_like(
            self , setup_teardown_set_paths, setup_teardown_update_all_diagonal_sum_p_values, monkeypatch
    ):
        cfg_file, s1, s2, s3 = setup_teardown_set_paths

        correlation_df, partial_ocu_dict, expected_df = setup_teardown_update_all_diagonal_sum_p_values

        tested_class = ComporesClusteredHeatmapCalculations(cfg_file, 's1', 's2', 's3')
        tested_class.set_current_response(1, 'response_2')

        def mock_update_otu_cumulative_p_value_analysis_state(self, coda_method: str, value: bool):
            pass
        monkeypatch.setattr(
            ComporesClusteredHeatmapCalculations,
            "update_otu_cumulative_p_value_analysis_state",
            mock_update_otu_cumulative_p_value_analysis_state
        )

        tested_class.update_all_diagonal_sum_p_values(correlation_df, partial_ocu_dict, pairs_like=True)

        # Assert the csv file was created in the correct path
        assert os.path.exists(tested_class.diagonal_file)

        # Read the resulted values are equal to the expected_df values
        result_df = pd.read_parquet(tested_class.diagonal_file)
        assert np.allclose(result_df, expected_df['pairs_like_true'])
        assert result_df.columns == 'minus_sum_log_p_values'
        assert [i_r == i_e for (i_r, i_e) in zip(result_df.index, expected_df['pairs_like_false'].index)]