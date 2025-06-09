import os
import shutil
import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .preprocessing import PREPROCESSING_RESULTS
from .utils import bootstrap_p_value, load_file


class ComporesClusteredHeatmapCalculations:
    def __init__(self, config_dict: dict, g1: str, g2: str, g3: str):
        self.config_dict = config_dict
        self.g1 = g1
        self.g2 = g2
        self.g3 = g3
        self.balance_methods = ['CLR']
        if self.config_dict["CODA_METHOD"] and self.config_dict["CODA_METHOD"] != '':
            self.balance_methods = self.balance_methods + [self.config_dict["CODA_METHOD"]]
        self.exp_name = f'{self.g1}-{self.g2}-{self.g3}'
        self.outputs_path = self.config_dict["PATH_TO_OUTPUTS"]
        self.path_to_preprocessed_microbiome = os.path.join(self.outputs_path, PREPROCESSING_RESULTS, "microbiome")
        self.path_to_preprocessed_response = os.path.join(self.outputs_path, PREPROCESSING_RESULTS, "response")
        self.balance_results_path = os.path.join(self.outputs_path, 'balance_calculation_results', self.exp_name)
        self.compores_basic_results_path = os.path.join(self.outputs_path, 'compores_basic_results', self.exp_name)
        self.ocu_sampling_rate = self.config_dict["OCU_SAMPLING_RATE"]
        self.otu_p_value_tracing_path = os.path.join(self.outputs_path, 'otu_significance_tracing', self.exp_name)
        self._prepare_heatmap_folders()
        self.response_index = None
        self.response_label = None
        self.all_pairs_file = None
        self.diagonal_file = None
        self.final_heatmap_file = None
        self.state = self.load_state()

    def load_state(self) -> bool:
        try:
            with open(os.path.join(self.outputs_path, f'state_{self.balance_methods[-1]}.json'), 'r') as f:
                state_dict = json.load(f)
        except FileNotFoundError:
            state_dict = {
                f'{self.g1}-{self.g2}-{self.g3}': {
                    'otu_cumulative_p_value': False
                }
            }
        return state_dict[f'{self.g1}-{self.g2}-{self.g3}']['otu_cumulative_p_value']

    def update_otu_cumulative_p_value_analysis_state(self, coda_method: str, value: bool):
        with open(os.path.join(self.outputs_path, f'state_{coda_method}.json'), 'r') as f:
            state_dict = json.load(f)
        state_dict[f'{self.g1}-{self.g2}-{self.g3}']['otu_cumulative_p_value'] = value
        with open(os.path.join(self.outputs_path, f'state_{coda_method}.json'), 'w') as f:
            json.dump(state_dict, f, indent=4)

    def _prepare_heatmap_folders(self):
        if os.path.exists(self.otu_p_value_tracing_path):
            shutil.rmtree(self.otu_p_value_tracing_path)
        os.makedirs(self.otu_p_value_tracing_path)

        processed_microbiome = pd.read_csv(
            os.path.join(str(self.path_to_preprocessed_microbiome), f'{self.exp_name}.tsv'),
            index_col=0, sep='\t'
        )
        taxa = processed_microbiome.columns
        taxa_str = [str(t) for t in taxa]

        path_to_intermediate_results = os.path.join(self.outputs_path, 'compores_basic_results', self.exp_name, 'CLR')
        response_labels = load_file('response_index.pkl', path_to_intermediate_results)

        if 'pairs' in self.balance_methods:
            os.makedirs(os.path.join(self.otu_p_value_tracing_path, 'pairs'))
            data_p = np.zeros((len(taxa), len(taxa)))
            all_pairs = pd.DataFrame(data_p, index=taxa_str, columns=taxa_str)
            for response_label in response_labels:
                with open(os.path.join(
                        self.otu_p_value_tracing_path, 'pairs',
                        f'{self.exp_name}_otu_pairs_minus_sum_log_p_values_{response_label}.parquet'
                ), 'wb') as f:
                    all_pairs.to_parquet(f, engine="fastparquet")

        os.makedirs(os.path.join(self.otu_p_value_tracing_path, 'CLR'))
        data_d = np.zeros(len(taxa))
        diagonal_items = pd.DataFrame(data_d, index=taxa_str)
        diagonal_items.columns = ['minus_sum_log_p_values']
        for response_label in response_labels:
            with open(os.path.join(
                    self.otu_p_value_tracing_path, 'CLR',
                    f'{self.exp_name}_otu_diagonal_minus_sum_log_p_values_{response_label}.parquet'
            ), 'wb') as f:
                diagonal_items.to_parquet(f, engine="fastparquet")

    def set_current_response(self, response_index: int, response_label: str):

        self.response_index = response_index
        self.response_label = response_label

        self.all_pairs_file = os.path.join(
            self.otu_p_value_tracing_path, 'pairs',
            f'{self.exp_name}_otu_pairs_minus_sum_log_p_values_{response_label}.parquet'
        )
        self.diagonal_file = os.path.join(
            self.otu_p_value_tracing_path, 'CLR',
            f'{self.exp_name}_otu_diagonal_minus_sum_log_p_values_{response_label}.parquet'
        )
        self.final_heatmap_file = os.path.join(
            self.otu_p_value_tracing_path, f'{self.exp_name}_otu_pairs_tracing_{response_label}.csv'
        )

    def build_otu_p_value_matrix(self, coda_method: str) -> None:
        """
        Creates a table with accumulated OTU-pair-level p-values: first, initiates and stores the initiated matrix in a
        CSV file, then, iterates over every processed OCU clustering steps and accumulates p-values for the OTU pairs
        appearing in the balance; performs that for the provided response.
        """
        coda_method_basic_results = os.path.join(self.compores_basic_results_path, coda_method)
        coda_method_sample_level_compores_results = os.path.join(
            self.balance_results_path, coda_method,
            "sample_level_results",
            self.response_label
        )
        pcc_arrays = load_file('pcc_shuffle_arrays.pkl', coda_method_basic_results)
        ocu_dict = load_file('ocu_dictionary.pkl', coda_method_basic_results)
        list_of_ocus = [int(ocu_key.split(' OCU')[0]) for ocu_key in ocu_dict.keys()]
        # Sort from high to low
        ocus = sorted(list_of_ocus, reverse=True)
        for ocu in ocus:
            correlation_file = os.path.join(
                coda_method_sample_level_compores_results,
                f"{self.exp_name}-ocu_{ocu}-{self.response_label}-clr_correlation_values.csv")
            if coda_method == 'pairs':
                correlation_file = os.path.join(
                    coda_method_sample_level_compores_results,
                    f"{self.exp_name}-ocu_{ocu}-{self.response_label}-pairs_correlation_values.csv")

            try:
                corr_matrix = pd.read_csv(correlation_file, index_col=0)
            except FileNotFoundError:
                raise FileNotFoundError(f"Unknown CoDA method, {coda_method}, or missing file.")

            pcc_arr = pcc_arrays[self.exp_name][ocu][self.response_index]
            p_value_matrix = corr_matrix.map(lambda x: bootstrap_p_value(x, pcc_arr))

            partial_ocu_dictionary = ocu_dict[f"{ocu} OCUs"]['OCUs']
            if coda_method == 'pairs':
                self.update_all_pairs_sum_p_values(p_value_matrix, partial_ocu_dictionary)
            elif coda_method == 'CLR':
                self.update_all_diagonal_sum_p_values(p_value_matrix, partial_ocu_dictionary)
            else:
                raise ValueError(f"Unknown CoDA method: {coda_method}.")

    def update_all_diagonal_sum_p_values(
            self, current_p_val_table: pd.DataFrame, ocu_dict: dict[str, dict], pairs_like: bool = True
    ) -> None:
        """
        This function goes over each file in the directory, extracts the OTUs related to each OCU
        from each table, and adds the value in that cell to the diagonal array item for the corresponding taxa.
        Updates diagonal DataFrame with added values based on current_p_val_table.

        :param current_p_val_table: DataFrame that holds the pairwise data p_values to update the all_pairs table with
        :param ocu_dict: relevant sub-dictionary of the dictionary that maps OCUs to OTUs at various clustering stages
        :param pairs_like: if calculated pairs-like (accounting for both nominator and denominator) or not
        :return: None
        """

        diagonal_items = pd.read_parquet(self.diagonal_file)
        # Make sure the index and columns are strings (should be the case, but fails sometimes)
        diagonal_items.index = diagonal_items.index.astype(str)
        diagonal_items.columns = diagonal_items.columns.astype(str)

        # Iterate through each row of the table
        table = current_p_val_table.copy()
        for num_ocu, value in table.itertuples(index=True, name=None):
            # Fetch the list of OTUs for num_ocu and den_ocu using the mapping function
            num_otus = ocu_dict[num_ocu]['taxa']
            # Iterate through all pairs of OTUs in num_otus and den_otus
            for num_taxa in num_otus:
                # Update the diagonal items at the position [num_taxa]
                if num_taxa in diagonal_items.index:
                    diagonal_items.at[num_taxa, diagonal_items.columns[0]] -= np.log(value)

            if pairs_like:
                # Add the same value to all diagonal items to account for the CLR balance denominator similarly to SLR
                diagonal_items = diagonal_items.map(lambda x: x - np.log(value))

        diagonal_items.to_parquet(self.diagonal_file, engine='fastparquet')

        self.update_otu_cumulative_p_value_analysis_state('CLR', True)

    def update_all_pairs_sum_p_values(self, current_p_val_table: pd.DataFrame, ocu_dict: dict[str, dict]) -> None:
        """
        This function goes over each file in the directory, extracts the OTUs related to each OCU pair
        from each table, and adds the value in that cell to the all_pairs table for the corresponding taxa.
        Updates all_pairs DataFrame with added values based on current_p_val_table.

        :param current_p_val_table: DataFrame that holds the pairwise data p_values to update the all_pairs table with
        :param ocu_dict: relevant sub-dictionary of the dictionary that maps OCUs to OTUs at various clustering stages
        :return: None
        """

        all_pairs = pd.read_parquet(self.all_pairs_file)
        # Make sure the index and columns are strings (should be the case, but fails sometimes)
        all_pairs.index = all_pairs.index.astype(str)
        all_pairs.columns = all_pairs.columns.astype(str)

        # Iterate through each row of the table
        table = current_p_val_table.copy()
        for (num_ocu, den_ocu), value in table.stack().items():
            # Fetch the list of OTUs for num_ocu and den_ocu using the mapping function
            num_otus = ocu_dict[num_ocu]['taxa']
            den_otus = ocu_dict[den_ocu]['taxa']
            # Iterate through all pairs of OTUs in num_otus and den_otus
            for num_taxa in num_otus:
                for den_taxa in den_otus:
                    # Update the all_pairs DataFrame at the position [num_taxa, den_taxa]
                    if num_taxa in all_pairs.columns and den_taxa in all_pairs.index:
                        all_pairs.at[num_taxa, den_taxa] -= np.log(value)

        all_pairs.to_parquet(self.all_pairs_file, engine='fastparquet')

    def prepare_final_otu_pair_p_value_matrix(self, scaled: bool = False) -> pd.DataFrame:
        """
        For the case SLR ('pairs') is requested as a CoDA method via the config file, the accumulated log p-value
        matrix for OTU pairs can be produced. In this case, the OTU pair-wise table is initiated at the previous step
        `build_otu_p_value_matrix` function, however it should be transformed to be symmetric, since the OCU pair
        correlation matrices it is constructed from only take positive balance options. In addition, the final version
        of the matrix will have the diagonal filled with the log p-values of CLR balance correlations, also accumulated
        over OTUs.
        """
        all_pairs = pd.read_parquet(self.all_pairs_file)
        # Make sure the index and columns are strings (should be the case, but fails sometimes)
        all_pairs.index = all_pairs.index.astype(str)
        all_pairs.columns = all_pairs.columns.astype(str)
        # Since pair balances are always taken with positive sign, there is always the opposite pair value that is empty
        # (A/B >0 -> B/A = 0)
        all_pairs += all_pairs.T
        if scaled:
            all_pairs = pd.DataFrame(
                StandardScaler().fit_transform(all_pairs), index=all_pairs.index, columns=all_pairs.columns
            )

        # After updating the matrix, we handle the diagonal
        diagonal_items = pd.read_parquet(self.diagonal_file)
        # Make sure the index is a string (should be the case, but fails sometimes)
        diagonal_items.index = diagonal_items.index.astype(str)
        diagonal_items.columns = diagonal_items.columns.astype(str)
        if scaled:
            diagonal_items = pd.DataFrame(
                StandardScaler().fit_transform(diagonal_items),
                index=diagonal_items.index, columns=diagonal_items.columns
            )
        # Substitute the all_pairs DataFrame diagonal values with the diagonal_items values
        for label in diagonal_items.index:
            all_pairs.at[label, label] = diagonal_items.at[label, diagonal_items.columns[0]]

        all_pairs.to_csv(self.final_heatmap_file)
        return all_pairs
