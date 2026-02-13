import os
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .exceptions_module import NoResponseLabelFound
from .preprocessing import PREPROCESSING_RESULTS
from .utils import load_file, gev_p_value, fetch_full_target_response_label


class ComporesClusteredPValueCalculations:
    def __init__(
            self, config_dict: dict, g1: str, g2: str, g3: str,
            init_flag: bool = False, partial_response_tag: str | None = None
    ):
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
        self.response_index = None
        self.response_label = None
        self._prepare_ocu_tracing_folders(init_flag, partial_response_tag)
        self.current_otu_tracing_file = None
        self.current_normalization_matrix = None
        self.current_estimated_p_value_file = None
        self.state = self.load_state()
        self.samples_ocu_list_length = None

    @staticmethod
    def _compute_p_value(x, pcc_arr):
        if np.isnan(x):
            return np.nan
        return gev_p_value(x, pcc_arr, correction_method=None)[0]

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

    def _prepare_ocu_tracing_folders(self, init_flag: bool, response_tag: str | None) -> None:
        if init_flag:

            for balance_method in self.balance_methods:
                path_to_intermediate_results = os.path.join(
                    self.outputs_path, "compores_basic_results", self.exp_name, balance_method
                )
                try:
                    response_labels = self.prepare_response_list_to_trace(path_to_intermediate_results, response_tag)
                except NoResponseLabelFound:
                    raise

                taxa_str = self._read_taxa_labels()

                os.makedirs(os.path.join(self.otu_p_value_tracing_path, balance_method), exist_ok=True)

                data_c = np.zeros((len(taxa_str), 2))
                clr_items = pd.DataFrame(data_c, index=taxa_str, columns=[
                    'minus_mean_log_p_value_positive', 'minus_mean_log_p_value_negative'
                ])
                clr_normalization_count_matrix = pd.DataFrame(data_c, index=taxa_str, columns=[
                    'count_positive', 'count_negative'
                ])
                if balance_method == 'pairs':
                    data_p = np.zeros((len(taxa_str), len(taxa_str)))
                    all_pairs = pd.DataFrame(data_p, index=taxa_str, columns=taxa_str)
                    for response_label in response_labels:

                        path = self._set_response_otu_tracing_path(balance_method, response_label)
                        all_pairs.to_parquet(path, engine="fastparquet")

                        norm_path = self._set_response_otu_tracing_normalization_path(balance_method, response_label)
                        all_pairs.astype(int).to_parquet(norm_path, engine="fastparquet")

                        clr_items.to_parquet(
                            path.replace(
                                f'otu_{balance_method}', f'otu_{balance_method}_condensed'
                            ), engine="fastparquet"
                        )
                        clr_normalization_count_matrix.astype(int).to_parquet(
                            norm_path.replace(
                                f'otu_{balance_method}', f'otu_{balance_method}_condensed'
                            ), engine="fastparquet"
                        )

                elif balance_method == 'CLR':

                    for response_label in response_labels:

                        path = self._set_response_otu_tracing_path(balance_method, response_label)
                        clr_items.to_parquet(path, engine="fastparquet")

                        norm_path = self._set_response_otu_tracing_normalization_path(balance_method, response_label)
                        clr_normalization_count_matrix.astype(int).to_parquet(norm_path, engine="fastparquet")

    def _set_response_otu_tracing_path(self, balance_method, response_label):
        file_name_prefix = f'{self.exp_name}_otu_{balance_method.lower()}_minus_mean_log_p_values_'
        return os.path.join(
            self.otu_p_value_tracing_path,
            balance_method,
            f'{file_name_prefix}{response_label}.parquet'
        )

    def _set_response_otu_tracing_normalization_path(self, balance_method, response_label):
        normalization_file_name_prefix = f'{self.exp_name}_otu_{balance_method.lower()}_normalization_count_'
        return os.path.join(
            self.otu_p_value_tracing_path,
            balance_method,
            f'{normalization_file_name_prefix}{response_label}.parquet'
        )

    def _set_response_otu_tracing_estimated_p_value_path(self, balance_method, response_label):
        estimated_p_value_file_name_prefix = f'{self.exp_name}_otu_{balance_method.lower()}_estimated_p_value_'
        return os.path.join(
            self.otu_p_value_tracing_path,
            balance_method,
            f'{estimated_p_value_file_name_prefix}{response_label}.tsv'
        )

    def prepare_response_list_to_trace(self, path_to_intermediate_results, response_tag):
        if response_tag:
            try:
                response_tag, _ = fetch_full_target_response_label(path_to_intermediate_results, response_tag)
                self.response_label = response_tag
                response_labels = [response_tag]
            except NoResponseLabelFound:
                raise
        else:
            response_labels = load_file("response_index.pkl", path_to_intermediate_results)

        return response_labels

    def _read_taxa_labels(self):
        processed_microbiome = pd.read_csv(
            os.path.join(str(self.path_to_preprocessed_microbiome), f'{self.exp_name}.tsv'),
            index_col=0, sep='\t'
        )
        taxa = processed_microbiome.columns
        taxa_str = [str(t) for t in taxa]
        return taxa_str

    def _read_files(self, condensed: bool = False, condensed_for: str = 'pairs'):
        ocu_tracing_file = pd.read_parquet(self.current_otu_tracing_file)
        normalization_count_matrix = pd.read_parquet(self.current_normalization_matrix)

        if condensed:
            ocu_tracing_file = pd.read_parquet(
                self.current_otu_tracing_file.replace(f'otu_{condensed_for}', f'otu_{condensed_for}_condensed')
            )
            normalization_count_matrix = pd.read_parquet(
                self.current_normalization_matrix.replace(f'otu_{condensed_for}', f'otu_{condensed_for}_condensed')
            )

        # Make sure the index and columns are strings (should be the case, but fails sometimes)
        self._ensure_string_labels_from_parquets([ocu_tracing_file, normalization_count_matrix])
        return ocu_tracing_file, normalization_count_matrix

    @staticmethod
    def _ensure_string_labels_from_parquets(df_list):
        for df in df_list:
            df.index = df.index.astype(str)
            df.columns = df.columns.astype(str)

    def set_current_response(self, response_index: int, response_label: str, balance_method: str) -> None:

        self.response_index = response_index
        self.response_label = response_label

        self.current_otu_tracing_file = self._set_response_otu_tracing_path(balance_method, response_label)
        self.current_normalization_matrix = self._set_response_otu_tracing_normalization_path(
            balance_method, response_label
        )
        self.current_estimated_p_value_file = self._set_response_otu_tracing_estimated_p_value_path(
            balance_method, response_label
        )

    def build_otu_p_value_matrix(self, coda_method: str) -> None:
        """
        Creates a table with accumulated OTU-level p-values: first, initiates and stores the initiated matrix in a
        file, then, iterates over every processed OCU clustering steps and accumulates p-values for CLR OTUs / OTU pairs
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
        self.samples_ocu_list_length = len(ocus)
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

            flat_corr = corr_matrix.values.flatten()
            with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as executor:
                p_values = list(executor.map(self._compute_p_value, flat_corr, [pcc_arr]*len(flat_corr)))

            p_value_matrix = pd.DataFrame(
                np.array(p_values).reshape(corr_matrix.shape),
                index=corr_matrix.index,
                columns=corr_matrix.columns
            )

            partial_ocu_dictionary = ocu_dict[f"{ocu} OCUs"]['OCUs']
            if coda_method == 'pairs':
                self.update_all_pair_items_sum_p_values(p_value_matrix, partial_ocu_dictionary)
            elif coda_method == 'CLR':
                self.update_all_clr_items_sum_p_values(p_value_matrix, corr_matrix, partial_ocu_dictionary)
            else:
                raise ValueError(f"Unknown CoDA method: {coda_method}.")

    def update_all_clr_items_sum_p_values(
            self, current_p_val_table: pd.DataFrame, current_correlation_matrix: pd.DataFrame, ocu_dict: dict[str, dict]
    ) -> None:
        """
        Iterates over CLR per-OCU p-value table, distributes CLR per-OCU p-values to constituent OTUs, while
        accumulating minus-log contributions per taxon. For each non-NaN p-value the implementation:
          - retrieves the OTU list for the OCU from `ocu_dict`;
          - computes the per-OTU contribution as `-log(p_value) / len(num_otus)`;
          - subtracts that contribution from the appropriate column in `clr_items` depending on the
            sign of the correlation in `current_correlation_matrix` (positive -> first column,
            negative -> second column);
          - increments the corresponding entries in the `normalization_count_matrix`.

        Side effects: updates and writes `clr_items` and `normalization_count_matrix` parquet files
        referenced by `self.current_otu_tracing_file` and `self.current_normalization_matrix`.

        :param current_p_val_table: DataFrame of CLR p-values indexed by OCU.
        :param current_correlation_matrix: DataFrame of CLR correlation values used to determine sign.
        :param ocu_dict: Sub-dictionary of OCU identifiers to metadata including `taxa` (OTU identifiers).
        :return: None
        """

        clr_items, normalization_count_matrix = self._read_files()

        # Iterate through each row of the table
        table = current_p_val_table.copy()
        for num_ocu, value in table.itertuples(index=True, name=None):
            if pd.isna(value):
                continue
            # Fetch the list of OTUs for num_ocu using the mapping function
            num_otus = ocu_dict[num_ocu]['taxa']
            # correction coefficient to spread the p-ue information across the OTUs in the OCU
            group_length = len(num_otus)
            # Iterate through all pairs of OTUs in num_otus
            for num_taxa in num_otus:
                # Update the clr items at the position [num_taxa]
                if num_taxa in clr_items.index:
                    if current_correlation_matrix.at[num_ocu, current_correlation_matrix.columns[0]] >= 0:
                        clr_items.at[num_taxa, clr_items.columns[0]] -= np.log(value) / group_length
                        normalization_count_matrix.at[num_taxa, 'count_positive'] += 1
                    else:
                        clr_items.at[num_taxa, clr_items.columns[1]] -= np.log(value) / group_length
                        normalization_count_matrix.at[num_taxa, 'count_negative'] += 1

        clr_items.to_parquet(self.current_otu_tracing_file, engine='fastparquet')
        normalization_count_matrix.to_parquet(self.current_normalization_matrix, engine='fastparquet')

    def update_all_pair_items_sum_p_values(self, current_p_val_table: pd.DataFrame, ocu_dict: dict[str, dict]) -> None:
        """
        Iterates over the stacked pairwise p-value table and distributes each pairwise p-value across all
        combinations of OTUs that compose the numerator and denominator OCUs. For each non-NaN p-value the
        implementation:
          - retrieves OTU lists for the numerator and denominator OCUs from `ocu_dict`;
          - computes the per-OTU-pair contribution as `-log(p_value) / (len(num_otus) * len(den_otus))`;
          - subtracts that contribution from `all_pairs` at each `(num_taxa, den_taxa)` position;
          - increments the corresponding entries in the normalization count matrices;
          - subtracts that contribution from the appropriate column in `all_pairs_condensed` depending on the
            sign of the correlation in the pair for the particular taxon (positive -> first column,
            negative -> second column);
          - increments the corresponding entries in the `normalization_count_matrix_condensed` matrices.
          - updates condensed per-taxon summaries by attributing the broken-pair contributions to each taxon.

        Side effects:updates and writes `all_pairs`, `normalization_count_matrix` and their condensed counterparts to
        the current parquet files referenced by `self.current_otu_tracing_file` and `self.current_normalization_matrix`.

        :param current_p_val_table: Pairwise p-value matrix indexed by (numerator OCU, denominator OCU.
        :param ocu_dict: Sub-dictionary of OCU identifiers to metadata including the list of `taxa` (OTU identifiers).
        :return: None
        """

        all_pairs, normalization_count_matrix = self._read_files()
        all_pairs_condensed, normalization_count_matrix_condensed = self._read_files(condensed=True)

        # Iterate through each row of the table
        table = current_p_val_table.copy()
        for (num_ocu, den_ocu), value in table.stack().items():
            if pd.isna(value):
                continue
            # Fetch the list of OTUs for num_ocu and den_ocu using the mapping function
            num_otus = ocu_dict[num_ocu]['taxa']
            den_otus = ocu_dict[den_ocu]['taxa']
            # correction coefficient to spread the p-value information across all combinations the OTU pairs
            group_length = len(num_otus) * len(den_otus)
            # Iterate through all pairs of OTUs in num_otus and den_otus
            for num_taxa in num_otus:
                for den_taxa in den_otus:
                    # Update the all_pairs DataFrame at the position [num_taxa, den_taxa]
                    if num_taxa in all_pairs.index and den_taxa in all_pairs.columns:
                        all_pairs.at[num_taxa, den_taxa] -= np.log(value) / group_length
                        normalization_count_matrix.at[num_taxa, den_taxa] += 1
                        all_pairs_condensed.at[
                            num_taxa, all_pairs_condensed.columns[0]
                        ] -= np.log(value) / group_length  # account for breaking the pair
                        normalization_count_matrix_condensed.at[num_taxa, 'count_positive'] += 1
                        all_pairs_condensed.at[
                            den_taxa, all_pairs_condensed.columns[1]
                        ] -= np.log(value) / group_length  # account for breaking the pair
                        normalization_count_matrix_condensed.at[den_taxa, 'count_negative'] += 1

        all_pairs.to_parquet(self.current_otu_tracing_file, engine='fastparquet')
        normalization_count_matrix.to_parquet(self.current_normalization_matrix, engine='fastparquet')
        all_pairs_condensed.to_parquet(
            self.current_otu_tracing_file.replace('otu_pairs', 'otu_pairs_condensed'), engine='fastparquet'
        )
        normalization_count_matrix_condensed.to_parquet(
            self.current_normalization_matrix.replace('otu_pairs', 'otu_pairs_condensed'), engine='fastparquet'
        )

    def prepare_final_otu_tracing_output(self) -> None:
        """Cumulative tables are initiated at the previous step by the `build_otu_p_value_matrix` function; here,
        the cumulative OTU-wise minus log p-value is normalized by the cumulative number of counts the corresponding
        OTU (in case of the 'CLR' balance method) or OTU pair (in case of the 'pairs' balance method) has appeared in
        the balances; in the uncondensed 'pairs' case, the final version of the matrix will have the diagonal filled
        with zeroes.
        """

        traces, normalization_count_matrix = self._read_files()

        if traces.columns.equals(traces.index):
            # Correct diagonal to be zeroes
            np.fill_diagonal(traces.values, 0)

            orig_tag = f'otu_{self.balance_methods[-1]}'
            condensed_tag = f'otu_{self.balance_methods[-1]}_condensed'

            traces_condensed, normalization_count_matrix_condensed = self._read_files(condensed=True)
            traces_condensed = self.normalize_df(normalization_count_matrix_condensed, traces_condensed)
            traces_condensed.to_csv(
                self.current_otu_tracing_file.replace(
                    orig_tag, condensed_tag
                ).replace(".parquet", ".tsv"), sep="\t")
            os.remove(self.current_otu_tracing_file.replace(orig_tag, condensed_tag))

            p_values_condensed = traces_condensed.map(lambda x: np.exp(-x))
            p_values_condensed = p_values_condensed.rename(
                columns={
                    p_values_condensed.columns[0]: "p_value_estimate_positive",
                    p_values_condensed.columns[1]: "p_value_estimate_negative",
                }
            )
            p_values_condensed.to_csv(self.current_estimated_p_value_file.replace(
                orig_tag, condensed_tag
            ).replace('.parquet', '.tsv'), sep="\t")

            normalization_count_matrix_condensed.to_csv(
                self.current_normalization_matrix.replace(
                    orig_tag, condensed_tag
                ).replace('.parquet', '.tsv'), sep="\t"
            )
            os.remove(self.current_normalization_matrix.replace(orig_tag, condensed_tag))

        # normalize the number of direct appearances
        traces = self.normalize_df(normalization_count_matrix, traces)

        # Save the final result with the same name, but as TSV, and remove the temporary parquet file
        traces.to_csv(self.current_otu_tracing_file.replace('.parquet', '.tsv'), sep='\t')
        os.remove(self.current_otu_tracing_file)

        # Cast the traces dataframe into p-values
        p_values = traces.map(lambda x: np.exp(-x))
        if not traces.columns.equals(traces.index):
            p_values = p_values.rename(columns={
                p_values.columns[0]: 'p_value_estimate_positive',
                p_values.columns[1]: 'p_value_estimate_negative'
            })
        p_values.to_csv(self.current_estimated_p_value_file, sep='\t')

        normalization_count_matrix.to_csv(self.current_normalization_matrix.replace('.parquet', '.tsv'), sep='\t')
        os.remove(self.current_normalization_matrix)

    @staticmethod
    def normalize_df(normalization_count_df: pd.DataFrame, to_normalize_df: pd.DataFrame):
        mask_df = normalization_count_df != 0
        mask = mask_df.values
        traces_values = to_normalize_df.values
        traces_values[mask] = traces_values[mask] / normalization_count_df.values[mask]
        to_normalize_df.iloc[:, :] = traces_values
        return to_normalize_df
