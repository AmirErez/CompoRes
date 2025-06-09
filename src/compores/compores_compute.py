import os
import sys

import numpy as np
import pandas as pd

from .logger_module import CompoResLogger
from .utils import is_considered_imputed_sample, load_file

COMPORES_LOGGER = CompoResLogger(log_name=__name__, filemode='a')


class CompoRes:
    def __init__(
            self,
            x: pd.DataFrame,
            y: pd.Series,
            total_ocu_number: int,
            logs_path: str,
            ocu_dictionary_path: str,
            balance_method: str,
            corr_type: str = 'pearson',
            shuffle_flag: bool = False,
            regular_run_flag: bool = True
    ):
        """
        CompoRes is a class designed for robust search for statistically significant correlation between microbiome data
        and target variables (treatment or response). At its core, it is based on compositional analysis of the
        microbiome data. It employs the 'balances' method to identify the primary balance of OCU variables that exhibit
        the highest correlation with the target variable. By default, the balance method is applied following the CLR
        transformation of the clustered microbiome data. Additionally, the SLR ('pairs' or 'pairwise') transformation
        can be specified via the config file. This methodology is based on the work of Rivera-Pinto et al., as detailed
        in their publication from 2018, "Balances: A New Perspective for Microbiome Analysis",
        https://msystems.asm.org/content/3/4/e00053-18.

        :param x: initial microbiome data matrix; should be normalized proportions, can contain zero rows and be sparse.
        :param y: response (target) variable.
        :param total_ocu_number: the number of OCUs under processing.
        :param balance_method: method of balance calculations, currently supported `pairs` or `CLR`.
        :param corr_type: type of correlation applied, currently supported `pearson` or `spearman`.
        :param shuffle_flag: a flag to differentiate between a regular run and a shuffled run.
        """
        self.log_files_path = logs_path
        self.total_ocu_number = total_ocu_number
        self.clustered_ocu_dictionary_path = ocu_dictionary_path
        self.shuffle_flag = shuffle_flag
        self.regular_run_flag = regular_run_flag
        self.logger_instance = COMPORES_LOGGER
        self.logger = self.logger_instance.get_logger()
        self.x = x
        self.x.index.astype(str)

        # customize the logger name to channel the detailed log for the current number of OCUs
        self._update_logger_file_name(os.path.join(self.log_files_path, f'compores_compute_{self.x.shape[1]}_OCUs.log'))

        try:
            self.check_if_normalized_input_matrix(x)
        except ValueError as e:
            self.logger.error(f"Error creating CompoRes object: {e}")
            self.cleanup_logger_file_handlers()
            sys.exit(1)
        if balance_method not in ['pairs', 'CLR']:
            error_msg = f"Invalid method '{balance_method}'. Supported methods: 'pairs', 'CLR'."
            self.logger.error(f"Error creating CompoRes object: {error_msg}")
            self.cleanup_logger_file_handlers()
            sys.exit(1)
        if corr_type not in ['pearson', 'spearman']:
            error_msg = f"Invalid correlation type '{corr_type}'. Supported types: 'pearson', 'spearman'."
            self.logger.error(f"Error creating CompoRes object: {error_msg}")
            self.cleanup_logger_file_handlers()
            sys.exit(1)
        self.y = y
        self.y.index.astype(str)

        self.method = balance_method
        self.corr = corr_type
        self.corr_matrix = pd.DataFrame()
        self.final_dictionary = pd.DataFrame()

    def _update_logger_file_name(self, path_to_logger: str) -> None:
        """
        Update the logger file name with the provided path.
        :param path_to_logger: Path to the new log file.
        :return: None
        """
        self.logger_instance.update_logger_file_handler(path_to_logger)

    def cleanup_logger_file_handlers(self) -> None:
        """
        Close all file handlers associated with the logger instance.
        :return: None
        """
        self.logger_instance.cleanup_logger_file_handlers()

    def get_results(self) -> pd.DataFrame:
        """
        Returns the final dictionary with the results of the compositional analysis.
        :return: Final dictionary with the results of the compositional analysis.
        """
        return self.final_dictionary

    def get_correlation_matrix(self) -> pd.DataFrame:
        """
        Returns the correlation matrix of the balances with the target variable.
        :return: Correlation matrix of the balances with the target variable, 2D for the 'pairs' method, 1D for 'CLR'.
        """
        return self.corr_matrix

    def run_analysis(self) -> None:
        """
        Runs the compositional analysis on the provided data. The method performs zero-replacement, log-transformation,
        and balance calculations based on the selected method and correlation type. The final dictionary with the
        results is constructed and assigned to the class.

        :return: None
        """
        num_y = self.y.copy()
        logc = np.log(self.x)

        if self.method == 'pairs':
            msg = "Starting CompoRes function with the 'pairs' method."
        else:
            msg = "Starting CompoRes function with the 'CLR' method."
        if self.regular_run_flag:
            if self.shuffle_flag:
                self.logger.debug("Running shuffled CompoRes instance")
            else:
                self.logger.debug(f"Running regular CompoRes instance for {num_y.name}")
                self.logger.debug(msg)

        if self.method == 'pairs':
            Tab_var, B = self.compute_balances(logc, num_y)
        else:
            Tab_var, B = self.compute_clr(logc, num_y)

        self.cleanup_logger_file_handlers()

        self.construct_final_dictionary(Tab_var, B)

        # Release memory
        self.x = None
        self.y = None
        del logc, num_y

    def check_if_normalized_input_matrix(self, x: pd.DataFrame()) -> None:
        """
        Checks whether the input matrix is normalized proportions and raises an error if not.
        :param x: Input data matrix, e.g., microbiome.
        :return: None.
        """

        input_row_sums = np.nansum(x.to_numpy(), axis=1)
        individual_sums_are_one = np.allclose(input_row_sums, 1.0)
        if not individual_sums_are_one and not self.shuffle_flag:
            error_msg = "Not all rows sum up to one, does the input matrix contain normalized proportions?"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def compute_balances(self, log_counts: pd.DataFrame, target_variable: pd.Series) -> tuple[dict, np.ndarray]:
        """Applies SLR to treat the constant-sum constraint of compositional and finds taxa amalgamations forming the
        pair balance with maximum correlation with the target (response).

        Sources: Greenacre, M. Compositional Data Analysis, 2021, Annual Review of Statistics and Its Application,
        Volume 8, PP 271-299, https://doi.org/10.1146/annurev-statistics-042720-124436;
        Rivera-Pinto et al. 2018 Balances: a new perspective for microbiome analysis.
        https://msystems.asm.org/content/3/4/e00053-18

        :param log_counts: preprocessed microbiome DataFrame containing log-transformed counts.
        :param target_variable: Target (response) variable.
        :return: a tuple of a dataframe with derived taxa and balance vector for the maximum correlation balance.
        """

        # Number and name of variables
        num_variables = log_counts.shape[1]
        variable_names = log_counts.columns

        # Initiate the output matrix
        balance_correlations_matrix = pd.DataFrame(
            np.full((num_variables, num_variables), np.nan),
            columns=variable_names,
            index=variable_names
        )

        def compute_correlation_coefficients(
                x_array: np.array, y_array: np.array, compute_method: str = self.corr
        ) -> np.array:
            """
            Computes the correlation coefficients between two arrays.

            :param x_array: First array, a 1-D or 2-D numpy matrix.
            :param y_array: Second array, a 1-D numpy array.
            :param compute_method: Type of correlation problem, e.g. pearson or spearman.
            :return: Resulting correlation coefficients for variables (columns) in x_array and the variable in y_array.
            """
            # If x_array 1D, convert to 2D with a single column and make sure the arrays are memory contiguous
            x_array_m = np.ascontiguousarray(x_array[:, None] if x_array.ndim == 1 else x_array)
            y_array_m = np.ascontiguousarray(y_array.copy())

            # Check if the shapes are compatible
            if x_array_m.shape[0] != y_array.shape[0]:
                raise ValueError("Input array shapes are not compatible.")

            if compute_method == 'spearman':
                x_array_m = np.apply_along_axis(lambda x: np.argsort(np.argsort(x)), axis=0, arr=x_array_m)
                y_array_m = np.argsort(np.argsort(y_array_m))

            # Calculate Pearson correlation for the both methods
            x_dev = x_array_m - np.mean(x_array_m, axis=0)
            y_dev = y_array_m - np.mean(y_array_m)
            correlation_coefficients = []
            for n in range(x_dev.shape[1]):
                col_dev = x_dev[:, n]
                ssx = np.sum(col_dev ** 2)
                ssy = np.sum(y_dev ** 2)
                correlation_coefficients.append(np.sum(col_dev * y_dev) / np.sqrt(ssx * ssy))

            correlation_coefficients = np.array(correlation_coefficients)
            if not self.shuffle_flag and self.regular_run_flag:
                self.logger.debug(f"{compute_method} correlation coefficients computed.")
            return correlation_coefficients

        # Compute correlations between the balances and the target variable
        i, j = np.tril_indices(num_variables, k=-1)
        balance = log_counts.iloc[:, i].values - log_counts.iloc[:, j].values
        correlations = compute_correlation_coefficients(balance, target_variable.values)

        # Fill the output matrix with the absolute values of the correlations
        positive_mask = correlations > 0
        balance_correlations_matrix.values[i[positive_mask], j[positive_mask]] = np.abs(correlations[positive_mask])
        balance_correlations_matrix.values[j[~positive_mask], i[~positive_mask]] = np.abs(correlations[~positive_mask])

        # save the output matrix for later use
        self.corr_matrix = balance_correlations_matrix

        # Find the maximum value in the resulting matrix
        max_mask = balance_correlations_matrix == balance_correlations_matrix.max().max()
        first_balance_column_names = balance_correlations_matrix[max_mask].stack().index.tolist()[0]

        first_balance_row_name = first_balance_column_names[0]
        first_balance_col_name = first_balance_column_names[1]

        POS = first_balance_row_name
        NEG = first_balance_col_name
        Tab_var = {"NUM": POS, "DEN": NEG}

        S1 = log_counts[POS]
        S2 = log_counts[NEG]
        B = np.sqrt(1 / 2) * (S1 - S2)

        return Tab_var, B

    def compute_clr(self, log_counts: pd.DataFrame, target_variable: pd.Series) -> tuple[dict, np.ndarray]:
        """
        Applies CLR to treat the constant-sum constraint of compositional data and find the taxa amalgamation with the
        maximum correlation with given target variable (response).

        Sources: Greenacre, M. Compositional Data Analysis, 2021, Annual Review of Statistics and Its Application,
        Volume 8, PP 271-299, https://doi.org/10.1146/annurev-statistics-042720-124436.

        :param log_counts: preprocessed microbiome DataFrame containing log-transformed counts.
        :param target_variable: target (response) variable.
        :return: a tuple of a dataframe with derived taxon and balance vector for the maximum correlation balance.
        """

        # Compute clr transformation
        log_counts_array = log_counts.to_numpy()
        geometric_mean = np.exp(np.mean(log_counts_array, axis=1))
        clr_transformed = np.log(np.exp(log_counts) / geometric_mean[:, np.newaxis])
        # Calculate correlations with target_variable
        correlations = np.abs(clr_transformed.corrwith(target_variable, method=self.corr))
        # save the output matrix for later use
        self.corr_matrix = correlations
        # Find column name with maximum correlation
        max_ind = correlations.idxmax()

        Tab_var = {"NUM": max_ind, "DEN": "CLR"}

        return Tab_var, clr_transformed.loc[:, max_ind].to_numpy()

    def map_imputed_samples_to_ocu_log_ratio(self, num_str: str, den_str: str, sample_list: list) -> list:
        clustered_ocu_dictionary = load_file('ocu_dictionary.pkl', self.clustered_ocu_dictionary_path)
        is_imputed = []
        for i in range(len(sample_list)):
            is_imputed.append(is_considered_imputed_sample(
                sample_list[i], num_str, den_str, clustered_ocu_dictionary, self.total_ocu_number
            ))
        return is_imputed

    def map_taxa_to_ocu_response_log_ratio_combination(
            self, num_str: str, den_str: str, sample_list: list
    ) -> tuple[list, list]:

        clustered_ocu_dictionary = load_file('ocu_dictionary.pkl', self.clustered_ocu_dictionary_path)

        num_taxa = []
        den_taxa = []

        i_k = f'{self.total_ocu_number} OCUs'

        for i in range(len(sample_list)):
            num_taxa.append(clustered_ocu_dictionary[i_k]['OCUs'][num_str]['taxa'])
            if den_str != 'CLR':
                den_taxa.append(clustered_ocu_dictionary[i_k]['OCUs'][den_str]['taxa'])
            else:
                den_taxa.append('CLR')

        return num_taxa, den_taxa

    def construct_final_dictionary(self, bal_dictionary: dict, final_bal: np.array) -> None:
        """Constructs the final dictionary with the results of the compositional analysis and assigns it to the class.
        :param bal_dictionary: The dictionary with the maximum correlation balance OCU information.
        :param final_bal: The balance vector for the maximum correlation balance.
        :return: None
        """
        # Variables in the NUMERATOR and the DENOMINATOR
        NUM = bal_dictionary["NUM"]
        DEN = bal_dictionary["DEN"]
        samples = self.x.index.values.tolist()
        imputed = self.map_imputed_samples_to_ocu_log_ratio(NUM, DEN, samples)
        # taxa_lists = self.map_taxa_to_ocu_response_log_ratio_combination(NUM, DEN, samples)

        df_dict = {
            'Sample': samples,
            'NUM_OCU': np.full(len(samples), NUM),
            'DEN_OCU': np.full(len(samples), DEN),
            # 'NUM_Taxa_List': taxa_lists[0],
            # 'DEN_Taxa_List': taxa_lists[1],
            'Final_LR_Value': final_bal,
            'Response': np.array(self.y),
            'Is_Imputed': imputed
        }
        self.final_dictionary = pd.DataFrame.from_dict(df_dict)
