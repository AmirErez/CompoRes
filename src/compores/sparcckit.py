from logging import Logger

import numpy as np
import pandas as pd
from numba import njit


class SparccKit:
    """The SparccKit class implements SparCC algorithm for estimating linear correlations in compositional data. SparCC
    infers correlations by analyzing log-ratio variances, accounting for the constant-sum constraint of compositional
    data. It assumes most component pairs are weakly correlated (sparsity), iteratively identifies and excludes the most
    strongly correlated pairs, and refines the correlation estimates until a stopping criterion is met. The result is a
    correlation matrix suitable for compositional datasets. The resulting correlation structure is used downstream for
    hierarchical clustering of given OTUs into OCU groups. Unlike the original SparCC algorithm, the current
    implementation assumes that the user-provided microbiome abundance matrix correctly represents the true
    distributions, thereby obviating the need for absolute counts modeling.
    """

    def __init__(self, logger: Logger, fractions: pd.DataFrame, threshold: float = 0.2, max_iter: int = 100):
        self.logger = logger
        self.original_fractions = fractions.copy()
        self.current_fractions = fractions.copy()
        self.threshold = threshold
        self.max_iter = max_iter
        self.logger.info(f"Number of SparccKit iterations set to {self.max_iter}.")
        self.D_full = fractions.shape[1]
        self.corr = None
        self.connectivity_threshold = 1
        self.original_labels = fractions.columns.to_list()
        self.current_labels = fractions.columns.to_list()  # Current labels of components not excluded
        self.original_indices = list(range(self.D_full))
        self.current_indices = list(range(self.D_full))  # Current indices of components not excluded

    def compute_variation_matrix(self, log_data):
        """Computes the variation matrix T for log-transformed compositional data as described in the SparCC algorithm.
        The variation matrix quantifies the pairwise log-ratio variances between all components, which is the basis for
        inferring correlations in compositional datasets. Each entry T[i, j] represents the variance of
        log(x_i) - log(x_j) across samples.
        """
        log_data = log_data.to_numpy()
        diff = log_data[:, :, None] - log_data[:, None, :]
        T = np.var(diff, axis=0, ddof=1)
        return pd.DataFrame(T, index=self.current_labels, columns=self.current_labels)

    def estimate_component_variations(self, T, exclude_pairs: set = None):
        """Estimates the component-wise log-variance vector as described in the SparCC algorithm. For each component,
        sums the pairwise log-ratio variances (from the variation matrix T), optionally excluding specified component
        pairs identified as the most strongly correlated in previous iterations. This step corresponds to equation
        (12) in the SparCC workflow, providing the input for basis variance estimation.
        """
        mask = pd.DataFrame(np.ones_like(T, dtype=bool), index=T.columns, columns=T.columns)
        if exclude_pairs:
            for i, j in exclude_pairs:
                if i in self.current_indices and j in self.current_indices:
                    mask.loc[self.original_labels[i], self.original_labels[j]] = False
                    mask.loc[self.original_labels[j], self.original_labels[i]] = False
            mask = mask.values
            np.fill_diagonal(mask, False)

        t = (T * mask).sum(axis=1).values
        return pd.DataFrame(t, index=T.index, columns=['variances'])

    def estimate_basis_variances(self, component_vars, exclude_pairs: set = None):
        """Estimates the basis variances omega^2 for each component by solving the linear system A.omega^2 = t.
        This step corresponds to equation (13) in the SParCC workflow. Matrix A is constructed to encode
        connectivity and sets diagonal entries to the number of non-excluded links per component. Excluded
        correlation pairs are zeroed in A as per the iterative exclusion step. Then the equation system is solved
        for omega^2 using a linear solver, with pseudo-inverse fallback for singular systems.
        """
        # Initialize A with ones off-diagonal, will fix excluded_pairs next
        D = component_vars.shape[0]
        A = np.ones((D, D)) - np.eye(D)  # 1s off-diagonal, 0s on diagonal
        A = pd.DataFrame(A, index=self.current_labels, columns=self.current_labels)
        # Zero out excluded pairs (and symmetric ones)
        if exclude_pairs is not None and len(exclude_pairs) > 0:
            for i, j in exclude_pairs:
                if i in self.current_indices and j in self.current_indices:
                    A.loc[self.original_labels[i], self.original_labels[j]] = 0
                    A.loc[self.original_labels[j], self.original_labels[i]] = 0
        # Diagonal A[i, i] should be number of non-excluded links for i
        A = A.values
        A[np.diag_indices(D)] = np.sum(A, axis=1)
        try:
            omega2 = np.linalg.solve(A, component_vars)
        except np.linalg.LinAlgError:
            self.logger.info(
                "Warning: LinAlgError in solving for omega2 after regularization. Trying pseudo-inverse."
            )
            omega2 = np.linalg.pinv(A) @ component_vars  # Fallback

        return np.clip(omega2, 0, np.inf)

    @staticmethod
    @njit
    def compute_basis_correlations(T, omega2):
        """Computes the SparCC basis correlation matrix R from the variation matrix T and estimated basis variances
        omega^2. For each component pair (i, j), calculates the correlation as described in equation (9) of the
        SparCC workflow. Values are clipped to [-1, 1]. This step reconstructs the compositional correlation
        structure after basis variance estimation.
        """
        D = len(omega2)
        R = np.eye(D)
        for i in range(D):
            for j in range(i + 1, D):
                denominator = 2 * np.sqrt(omega2[i] * omega2[j])
                if denominator < 1e-6:
                    value = 0.0
                else:
                    numerator = omega2[i] + omega2[j] - T[i, j]
                    value = (numerator / denominator).item()
                    if value > 1.0:
                        value = 1.0
                    elif value < -1.0:
                        value = -1.0
                R[i, j] = value
                R[j, i] = value
        return R

    def run(self):
        """Executes the iterative SparCC workflow as described by Friedman and Alm, estimating linear correlations in
        compositional data.
        The method:
          1. Computes the initial variation matrix and basis variances from log-transformed abundances.
          2. Iteratively identifies and excludes the most strongly correlated component pairs,
          updating the set of excluded pairs.
          3. After each exclusion, recalculates the variation matrix, component variances,
          and correlation matrix for the reduced set.
          4. Excludes components that become fully connected (i.e., all their correlations are excluded).
          5. Terminates when no strong correlations remain or too few components are left.
        The final correlation matrix reflects the inferred sparse correlation structure, suitable for downstream
        compositional analysis.
        """
        excluded_pairs = set()
        excluded_pair_correlations = set()
        excluded_components = set()

        # Basic SparCC steps (run once)
        T = self.compute_variation_matrix(np.log(self.current_fractions))
        comp_vars = self.estimate_component_variations(T)
        omega2 = self.estimate_basis_variances(comp_vars)  # eq 13
        corr = self.compute_basis_correlations(T.values, omega2)  # eq9
        corr = pd.DataFrame(corr, index=self.original_labels, columns=self.original_labels)
        self.corr = corr.copy()

        for iteration in range(1, self.max_iter + 1):
            # Identify most strongly correlated pair not yet excluded
            max_corr = 0
            new_pair = None
            new_pair_correlation = None
            for i in range(self.D_full):
                if i in excluded_components:
                    continue
                for j in range(i + 1, self.D_full):
                    if j in excluded_components or (i, j) in excluded_pairs or (j, i) in excluded_pairs:
                        continue
                    corr_val = corr.loc[self.original_labels[i], self.original_labels[j]]
                    if abs(corr_val) > max_corr:
                        max_corr = abs(corr_val)
                        new_pair = (i, j)
                        new_pair_correlation = corr_val

            if new_pair is None or max_corr < self.threshold:
                self.logger.info(f"Terminating at iteration {iteration}: no new strong correlations found.")
                break

            excluded_pairs.add(new_pair)
            excluded_pair_correlations.add((new_pair, new_pair_correlation))
            self.logger.debug(
                f"Iteration {iteration}, max correlation: {max_corr:.4f}, new pair: {new_pair}"
            )

            # Identify and exclude fully connected components
            new_excluded_components = set()
            for i in range(self.D_full):
                if i in excluded_components:
                    continue
                connected = [j for j in range(self.D_full)
                             if j != i and ((i, j) in excluded_pairs or (j, i) in excluded_pairs)]
                # if len(connected) >= D - len(excluded_components) - 1:
                if len(connected) >= self.D_full * self.connectivity_threshold:
                    new_excluded_components.add(i)
                    self.logger.debug(f"Component {i} is fully connected with {len(connected)} pairs, excluding it.")
                    for j in range(self.D_full):
                        if j != i and j not in excluded_components:
                            excluded_pairs.add((i, j))
                            excluded_pair_correlations.add(
                                ((i, j), corr.loc[self.original_labels[i], self.original_labels[j]])
                            )

            if new_excluded_components:
                self.logger.debug(f"New excluded components: {new_excluded_components}")
                excluded_components.update(new_excluded_components)
                if self.D_full - len(excluded_components) <= 3:
                    self.logger.info(f"Terminating at iteration {iteration}: fewer than 3 components remaining.")
                    break

                self.current_indices = [idx for idx in self.original_indices if idx not in excluded_components]
                current_fractions = self.original_fractions.copy().iloc[:, self.current_indices]
                self.current_fractions = current_fractions.div(current_fractions.sum(axis=1), axis=0)

            # Re-compute variation matrix T and component variations
            self.current_labels = self.current_fractions.columns.to_list()
            T = self.compute_variation_matrix(np.log(self.current_fractions))
            self.logger.debug(
                f"Re-computed variation matrix T at iteration {iteration}, shape: {T.shape}"
            )
            self.logger.debug(f"Current labels: {self.current_labels}")

            comp_vars = self.estimate_component_variations(T, exclude_pairs=excluded_pairs)
            omega2 = self.estimate_basis_variances(comp_vars, exclude_pairs=excluded_pairs)
            corr = self.compute_basis_correlations(T.values, omega2)
            corr = pd.DataFrame(corr, index=self.current_labels, columns=self.current_labels)

            self.logger.info(f"SparccKit iteration {iteration}:"
                             f" excluded correlation pair indices {new_pair},"
                             f" correlation value = {max_corr:.4f}")

        # Construct final correlation matrix
        for (i, j), value in excluded_pair_correlations:
            self.corr.at[self.original_labels[i], self.original_labels[j]] = value
            self.corr.at[self.original_labels[j], self.original_labels[i]] = value
        for idx in corr.index:
            for lbl in corr.columns:
                self.corr.at[idx, lbl] = corr.at[idx, lbl]

    def get_corr_file(self):
        return pd.DataFrame(self.corr)
