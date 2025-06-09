import logging
import os
import shutil

import pytest
import pandas as pd
from src.compores.preprocessing import Preprocessor


class TestCheckInputFilesForSameIndexes:

    @pytest.fixture(scope="function")
    def logger_mock(self):
        logger = logging.getLogger(__name__)

        return logger

    @pytest.fixture(scope="function")
    def setup_teardown_for_same_sample_sets(self):
        # Create sample dataframes for testing
        df1 = pd.DataFrame({
            "Index": ['sample4', 'sample2', 'sample1', 'sample3'],
            "Data1": [1, 2, 3, 4],
            "Data2": [4, 5, 6, 7],
            "Data3": [7, 8, 9, 10]
        })
        df2 = pd.DataFrame(
            {"Index": ['sample2', 'sample1', 'sample3', 'sample4'], "Data": [8, 6, 4, 2]}
        )

        # Save dataframes to temporary files
        os.makedirs(os.path.join(os.getcwd(), 'input', 'microbiome'))
        os.makedirs(os.path.join(os.getcwd(), 'input', 'response'))
        file1_path = os.path.join(os.getcwd(), 'input', 'microbiome/file1.tsv')
        file2_path = os.path.join(os.getcwd(), 'input', 'response/file2.tsv')
        df1.to_csv(file1_path, sep="\t", index=False)
        df2.to_csv(file2_path, sep="\t", index=False)

        expected_df1 = df1.set_index("Index")
        expected_df2 = df2.set_index("Index")
        expected_df1.sort_index(inplace=True)
        expected_df2.sort_index(inplace=True)

        yield file1_path, file2_path, expected_df1, expected_df2

        # Cleanup: Remove the to be created output directory and its content
        shutil.rmtree("input")
        shutil.rmtree("preprocessed_samples")

    @pytest.fixture(scope="function")
    def setup_teardown_for_different_sample_sets(self, tmp_path):
        # TODO: add a test to test the case when two few columns and samples are left in the processed files
        # Create sample dataframes for testing
        df1 = pd.DataFrame({
            "Index": ['sample1', 'sample2', 'sample3', 'sample4'],
            "Data1": [1, 2, 3, 4],
            "Data2": [4, 5, 6, 7],
            "Data3": [7, 8, 9, 10]
        })
        df2 = pd.DataFrame({"Index": ['sample1', 'sample3', 'sample4', 'sample5'], "Data": [8, 6, 4, 2]})

        expected_df1 = pd.DataFrame({
            "Index": ['sample1', 'sample3', 'sample4'], "Data1": [1, 3, 4], "Data2": [4, 6, 7], "Data3": [7, 9, 10]
        }).set_index("Index")
        expected_df2 = pd.DataFrame({
            "Index": ['sample1', 'sample3', 'sample4'], "Data": [8, 6, 4]
        }).set_index("Index")

        # Save dataframes to temporary files
        os.makedirs(os.path.join(os.getcwd(), 'input', 'microbiome'))
        os.makedirs(os.path.join(os.getcwd(), 'input', 'response'))
        file1_path = os.path.join(os.getcwd(), 'input', 'microbiome/file1.tsv')
        file2_path = os.path.join(os.getcwd(), 'input', 'response/file2.tsv')
        df1.to_csv(file1_path, sep='\t', index=False)
        df2.to_csv(file2_path, sep='\t', index=False)

        yield file1_path, file2_path, expected_df1, expected_df2

        # Cleanup: Remove the to be created output directory and its content
        shutil.rmtree("input")
        shutil.rmtree("preprocessed_samples")

    def test_files_with_same_indexes(self, setup_teardown_for_same_sample_sets, logger_mock):
        file1_path, file2_path, expected_df1, expected_df2 = setup_teardown_for_same_sample_sets
        logger = logger_mock
        preprocess = Preprocessor(
            logger, '', '', '', file1_path, file2_path, '', '', '', '', '', '', '','', 1
        )
        preprocess.check_input_files_for_same_indexes(file1_path, file2_path, 'raw')

        actual_df1 = pd.read_csv(os.path.join(os.getcwd(), 'preprocessed_samples/microbiome/file1.tsv'), sep='\t', index_col=0)
        actual_df2 = pd.read_csv(os.path.join(os.getcwd(), 'preprocessed_samples/response/file2.tsv'), sep='\t', index_col=0)

        assert actual_df1.equals(expected_df1)
        assert actual_df2.equals(expected_df2)

    def test_files_with_different_indexes(self, setup_teardown_for_different_sample_sets, logger_mock):
        file1_path, file2_path, expected_df1, expected_df2 = setup_teardown_for_different_sample_sets
        logger = logger_mock
        preprocess = Preprocessor(
            logger, '', '', '', file1_path, file2_path, '', '', '', '', '', '', '','', 1
        )
        preprocess.check_input_files_for_same_indexes(file1_path, file2_path, 'raw')

        actual_df1 = pd.read_csv(os.path.join(os.getcwd(), 'preprocessed_samples/microbiome/file1.tsv'), sep='\t', index_col=0)
        actual_df2 = pd.read_csv(os.path.join(os.getcwd(), 'preprocessed_samples/response/file2.tsv'), sep='\t', index_col=0)

        assert actual_df1.equals(expected_df1)
        assert actual_df2.equals(expected_df2)
