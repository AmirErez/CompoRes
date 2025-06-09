import os

import numpy as np
import pandas as pd
import seaborn as sns

from src.compores.compores_plotting import delete_older_shuffle_files, extract_heatmap_clustering_information


class TestComporesPlotting:
    # Files with fewer shuffles than current_shuffles are deleted
    def test_delete_older_shuffle_files(self, tmp_path):
        # Create a temporary directory
        tmp_dir = tmp_path / "shuffle_files"
        tmp_dir.mkdir()

        # Create shuffle files with fewer shuffles than current_shuffles
        shuffle_files = [
            "pl_5_shuffles.tsv",
            "plot_7_shuffles.png",
            "plot_10_shuffles.png",
            "plot_15.png", # files with no shuffles word in the name are not considered
        ]

        for shuffle_file in shuffle_files:
            with open(tmp_dir / shuffle_file, "w") as f:
                f.write("")

        # Call delete_older_shuffle_files
        current_shuffles = 8
        delete_older_shuffle_files(tmp_dir, current_shuffles)

        # Check that shuffle files with fewer shuffles are deleted and the rest are kept
        assert len(os.listdir(tmp_dir)) == 2
        assert "pl_5_shuffles.tsv" not in os.listdir(tmp_dir)
        assert "plot_7_shuffles.png" not in os.listdir(tmp_dir)
        assert "plot_10_shuffles.png" in os.listdir(tmp_dir)
        assert "plot_15.png" in os.listdir(tmp_dir)

    # Function correctly processes a valid ClusterGrid object and extracts clustering information
    def test_extract_heatmap_clustering_information_valid_processing(self):

        test_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        test_data_labels = ['A', 'B', 'C']
        test_data_df = pd.DataFrame(test_data, columns=test_data_labels, index=test_data_labels)

        cluster_map = sns.clustermap(
            test_data_df,
            metric="euclidean",
            method="ward",
            cmap="PuBuGn",
            row_cluster=True,  # Cluster rows
            col_cluster=True,  # Cluster columns
            xticklabels=False,  # Hide x-axis labels
            yticklabels=True,  # Show y-axis labels
            figsize=(12, 8)
        )

        cluster_info = extract_heatmap_clustering_information(cluster_map, test_data_labels)

        # Assertions
        assert len(cluster_info) == len(test_data) + 2  # 3 initial + 2 merged
        assert cluster_info[0]['cluster 1'] == ['C']
        assert cluster_info[1]['cluster 1'] == ['A']
        assert cluster_info[2]['cluster 1'] == ['B']
        assert cluster_info[3]['cluster 1'] == ['A']
        assert cluster_info[3]['cluster 2'] == ['B']
        assert np.floor(cluster_info[3]['distance']) == 5
        assert cluster_info[3]['number of elements'] == 2
        assert cluster_info[4]['cluster 1'] == ['C']
        assert cluster_info[4]['cluster 2'] == ['A', 'B']
        assert np.floor(cluster_info[4]['distance']) == 9
        assert cluster_info[4]['number of elements'] == 3
