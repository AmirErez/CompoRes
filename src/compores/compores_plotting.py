import json
import os
import re
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from seaborn.matrix import ClusterGrid
from matplotlib.ticker import LogLocator, ScalarFormatter, NullFormatter

from .utils import load_file, cast_nested_dict_to_array
from matplotlib.lines import Line2D


def plot_ocu_best_balance_by_response(
        path_to_meta_data: str,
        response_num: int,
        response_name: str,
        dir_num: int,
        name: str,
        result: pd.DataFrame,
        path_to_plotted_result: str,
        intercept: float,
        slope: float,
        r_value: float
) -> None:
    """
    Scatter plot of response values and the fit against balance values for the best correlated balance.
    :param path_to_meta_data: the metadata on the experiment samples, which category they belong to
    :param response_num: response series number
    :param response_name: response series name
    :param dir_num: where to save
    :param name: for title (the experiment)
    :param result: the result of the CompoRes algorithm's run
    :param path_to_plotted_result: path to dir to save the results
    :param intercept:
    :param slope:
    :param r_value:
    :return:
    """
    res = result.copy()
    res.set_index('Sample', inplace=True)
    try:
        tags = pd.read_csv(path_to_meta_data, sep='\t', index_col=0)
        tags.index.name = 'SampleID'
        # merge the metadata with the result
        res = res.merge(tags, left_index=True, right_index=True, how='left')
    except FileNotFoundError:
        res['Category'] = 'Uncategorized'

    unique_tags = res['Category'].unique().tolist()

    fig, ax = plt.subplots()
    # plot the scatter plot: first category, then second category
    for tag, color in zip(unique_tags, ['#5da833', '#c4941d'][:len(unique_tags)]):
        ax.scatter(
            res[res['Category'] == tag]['Final_LR_Value'],
            res[res['Category'] == tag]['Response'],
            color=color,
            facecolors=res[res['Category'] == tag]['Is_Imputed'].map(lambda x: 'none' if x else color),
            label=tag
        )
    comment_text = f"Imputed: {res['Is_Imputed'].sum()} / Total: {len(res)}"
    ax.text(0.05, 0.95, comment_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    # plot the linear fit
    ax.plot(res['Final_LR_Value'], intercept + slope * res['Final_LR_Value'], 'k--', label='fitted line')
    fig_title = f"Best correlated log-ratio: {dir_num} OCUs"
    legend_title = f'Microbiome: {name}   Response: {response_name} (#{response_num + 1})'
    ax.set_title(f'{fig_title}\n{legend_title}')
    # add the Pearson correlation value to the plot
    ax.text(
        0.8, 0.2,
        rf"$\rho$ = {round(r_value, 3)}",
        horizontalalignment='left', verticalalignment='top',
        transform=plt.gca().transAxes
    )
    if res['DEN_OCU'].iloc[0] == 'CLR':
        ax.set_xlabel(f"CLR transformed value: {res['NUM_OCU'].iloc[0]}")
    else:
        ax.set_xlabel(f"Transformed value: {res['NUM_OCU'].iloc[0]}  to {res['DEN_OCU'].iloc[0]} pair balance")
    ax.set_ylabel(f"Response value: {response_name}")
    ax.legend(loc='lower right', ncols=2)
    # Add an empty dot for 'Imputed'
    imputed_handle = Line2D(
        [0], [0], marker='o', color='w', markerfacecolor='none', markeredgecolor='black',
        label='Imputed', markersize=8
    )

    # Get the existing legend and add the new handle
    handles, labels = ax.get_legend_handles_labels()
    handles.append(imputed_handle)
    labels.append('Imputed')

    # Update the legend with the new handle
    ax.legend(handles, labels, loc='lower right', ncols=2)
    fig.tight_layout()
    path_to_save = os.path.join(path_to_plotted_result, f"{dir_num}")
    os.makedirs(path_to_save, exist_ok=True)
    fig.savefig(
        os.path.join(path_to_save, f"{name}_response_{response_num + 1}_{response_name}.png"), bbox_inches='tight'
    )
    plt.close(fig)


def plot_correlation_signal_significance_over_ocus(
        intermediate_folder: str,
        plot_folder: str,
        balance_method: str,
        shuffling_cycle_num: int,
        shuffles_in_one_cycle: int = 0
) -> None:
    """
    Resulting correlation signal significance plot for the best balances across different OCUs.
    :param intermediate_folder: path to the results of PCC and shuffled PCC distribution calculations
    :param plot_folder: path to save the figure
    :param balance_method: the method used for CoDA transformation
    :param shuffling_cycle_num: current shuffling cycle number
    :param shuffles_in_one_cycle: number of shuffles in one cycle, defaults to 0
    :return:
    """
    # load figure input data
    dictionary = load_file('cluster_dict.pkl', intermediate_folder)
    index_list = load_file('response_index.pkl', intermediate_folder)

    try:
        shuffle_median = load_file('shuffle_median.pkl', intermediate_folder)
    except FileNotFoundError:
        shuffle_median = {}

    try:
        shuffle_ci_25 = load_file('shuffle_ci_25.pkl', intermediate_folder)
        shuffle_ci_75 = load_file('shuffle_ci_75.pkl', intermediate_folder)
        shuffle_ci_2_5 = load_file('shuffle_ci_2_5.pkl', intermediate_folder)
        shuffle_ci_97_5 = load_file('shuffle_ci_97_5.pkl', intermediate_folder)
    except FileNotFoundError:
        shuffle_ci_25 = {}
        shuffle_ci_75 = {}
        shuffle_ci_2_5 = {}
        shuffle_ci_97_5 = {}

    for key in dictionary.keys():
        df_pcc = pd.DataFrame(dictionary[key])

        df_pcc.index = index_list

        if shuffle_median and shuffle_ci_25 and shuffle_ci_75 and shuffle_ci_2_5 and shuffle_ci_97_5:
            df_pcc_median_values = pd.DataFrame(cast_nested_dict_to_array(shuffle_median[key]))
            df_pcc_ci_25_values = pd.DataFrame(cast_nested_dict_to_array(shuffle_ci_25[key]))
            df_pcc_ci_75_values = pd.DataFrame(cast_nested_dict_to_array(shuffle_ci_75[key]))
            df_pcc_ci_2_5_values = pd.DataFrame(cast_nested_dict_to_array(shuffle_ci_2_5[key]))
            df_pcc_ci_97_5_values = pd.DataFrame(cast_nested_dict_to_array(shuffle_ci_97_5[key]))
            for df in [
                df_pcc_median_values,
                df_pcc_ci_25_values, df_pcc_ci_75_values,
                df_pcc_ci_2_5_values, df_pcc_ci_97_5_values
            ]:
                df.index = index_list
        else:
            df_pcc_median_values = None
            df_pcc_ci_25_values = None
            df_pcc_ci_75_values = None
            df_pcc_ci_2_5_values = None
            df_pcc_ci_97_5_values = None

        for response in df_pcc.index:
            plot_correlation_signal_significance_to_response(
                balance_method, df_pcc, df_pcc_ci_25_values, df_pcc_ci_2_5_values, df_pcc_ci_75_values,
                df_pcc_ci_97_5_values, df_pcc_median_values, key, plot_folder, response, shuffles_in_one_cycle,
                shuffling_cycle_num)


def delete_older_shuffle_files(plot_folder, current_shuffles):
    # Regex to match files with the same pattern
    pattern = re.compile(r"_(\d+)_shuffles")

    for filename in os.listdir(plot_folder):
        match = pattern.search(filename)
        if match:
            # Extract the shuffle count from the filename
            shuffle_count = int(match.group(1))
            if shuffle_count < current_shuffles:
                # Delete the file if it has fewer shuffles
                os.remove(os.path.join(plot_folder, filename))


def plot_correlation_signal_significance_to_response(
        balance_method_val, pcc, pcc_ci_25_values, pcc_ci_2_5_values, pcc_ci_75_values, pcc_ci_97_5_values,
        pcc_median_values, case_key, plot_folder, response_tag, shuffles_per_cycle, shuffling_cycle_counter
):
    fig, ax = plt.subplots()
    ax.plot(pcc.columns, pcc.loc[response_tag, :], marker='.', linestyle='-', color='#c4941d', label=response_tag)
    plot_ci_95 = pcc_ci_2_5_values is not None and pcc_ci_97_5_values is not None
    if pcc_median_values is not None and plot_ci_95:
        # add median and 2.5%-97.5% to the plot
        pcc_ci_2_5_to_response = pcc_ci_2_5_values.loc[response_tag, :]
        pcc_ci_97_5_to_response = pcc_ci_97_5_values.loc[response_tag, :]
        ax.errorbar(
            pcc.columns, pcc_median_values.loc[response_tag, :],
            yerr=np.stack((pcc_median_values.loc[response_tag, :] - pcc_ci_2_5_to_response,
                           pcc_ci_97_5_to_response - pcc_median_values.loc[response_tag, :])),
            fmt='o', markersize=2, label=f"{response_tag}:shuffled:CI_95%",
            color='#a45acd'
        )
        ax.fill_between(
            pcc.columns, pcc_ci_97_5_to_response, pcc_ci_2_5_to_response,
            alpha=0.1, color=plt.gca().lines[-1].get_color()
        )
    plot_ci_50 = pcc_ci_25_values is not None and pcc_ci_75_values is not None
    if pcc_median_values is not None and plot_ci_50:
        # add median and 25%-75% CI to the plot
        pcc_ci_25_to_response = pcc_ci_25_values.loc[response_tag, :]
        pcc_ci_75_to_response = pcc_ci_75_values.loc[response_tag, :]
        ax.errorbar(
            pcc.columns, pcc_median_values.loc[response_tag, :],
            yerr=np.stack((pcc_median_values.loc[response_tag, :] - pcc_ci_25_to_response,
                           pcc_ci_75_to_response - pcc_median_values.loc[response_tag, :])),
            fmt='o', markersize=3, label=f"{response_tag}:shuffled:CI_50%",
            color='#6a5acd'
        )
        ax.fill_between(
            pcc.columns, pcc_ci_75_to_response, pcc_ci_25_to_response,
            alpha=0.2, color=plt.gca().lines[-1].get_color()
        )
    if shuffling_cycle_counter == 0:
        fig_title = "Correlation signal between LR transformed\nmicrobiome data and response"
    else:
        fig_title = f"Significance of the correlation signal between\n" \
                    f"LR transformed microbiome data and response after\n" \
                    f"{shuffling_cycle_counter} shuffling cycle(s) X {shuffles_per_cycle} shuffles per cycle"
    if balance_method_val == 'pairs':
        legend_title = f'Microbiome: {case_key}\nLR transformation: SLR {balance_method_val[0:-1]} balance'
    else:
        legend_title = f'Microbiome: {case_key}\nLR transformation: {balance_method_val}'
    ax.set_title(f'{fig_title}\n{legend_title}')
    # put legend in top left corner
    ax.legend(loc='lower right', fontsize=6)
    ax.set_ylabel(r"Absolute value of Pearson Correlation Coefficient $\rho$", size=10)
    ax.set_xlabel("# of OCUs", size=10)
    # change the `x` axis to log scale
    ax.set_xscale("log")
    # Customize the tick locations and labels
    # Set major ticks at base 10 and common intermediates (10, 20, 50, 100, 200, etc.)
    ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=10))
    # Hide minor tick labels (but keep them in grid if needed)
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=10))
    ax.xaxis.set_minor_formatter(NullFormatter())
    # Use scalar (non-sci) formatting for major ticks
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_ylim(0, 1)
    ax.set_ylim(0, 1)
    os.makedirs(plot_folder, exist_ok=True)

    current_shuffles = shuffling_cycle_counter * shuffles_per_cycle

    fig.savefig(os.path.join(
        plot_folder, f"{case_key}_{response_tag}_{current_shuffles}_shuffles.png"
    ), bbox_inches='tight')

    plt.close(fig)

    delete_older_shuffle_files(plot_folder, current_shuffles)


def create_clustered_heatmap(
        data: pd.DataFrame, plot_folder: str, case_key: str, response_tag: str, target_response: str
) -> None:

    file_extension = 'png'

    vmin = data.stack().min()  # Minimum value
    vmax = data.stack().max()  # Maximum value

    # Perform hierarchical clustering on rows and columns and create a heatmap
    cluster_map = sns.clustermap(
        data,
        metric="euclidean",
        method="ward",
        cmap="PuBuGn",
        vmin=vmin,  # Set minimum value for the color scale
        vmax=vmax,  # Set maximum value for the color scale
        row_cluster=True,  # Cluster rows
        col_cluster=True,  # Cluster columns
        xticklabels=False,  # Hide x-axis labels
        yticklabels=True,  # Show y-axis labels
        figsize=(12, 8)
    )
    # Ensure all y-axis labels are aligned with clustered data
    reordered_labels = [data.index[i] for i in cluster_map.dendrogram_row.reordered_ind]
    cluster_map.ax_heatmap.set_yticks([i + 0.5 for i in range(len(reordered_labels))])
    cluster_map.ax_heatmap.set_yticklabels(reordered_labels, rotation=0, fontsize=4)
    cluster_map.ax_heatmap.yaxis.set_tick_params(
        labelleft=True, labelright=False, labelrotation=0, pad=2, left=True, right=False,
        size=50, width=0.2, colors='tab:blue',
    )
    cluster_map.fig.suptitle(
        t=f'OTU Pairwise Significance Over OCU Clustering for {response_tag}', x=0.4, y=1.01, fontsize=10
    )

    # Save the heatmap to the specified folder
    os.makedirs(plot_folder, exist_ok=True)
    if response_tag == target_response:
        file_extension = 'svg'
    cluster_map.savefig(
        os.path.join(
            plot_folder, f'{case_key}_{response_tag}_clustered_heatmap.{file_extension}'
        ), format=file_extension, bbox_inches='tight'
    )

    plt.close(cluster_map.fig)

    # Extract and save clustering information to a JSON file
    clustering_info = extract_heatmap_clustering_information(cluster_map, data.index.to_list())
    with open(os.path.join(plot_folder, f'{case_key}_{response_tag}_clustered_heatmap_info.json'), 'w') as json_file:
        json.dump(clustering_info, json_file, indent=4)


def extract_heatmap_clustering_information(cluster_map: ClusterGrid, labels: list) -> dict:
    """Extracts clustering information from a heatmap generated by a hierarchical clustering algorithm.

    The function processes the ordered indices and linkage matrix and creates a mapping of clusters to their parent
    clusters, including the distance between clusters, and the number of elements merged at each linkage step.

    :param cluster_map: The ClusterMap object generated by seaborn's clustermap; contains clustering information.
    :param labels: The list of indices in the data object used to generate the heatmap.
    :return: A dictionary containing clustering information. Each key corresponds to a cluster index created during
    the clustering process. The values contain details about the clusters merged, the distance metric for the merge,
    and the number of elements merged at each step.
    """
    clusters = cluster_map.dendrogram_row.reordered_ind
    linkage = cluster_map.dendrogram_row.linkage.tolist()

    # Group labels by clusters
    clustering_info = {}
    for i, cluster in enumerate(clusters):
        clustering_info[i] = {
            "cluster index": i,
            "cluster 1": [labels[cluster]],
            "cluster 2": None,
            "distance": None,
            "number of elements": 1
        }

    for item in linkage:
        if item[0] < len(labels):
            cluster_1 = [labels[int(item[0])]]
        else:
            cluster_1 = clustering_info[int(item[0])]["cluster 1"] + clustering_info[int(item[0])]["cluster 2"]
        if item[1] < len(labels):
            cluster_2 = [labels[int(item[1])]]
        else:
            cluster_2 = clustering_info[int(item[1])]["cluster 1"] + clustering_info[int(item[1])]["cluster 2"]
        cluster_index = int(linkage.index(item)) + len(labels)
        heatmap_item = {
            "cluster index": cluster_index,
            "cluster 1": cluster_1,
            "cluster 2": cluster_2,
            "distance": item[2],
            "number of elements": int(item[3])
        }
        # Add heatmap item to a json object
        clustering_info[cluster_index] = heatmap_item

    return clustering_info
