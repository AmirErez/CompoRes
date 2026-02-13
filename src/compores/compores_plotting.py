import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
        path_to_plotted_result: str | os.PathLike,
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
    dictionary = load_file('correlation_coefficient.pkl', intermediate_folder)
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
        pcc_median_values, case_key, plot_folder, response_tag, shuffles_per_cycle, shuffling_cycle_counter,
        mark_negative_pcc: bool = False
):
    fig, ax = plt.subplots()
    pcc_vals = pcc.loc[response_tag, :]
    pcc_vals_mask = pcc_vals < 0
    ax.plot(pcc.columns, np.abs(pcc_vals), marker='.', linestyle='-', color='#c4941d', label=response_tag)
    if pcc_vals_mask.any() and mark_negative_pcc:
        ax.scatter(
            pcc.columns[pcc_vals_mask],
            np.abs(pcc_vals)[pcc_vals_mask],
            marker="o",
            edgecolors="#c4941d",
            facecolors="none",
            label="correlation coefficient is negative",
            zorder=5,
        )
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
    # put legend in a top-left corner
    ax.legend(loc='lower right', fontsize=6)
    ax.set_ylabel(r"Absolute value of Correlation Coefficient $\rho$", size=10)
    ax.set_xlabel("# of OCUs", size=10)
    # change the `x` axis to a log scale
    ax.set_xscale("log")
    # Customize the tick locations and labels
    # Set major ticks at base 10 and common intermediates (10, 20, 50, 100, 200, etc.)
    ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=10))
    # Hide minor tick labels (but keep them in the grid if needed)
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
