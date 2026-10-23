import ast
import glob
import html
import json
import math
import os
import re
from datetime import datetime
from importlib.metadata import version, PackageNotFoundError
from urllib.parse import quote

import pandas as pd

from .compores_plotting import taxa_correlation_heatmap_path

REPORT_FILE_NAME = "report.html"
RANKING = "compores_response_ranking"
BALANCES = "balance_calculation_results"
SIGNAL_PLOTS = os.path.join("plots", "compores_signal_significance")
SCATTER_PLOTS = os.path.join("plots", "response_vs_best_balance")
TRACING = "otu_significance_tracing"
POWER_ANALYSIS = "synthetic_power_analysis"
GEV_FLOOR = 1e-11
MAX_LISTED_TAXA = 6
# The ranking lists every response; the details are written for the top-ranked ones, to keep the report small
MAX_DETAILED_RESPONSES = 200

CSS = """
:root { --bg: #ffffff; --fg: #1d1d1f; --muted: #6b6b73; --line: #e3e3e8; --card: #f7f7f9; --accent: #2f6fdb;
        --bar-gev: #8fb3ef; --bar-empirical: #b9b9c6; --code: #f0f0f3; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #16161a; --fg: #e8e8ec; --muted: #9a9aa6; --line: #2e2e36; --card: #1e1e24; --accent: #7aa7ff;
          --bar-gev: #4a72b5; --bar-empirical: #5a5a68; --code: #26262e; } }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg);
  font: 15px/1.5 -apple-system, "Segoe UI", Roboto, sans-serif; }
main { max-width: 1200px; margin: 0 auto; padding: 24px 16px 64px; overflow-wrap: anywhere; }
h1 { font-size: 26px; margin: 0 0 4px; }
h2 { font-size: 20px; margin: 40px 0 8px; border-bottom: 1px solid var(--line); padding-bottom: 6px; }
h3 { font-size: 16px; margin: 20px 0 8px; }
h2, details.response { scroll-margin-top: 48px; }
a { color: var(--accent); text-decoration: none; } a:hover { text-decoration: underline; }
.muted { color: var(--muted); }
code { background: var(--code); padding: 1px 5px; border-radius: 4px; font-size: 13px; }
nav { position: sticky; top: 0; background: var(--bg); border-bottom: 1px solid var(--line); padding: 8px 0;
  z-index: 1; }
nav a { margin-right: 16px; font-size: 14px; }
.summary { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 8px; margin: 16px 0; }
.summary div { background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 8px 12px; }
.summary b { display: block; font-size: 18px; }
.note { background: var(--card); border-left: 3px solid var(--accent); padding: 8px 12px; margin: 12px 0;
  font-size: 14px; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 14px; overflow-wrap: normal; }
th, td { text-align: left; padding: 5px 8px; border-bottom: 1px solid var(--line); vertical-align: top; }
th { font-weight: 600; } th.sortable { cursor: pointer; } th.sortable:hover { color: var(--accent); }
td a, td.nowrap { white-space: nowrap; }
td.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
td.bar { position: relative; min-width: 90px; padding-bottom: 9px; }
td.bar i { position: absolute; left: 8px; bottom: 4px; height: 4px; border-radius: 2px; }
td.gev i { background: var(--bar-gev); } td.empirical i { background: var(--bar-empirical); }
input.filter { width: 100%; max-width: 360px; padding: 6px 10px; border: 1px solid var(--line); border-radius: 6px;
  background: var(--bg); color: var(--fg); margin: 8px 0; }
details.response { border: 1px solid var(--line); border-radius: 8px; margin: 8px 0; background: var(--card); }
details.response > summary { cursor: pointer; padding: 10px 12px; font-weight: 600; }
details.response > div { padding: 0 12px 12px; }
.method { display: grid; grid-template-columns: minmax(0, 420px) minmax(0, 1fr); gap: 16px; align-items: start; }
.method > div { min-width: 0; }
img.plot { width: 100%; max-width: 560px; border: 1px solid var(--line); border-radius: 6px; background: #fff; }
img.plot.heatmap { max-width: 760px; }
.levels a { margin-right: 6px; font-size: 13px; }
ul.files { columns: 2; font-size: 14px; }
@media (max-width: 900px) { .method { grid-template-columns: minmax(0, 1fr); } }
@media (max-width: 700px) { ul.files { columns: 1; } }
"""

SCRIPT = """
function openTarget() {
  const target = location.hash && document.getElementById(decodeURIComponent(location.hash.slice(1)));
  if (target && target.tagName === 'DETAILS') { target.open = true; target.scrollIntoView(); }
}
window.addEventListener('hashchange', openTarget);
window.addEventListener('load', openTarget);
document.querySelectorAll('table.sortable').forEach(table => {
  table.querySelectorAll('th.sortable').forEach((th, index) => th.addEventListener('click', () => {
    const body = table.tBodies[0], rows = Array.from(body.rows), descending = th.dataset.order !== 'desc';
    const key = row => {
      const value = row.cells[index].dataset.value ?? row.cells[index].textContent;
      const number = parseFloat(value);
      return isNaN(number) ? value.toLowerCase() : number;
    };
    rows.sort((a, b) => (key(a) > key(b) ? 1 : key(a) < key(b) ? -1 : 0) * (descending ? -1 : 1));
    table.querySelectorAll('th').forEach(other => delete other.dataset.order);
    th.dataset.order = descending ? 'desc' : 'asc';
    rows.forEach(row => body.appendChild(row));
  }));
});
document.querySelectorAll('input.filter').forEach(input => input.addEventListener('input', () => {
  const text = input.value.toLowerCase();
  input.closest('section.experiment').querySelectorAll('[data-response]').forEach(element => {
    element.style.display = element.dataset.response.toLowerCase().includes(text) ? '' : 'none';
  });
}));
"""


def _escape(value) -> str:
    return html.escape(str(value), quote=True)


def _url(path: str) -> str:
    """Turns a path relative to the output folder into a link target, which names with spaces, `#` or `%` keep valid."""
    return _escape(quote(path.replace(os.sep, "/")))


def _link(path: str, text: str) -> str:
    return f'<a href="{_url(path)}">{_escape(text)}</a>'


def _anchor(*parts: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", "-".join(parts))


def format_p_value(p_value: float) -> str:
    """Formats a p-value for the report, marking the values at the floor of the GEV estimate."""
    if pd.isna(p_value):
        return "–"
    if p_value <= GEV_FLOOR:
        return "≤ 1e-11"
    return f"{p_value:.2g}"


def format_taxa(value) -> tuple[str, str]:
    """Formats the taxa of one side of a balance, stored as the text of a list, e.g. "['OTU_5', 'OTU_24']".

    :return: A short text, which lists at most `MAX_LISTED_TAXA` taxa, and the full list, for a tooltip.
    """
    if not isinstance(value, str):
        return "–", ""
    try:
        taxa = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value, value
    if not isinstance(taxa, (list, tuple)) or not taxa:
        return "–", ""
    full_text = ", ".join(map(str, taxa))
    if len(taxa) <= MAX_LISTED_TAXA:
        return full_text, full_text
    return f"{len(taxa)} taxa: {', '.join(map(str, taxa[:MAX_LISTED_TAXA]))}, …", full_text


def _list_folders(path: str) -> list[str]:
    return sorted(name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))) \
        if os.path.isdir(path) else []


def _latest_ranking(outputs_path: str, experiment: str, method: str) -> tuple[pd.DataFrame | None, int]:
    """Reads the response ranking of the last completed shuffling cycle."""
    cycle_files = {}
    for path in glob.glob(os.path.join(outputs_path, RANKING, experiment, method, "*_at_shuffling_cycle_*.csv")):
        match = re.search(r"_at_shuffling_cycle_(\d+)\.csv$", path)
        if match:
            cycle_files[int(match.group(1))] = path
    if not cycle_files:
        return None, 0
    last_cycle = max(cycle_files)
    return pd.read_csv(cycle_files[last_cycle], index_col=0), last_cycle


def _total_shuffles(outputs_path: str, experiment: str, method: str) -> int:
    try:
        with open(os.path.join(outputs_path, f"state_{method}.json"), "r") as f:
            return sum(json.load(f).get(experiment, {}).get("run_comp_shuffle_iter", []))
    except (FileNotFoundError, json.JSONDecodeError):
        return 0


def _signal_plots(outputs_path: str, experiment: str, method: str) -> dict[str, str]:
    """Finds the signal significance plot of every response, for the largest number of shuffles."""
    folder = os.path.join(SIGNAL_PLOTS, experiment, method)
    if not os.path.isdir(os.path.join(outputs_path, folder)):
        return {}
    pattern = re.compile(rf"^{re.escape(experiment)}_(.+)_(\d+)_shuffles\.png$")
    plots = {}
    for name in sorted(os.listdir(os.path.join(outputs_path, folder))):
        match = pattern.match(name)
        if match and int(match.group(2)) >= plots.get(match.group(1), (-1, ""))[0]:
            plots[match.group(1)] = (int(match.group(2)), os.path.join(folder, name))
    return {response: path for response, (_, path) in plots.items()}


def _scatter_levels(outputs_path: str, experiment: str, method: str) -> list[str]:
    """Lists the OCU levels of the response vs. best balance plots, from the finest."""
    folder = os.path.join(outputs_path, SCATTER_PLOTS, experiment, method)
    return sorted((name for name in _list_folders(folder) if name.isdigit()), key=int, reverse=True)


def _ordered_responses(methods: list[str], rankings: dict) -> list[str]:
    """Lists the responses, in the order of the GEV ranking of the first balance method."""
    responses = list(dict.fromkeys(
        response for method in methods if rankings[method] is not None for response in rankings[method].index
    ))
    first_ranking = next((rankings[method] for method in methods if rankings[method] is not None), None)
    if first_ranking is not None:
        responses.sort(key=lambda r: -first_ranking.loc[r, "gev"] if r in first_ranking.index else math.inf)
    return responses


def _ranking_table(experiment, methods, rankings, total_shuffles, responses, detailed_responses) -> str:
    head = "".join(
        f'<th class="sortable">{_escape(method)} GEV</th><th class="sortable">{_escape(method)} empirical</th>'
        for method in methods
    )
    rows = []
    for rank, response in enumerate(responses, 1):
        cells = []
        for method in methods:
            # The bars are scaled to the largest possible score of each column
            for column, css_class, max_score in (
                    ("gev", "gev", -math.log(GEV_FLOOR)),
                    ("bootstrap", "empirical", math.log(total_shuffles[method] + 1) or 1.0)
            ):
                ranking = rankings[method]
                if ranking is None or response not in ranking.index:
                    cells.append('<td class="num">–</td>')
                    continue
                score = float(ranking.loc[response, column])
                width = max(0.0, min(1.0, score / max_score))
                cells.append(
                    f'<td class="num bar {css_class}" data-value="{score}">{score:.2f}'
                    f'<i style="width: calc({width:.3f} * (100% - 16px))"></i></td>'
                )
        rows.append(
            f'<tr data-response="{_escape(response)}"><td class="num">{rank}</td>'
            + (f'<td><a href="#{_anchor(experiment, response)}">{_escape(response)}</a></td>'
               if response in detailed_responses else f'<td class="nowrap">{_escape(response)}</td>')
            + f'{"".join(cells)}</tr>'
        )
    return (
        '<div class="scroll"><table class="sortable"><thead><tr><th class="sortable">#</th>'
        f'<th class="sortable">Response</th>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
    )


def _balance_table(path: str) -> str:
    """Tabulates the best balance at every OCU level, from a `..._combined_ocu_level_results.csv` file."""
    table = pd.read_csv(path, index_col=0)
    has_denominator = "DEN_Taxa_List" in table.columns and table["DEN_Taxa_List"].notna().any()
    head = (
        f"<th>OCU level</th><th>{'Numerator' if has_denominator else 'OCU'}</th>"
        + ("<th>Denominator</th>" if has_denominator else "")
        + "<th>rho</th><th>Shuffled median</th><th>p (GEV)</th><th>p (empirical)</th>"
    )
    rows = []
    for level, row in table.iterrows():
        numerator, numerator_tooltip = format_taxa(row.get("NUM_Taxa_List"))
        cells = [f'<td class="nowrap">{_escape(level)}</td>',
                 f'<td title="{_escape(numerator_tooltip)}">{_escape(numerator)}</td>']
        if has_denominator:
            denominator, denominator_tooltip = format_taxa(row.get("DEN_Taxa_List"))
            cells.append(f'<td title="{_escape(denominator_tooltip)}">{_escape(denominator)}</td>')
        cells += [
            f'<td class="num">{row["rho"]:.2f}</td>',
            f'<td class="num">{row["rho_shuffle_median"]:.2f}</td>',
            f'<td class="num">{format_p_value(row["rho_p_value"])}</td>',
            f'<td class="num">{format_p_value(row["rho_p_value_bootstrap"])}</td>',
        ]
        rows.append(f"<tr>{''.join(cells)}</tr>")
    caption = "" if has_denominator else (
        '<p class="muted">CLR: the best balance at each level is one OCU against the geometric mean of all OCUs; a '
        "negative rho means that the response falls as the OCU rises.</p>"
    )
    return (
        f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
        f"{caption}"
    )


def _response_details(outputs_path, experiment, methods, rankings, signal_plots, scatter_levels, response) -> str:
    scores = " · ".join(
        f"{_escape(method)} GEV {rankings[method].loc[response, 'gev']:.2f}" for method in methods
        if rankings[method] is not None and response in rankings[method].index
    )
    parts = []
    for method in methods:
        plot = signal_plots[method].get(response)
        plot_html = (
            f'<a href="{_url(plot)}"><img class="plot" loading="lazy" src="{_url(plot)}" '
            f'alt="{_escape(method)} signal significance of {_escape(response)}"></a>'
        ) if plot else ""
        content = []
        table_path = os.path.join(
            BALANCES, experiment, method, "ocu_level_results", "regular", f"{response}_combined_ocu_level_results.csv"
        )
        if os.path.exists(os.path.join(outputs_path, table_path)):
            content.append(_balance_table(os.path.join(outputs_path, table_path)))
            sample_level_path = os.path.join(BALANCES, experiment, method, "sample_level_results", response)
            content.append(
                f'<p class="muted">Table: {_link(table_path, "CSV")} · values of every sample: '
                f'{_link(sample_level_path, "folder")}</p>'
            )
        scatter_folder = os.path.join(SCATTER_PLOTS, experiment, method)
        level_links = [
            _link(os.path.join(scatter_folder, level, f"{experiment}_{response}.png"), level)
            for level in scatter_levels[method]
            if os.path.exists(os.path.join(outputs_path, scatter_folder, level, f"{experiment}_{response}.png"))
        ]
        if level_links:
            content.append(f'<p class="levels">Response vs. best balance, by OCU level: {"".join(level_links)}</p>')
        if plot_html or content:
            parts.append(
                f'<h3>{_escape(method)}</h3><div class="method"><div>{plot_html}</div>'
                f'<div>{"".join(content)}</div></div>'
            )
    return (
        f'<details class="response" id="{_anchor(experiment, response)}" data-response="{_escape(response)}">'
        f'<summary>{_escape(response)} <span class="muted">{scores}</span></summary>'
        f'<div>{"".join(parts)}</div></details>'
    )


def _files_list(outputs_path: str, experiment: str, methods: list[str]) -> str:
    candidates = [
        (os.path.join("logs", "compores_sessions.log"), "Session log"),
        (os.path.join("logs", experiment, "compores_main.log"), "Main log of the experiment"),
        (os.path.join("logs", experiment), "Logs of every OCU level"),
        (os.path.join("preprocessing_results", "sparcckit", f"taxa_correlation_{experiment}.tsv"),
         "SparCC taxon correlations"),
        (os.path.join("preprocessing_results", "microbiome", "OCUs", experiment,
                      f"{experiment}_ocu_clustering_dictionary.json"), "OCU membership at every level"),
        ("preprocessed_samples", "Input tables, as aligned by sample"),
    ]
    candidates += [(os.path.join(RANKING, experiment, method), f"{method} rankings after every shuffling cycle")
                   for method in methods]
    candidates += [(f"state_{method}.json", f"{method} progress, used to resume") for method in methods]
    items = [f"<li>{_link(path, text)}</li>" for path, text in candidates
             if os.path.exists(os.path.join(outputs_path, path))]
    return f'<ul class="files">{"".join(items)}</ul>'


def _tracing_table(outputs_path: str, experiment: str, responses: list[str]) -> str:
    """Lists the top OTUs of the CLR OTU tracing of every traced response, in the order of the response ranking."""
    prefix = f"{experiment}_otu_clr_minus_mean_log_p_values_"
    paths = {
        os.path.basename(path)[len(prefix):-len(".tsv")]: path
        for path in glob.glob(os.path.join(outputs_path, TRACING, experiment, "CLR", f"{prefix}*.tsv"))
    }
    rows = []
    for response in [r for r in responses if r in paths] + sorted(set(paths) - set(responses)):
        path = paths[response]
        scores = pd.read_csv(path, sep="\t", index_col=0)
        top_otus = []
        for column in ("minus_mean_log_p_value_positive", "minus_mean_log_p_value_negative"):
            top = scores[column].sort_values(ascending=False).head(3) if column in scores.columns else pd.Series()
            top_otus.append(", ".join(f"{_escape(otu)} ({score:.1f})" for otu, score in top.items()))
        rows.append(
            f'<tr data-response="{_escape(response)}"><td>{_escape(response)}</td><td>{top_otus[0]}</td>'
            f'<td>{top_otus[1]}</td><td>{_link(os.path.relpath(path, outputs_path), "TSV")}</td></tr>'
        )
    if not rows:
        return ""
    return (
        '<p>The three top-scoring OTUs of each response in the <code>CLR</code> OTU tracing, with their scores: −ln(p) '
        "averaged over the OCU levels, a ranking score rather than a calibrated p-value. The <code>pairs</code> scores "
        f"and the full tables are in the {_link(os.path.join(TRACING, experiment), 'tracing folder')}.</p>"
        '<div class="scroll"><table><thead><tr><th>Response</th><th>Top positive OTUs</th><th>Top negative OTUs</th>'
        f'<th></th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
    )


def _taxa_clustering(outputs_path: str, experiment: str) -> str:
    """Shows the clustered heatmap of the taxon correlations, which the OCUs are cut from."""
    heatmap = taxa_correlation_heatmap_path("", experiment)
    if not os.path.exists(os.path.join(outputs_path, heatmap)):
        return ""
    correlations = os.path.join("preprocessing_results", "sparcckit", f"taxa_correlation_{experiment}.tsv")
    membership = os.path.join("preprocessing_results", "microbiome", "OCUs", experiment,
                              f"{experiment}_ocu_clustering_dictionary.json")
    return (
        "<p>The SparCC correlations of the taxa that passed the filters, ordered by the hierarchical clustering that "
        "defines the OCUs: average linkage of the distance 0.5 × (1 − correlation). Every OCU level cuts this tree "
        f"into that number of clusters. Data: {_link(correlations, 'correlations')}, "
        f"{_link(membership, 'OCU membership at every level')}.</p>"
        f'<a href="{_url(heatmap)}"><img class="plot heatmap" loading="lazy" src="{_url(heatmap)}" '
        'alt="Clustered heatmap of the taxon correlations"></a>'
    )


def _power_analysis_plots(outputs_path: str) -> str:
    plots = sorted(glob.glob(os.path.join(outputs_path, POWER_ANALYSIS, "*", "*_mean_auroc_vs_noise_*.png")))
    return "".join(
        f'<h3>{_escape(os.path.basename(os.path.dirname(path)))}</h3>'
        f'<a href="{_url(os.path.relpath(path, outputs_path))}"><img class="plot" loading="lazy" '
        f'src="{_url(os.path.relpath(path, outputs_path))}" alt="Mean ROC AUC vs. noise level"></a>'
        for path in plots
    )


def _experiment_section(outputs_path: str, experiment: str, show_name: bool) -> tuple[str, list[str]]:
    """Builds the report section of one experiment, and its navigation links."""
    method_folders = _list_folders(os.path.join(outputs_path, RANKING, experiment))
    methods = [method for method in ("CLR", "pairs") if method in method_folders]
    methods += [method for method in method_folders if method not in methods]
    rankings, cycles, total_shuffles = {}, {}, {}
    for method in methods:
        rankings[method], cycles[method] = _latest_ranking(outputs_path, experiment, method)
        total_shuffles[method] = _total_shuffles(outputs_path, experiment, method)
    responses = _ordered_responses(methods, rankings)
    detailed_responses = responses[:MAX_DETAILED_RESPONSES]
    signal_plots = {method: _signal_plots(outputs_path, experiment, method) for method in methods}
    scatter_levels = {method: _scatter_levels(outputs_path, experiment, method) for method in methods}

    levels = []
    for method in methods:
        tables = sorted(glob.glob(os.path.join(outputs_path, BALANCES, experiment, method, "ocu_level_results",
                                               "regular", "*_combined_ocu_level_results.csv")))
        if tables:
            levels = [int(str(level).split()[0]) for level in pd.read_csv(tables[0], index_col=0).index]
            break
    levels_text = ", ".join(map(str, levels)) if len(levels) <= 4 else f"{levels[0]}, {levels[1]}, …, {levels[-1]}"
    shuffles_text = "; ".join(f"{method}: {total_shuffles[method]}" for method in methods)
    cycles_text = "; ".join(f"{method}: {cycles[method]}" for method in methods)

    title = f"{experiment}: " if show_name else ""
    section_id = _anchor("e", experiment)
    tracing = _tracing_table(outputs_path, experiment, responses)
    taxa_clustering = _taxa_clustering(outputs_path, experiment)
    parts = [
        f'<h2 id="{section_id}-ranking">{_escape(title)}Response ranking</h2>',
        '<div class="summary">'
        f"<div><b>{len(responses)}</b>responses</div>"
        f"<div><b>{_escape(', '.join(methods))}</b>balance methods</div>"
        f"<div><b>{len(levels)}</b>OCU levels ({_escape(levels_text)})</div>"
        f"<div><b>{_escape(shuffles_text)}</b>shuffles per OCU level</div>"
        f"<div><b>{_escape(cycles_text)}</b>shuffling cycles completed</div></div>",
        '<p class="note">Each score is a response\'s −ln(p) averaged over the OCU levels, where p is the p-value of '
        "its best balance at a level: the higher the score, the stronger the association. The scores rank the "
        "responses; "
        f"they are not calibrated p-values. The GEV p-values are floored at 1e-11, so a GEV score is at most "
        f"{-math.log(GEV_FLOOR):.1f}; with n shuffles, an empirical score is at most ln(n + 1). Click a column title "
        "to sort the table, and a response for its details.</p>",
        '<input class="filter" type="search" placeholder="Filter responses" aria-label="Filter responses">',
        _ranking_table(experiment, methods, rankings, total_shuffles, responses, set(detailed_responses)),
        f'<h2 id="{section_id}-responses">{_escape(title)}Responses</h2>',
        '<p class="muted">For every response and balance method: the best correlation at every OCU level against the '
        "correlations of the shuffled data, and the best balance at every level.</p>",
    ]
    if len(responses) > len(detailed_responses):
        parts.append(
            f'<p class="note">The details below are for the {len(detailed_responses)} top-ranked of the '
            f"{len(responses)} responses. The results of every response are in the "
            f"{_link(os.path.join(BALANCES, experiment), 'balance results')} and "
            f"{_link(os.path.join('plots'), 'plots')} folders.</p>"
        )
    parts += [
        _response_details(outputs_path, experiment, methods, rankings, signal_plots, scatter_levels, response)
        for response in detailed_responses
    ]
    if taxa_clustering:
        parts.append(f'<h2 id="{section_id}-taxa">{_escape(title)}Taxon clustering</h2>{taxa_clustering}')
    if tracing:
        parts.append(f'<h2 id="{section_id}-tracing">{_escape(title)}OTU tracing</h2>{tracing}')
    parts.append(
        f'<h2 id="{section_id}-files">{_escape(title)}Files</h2>{_files_list(outputs_path, experiment, methods)}'
    )

    prefix = f"{experiment}: " if show_name else ""
    nav = [f'<a href="#{section_id}-ranking">{_escape(prefix)}Ranking</a>',
           f'<a href="#{section_id}-responses">Responses</a>']
    if taxa_clustering:
        nav.append(f'<a href="#{section_id}-taxa">Taxon clustering</a>')
    if tracing:
        nav.append(f'<a href="#{section_id}-tracing">OTU tracing</a>')
    nav.append(f'<a href="#{section_id}-files">Files</a>')
    return f'<section class="experiment" id="{section_id}">{"".join(parts)}</section>', nav


def write_results_report(outputs_path: str) -> str | None:
    """Writes `report.html` to the output folder: an overview of the results, which links to the result files.

    The report uses paths relative to the output folder, so it stays valid when the folder is moved or copied.

    :param outputs_path: The output folder of CompoRes, `PATH_TO_OUTPUTS` in the configuration file.
    :return: The path to the report, or None if the folder holds no response ranking yet.
    """
    outputs_path = os.path.abspath(outputs_path)
    experiments = _list_folders(os.path.join(outputs_path, RANKING))
    if not experiments:
        return None
    sections, nav = [], []
    for experiment in experiments:
        section, experiment_nav = _experiment_section(outputs_path, experiment, show_name=len(experiments) > 1)
        sections.append(section)
        nav += experiment_nav
    power_analysis = _power_analysis_plots(outputs_path)
    if power_analysis:
        sections.append(f'<h2 id="power-analysis">Synthetic power analysis</h2>{power_analysis}')
        nav.append('<a href="#power-analysis">Power analysis</a>')

    title = f"CompoRes results: {', '.join(experiments)}"
    try:
        written_by = f"CompoRes {version('compores')}"
    except PackageNotFoundError:
        written_by = "CompoRes"
    page = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width, initial-scale=1"><title>{_escape(title)}</title>'
        f"<style>{CSS}</style></head><body><main><h1>{_escape(title)}</h1>"
        f'<p class="muted">Results in <code>{_escape(outputs_path)}</code>; report written by {_escape(written_by)} on '
        f'{datetime.now().strftime("%Y-%m-%d %H:%M")}. The links open the result files in this folder.</p>'
        f'<nav>{"".join(nav)}</nav>{"".join(sections)}</main><script>{SCRIPT}</script></body></html>'
    )
    report_path = os.path.join(outputs_path, REPORT_FILE_NAME)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(page)
    return report_path
