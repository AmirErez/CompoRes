# CompoRes

![commits](https://img.shields.io/github/commit-activity/t/AmirErez/CompoRes)
![last](https://img.shields.io/github/last-commit/AmirErez/CompoRes)
![checks](https://img.shields.io/github/check-runs/AmirErez/CompoRes/main?label=code%20checks)

CompoRes searches for associations between microbiome composition and measured responses, such as host gene expression or environmental measurements. It groups correlated taxa into operational clustered units (OCUs), searches for response-associated log-ratios at several clustering levels, and compares the strongest correlations with shuffled-data results.

**You provide:** sample-by-taxon and sample-by-response TSV files, a YAML configuration, and optional sample categories. **You receive:** an HTML report (`report.html`) of the response rankings, observed-versus-shuffled correlation plots and best log-ratios, a clustered heatmap of the taxon correlations, and optional taxon tracing and synthetic classification analyses. Associations describe relative abundance patterns; they do not establish causation or absolute-abundance changes.

## Quick start

Linux (glibc 2.28+, e.g. RHEL/Rocky 8+ or Ubuntu 20.04+) and macOS (Intel or Apple Silicon) are supported. You need `conda` or `mamba`; if you have neither, install [Miniforge](https://github.com/conda-forge/miniforge). No administrator rights are needed for installation into your conda environment.

```bash
git clone https://github.com/AmirErez/CompoRes.git
cd CompoRes
conda env create --file env/compores_env.yml
conda activate compores_env
pip install -e .
compores --config configs/config_synthetic_data_quick.yaml
```

You may substitute `mamba` for `conda`. The quick run uses 50 taxa, 100 samples, 10 responses, 10 OCU levels and 10 shuffles per level. It took about 75 seconds on an 18-core Apple Silicon Mac; your runtime will vary. Results go to `artificial_experiment_output_quick`. Open its `report.html` in a browser (`open artificial_experiment_output_quick/report.html` on macOS, `xdg-open` on Linux): it ranks the responses and shows the plots and best balances of each, with links to all result files. The five planted responses should tend to rank above five noise responses; exact values vary.

For the full synthetic run:

```bash
compores --demo
```

The full demo took 11 to 13 minutes on the same machine (48 levels, 50 shuffles per level), writing to `artificial_experiment_output` in the repository. From the repository root, `compores --config configs/config_synthetic_data.yaml` selects the same run. See the [illustrated tutorial](docs/tutorial.md) for the complete walkthrough, tables and taxon tracing.

**CLI behavior:** `compores`, `compores -h` and `compores --help` print help and exit without starting analysis; before version 0.3.0, plain `compores` started the full demo (see the [changelog](CHANGELOG.md)). `--demo` and `--config` are mutually exclusive. For backward compatibility, analysis options without either selection (for example, `compores --preprocess`) still run the synthetic demo and print a notice; use an explicit selection in new commands. The bundled demo requires the repository and its editable install (`pip install -e .`); demo assets are not included in a normal wheel install.

Installation details, editable installs, environment recreation, uninstalling, and the `python -m compores` alternative are in the [reference](docs/reference.md#installation).

## Run your own data

1. Prepare matching sample IDs in the input tables described below.
2. Copy the repository's commented `configs/config.yaml` and edit the copy. See [Configuration](#configuration).
3. Run with an explicit path (replace the example with your actual file):

```bash
compores --config /path/to/my_config.yaml
```

For preprocessing and clustering only:

```bash
compores --config /path/to/my_config.yaml --preprocess
```

Preprocessing outputs appear in `preprocessing_results/`, and the clustered heatmap of the taxon correlations in `plots/taxa_correlation_clustering/`: a quick look at how the taxa will be grouped into OCUs. The installed command works from any directory. Without installation, run `python -m compores` from `src/`; use an absolute configuration path to avoid ambiguity.

## Input TSV files

Use actual tabs, not spaces or commas. Each row is a sample; the first column contains sample IDs. **The first header cell is empty**: each header line below begins with a literal tab. Examples are minimal format illustrations, not a meaningful analysis dataset.

With `GROUP1` = `Neo`, `GROUP2` = `IP`, and `GROUP3` = `SI_lumen`, filenames are:

**Microbiome: `Neo-IP-SI_lumen.tsv`**

```tsv
	OTU_1	OTU_2	OTU_3
sample_1	10	20	30
sample_2	15	25	10
sample_3	30	10	20
```

**Responses: `Neo-IP.tsv`**

```tsv
	gene_1	gene_2
sample_1	2.1	5.0
sample_2	3.2	4.1
sample_3	4.0	6.2
```

**Optional metadata: `Neo-IP-metadata.tsv`**

```tsv
	Category
sample_1	control
sample_2	treated
sample_3	control
```

Metadata supports up to two categories for coloring scatter plots. It is used only for this coloring: it does not adjust the analysis or restrict the shuffling, so it does not control for confounding, repeated measures or batch effects.

Inputs must be numeric, with at least three retained samples and three microbiome taxa, and at least one response. Microbiome values must be non-negative; negative response values are allowed. Missing response values are currently replaced with zero, and response columns containing only zeros after replacement are removed. Missing measurements are not necessarily biological zeros: prepare response missingness deliberately before running. Microbiome preprocessing filters rare taxa, imputes zeros/missing taxon values, and normalizes each sample to sum to one. Filtering must leave sufficient taxa for analysis.

Use unique sample IDs and consistent metadata IDs. CompoRes retains the intersection of microbiome and response sample IDs, sorts the retained samples, and logs samples omitted because they occur in only one file. Check those exclusions against your intended cohort. Duplicate IDs normally raise an error; `--deduplicate` keeps only the first occurrence. Resolve duplicates deliberately rather than silently discarding measurements.

## Configuration

Start from `configs/config.yaml`; it is the authority for complete syntax, supported values and settings, each explained by a comment. The main settings:

| Setting | What to check or change |
|---|---|
| `GROUP1`, `GROUP2`, `GROUP3` | Name the experiment and determine input filenames. `Neo`, `IP`, `SI_lumen` gives the filenames above. Preserve the example file's value structure. |
| `PATH_TO_MICROBIOME` | Directory containing `{GROUP1}-{GROUP2}-{GROUP3}.tsv`. |
| `PATH_TO_RESPONSE` | Directory containing `{GROUP1}-{GROUP2}.tsv`. |
| `PATH_TO_METADATA` | Directory containing `{GROUP1}-{GROUP2}-metadata.tsv`; set to an empty string if absent. |
| `PATH_TO_OUTPUTS` | Destination for results and resume state; use a fresh destination for a distinct analysis. |
| `TAXA_FILTER` | Removes taxa present in fewer than this fraction of samples. |
| `CODA_METHOD` | CLR is computed by default; `pairs` adds pairwise log-ratios. Leave empty for CLR only, using the example YAML's syntax. |
| `CORR` | `pearson` or `spearman`; the synthetic examples use `pearson`. With `spearman`, the best balance is chosen by Spearman correlation, but its reported correlation (`rho`) and p-values are the Pearson correlation of that balance, for the observed and the shuffled data alike. |
| `OCU_SAMPLING_RATE` | Step between processed OCU counts. Larger steps process fewer levels. Quick/full examples use 5/1 respectively. |
| `MAX_OCU` | Maximum OCU count processed for `pairs`; 800 in the example configuration. Higher levels still run for CLR. |
| `SHUFFLE` | Which sample labels are permuted to build the null distribution: `microbiome` or `response`; the synthetic examples use `microbiome`. The permutation is over all samples, not within groups. |
| `N_SHUFFLES`, `SHUFFLE_CYCLES` | Shuffles per cycle and cycle count; synthetic quick/full totals are 5 × 2 = 10 and 10 × 5 = 50 per level. |
| `SPARCCKIT_MAX_ITER` | Controls the dataset-size-dependent SparCC iteration schedule. |

Relative input paths are resolved relative to the configuration file's directory. Older working-directory-relative configurations are detected with a logged warning when run from their original directory. Inspect the log and resolved inputs when migrating an old configuration.

## Output navigation

Outputs are rooted at `PATH_TO_OUTPUTS`, generally split by experiment `{GROUP1}-{GROUP2}-{GROUP3}` and method (`CLR` and optionally `pairs`).

| Start here | What it contains |
|---|---|
| `report.html` | An overview to open in a browser: the response ranking, the signal significance plots and best balances of the responses (of the 200 top-ranked ones, for larger runs), and the taxon clustering heatmap, linking to the files below. Written at the end of every run, unless `--no_plotting` is given. |
| `compores_response_ranking/` | Response scores averaged over OCU levels, with a separate file per shuffling cycle. Use the final completed cycle. |
| `plots/compores_signal_significance/` | Observed best absolute correlation against shuffled maxima across OCU levels. |
| `plots/response_vs_best_balance/` | Scatter plots by response and level, plus `..._OCU_taxa_mapping.csv` files. |
| `plots/taxa_correlation_clustering/` | Heatmap of the SparCC taxon correlations, ordered by the hierarchical clustering that the OCUs are cut from, with its dendrogram (up to 1000 taxa). |
| `balance_calculation_results/` | `ocu_level_results/regular/` holds best balances, correlations, regressions and p-value estimates; `sample_level_results/` holds sample-level values. |
| `otu_significance_tracing/` | Optional directional OTU and OTU-pair scores. |
| `synthetic_power_analysis/` | Optional synthetic classification results. |
| `preprocessing_results/` | Processed tables, SparCC correlations, OCU matrices, imputation and clustering JSON metadata. |
| `preprocessed_samples/` | Sample-aligned input tables; check IDs and counts against your intended sample set. |
| `compores_basic_results/` | Intermediate `.pkl` files used for plotting and later analysis. |
| `logs/` | Session, experiment and per-OCU logs. |
| `state_CLR.json`, `state_pairs.json` | Progress used to resume completed steps. |

For example, the final quick-run pairs ranking is
`compores_response_ranking/50otu-100samples-synthetic/pairs/minus_mean_log_p_value_sorted_responses_at_shuffling_cycle_2.csv`.

## Methods and log-ratio definitions

SparCC estimates taxon correlations. Average-linkage hierarchical clustering uses distances `0.5 × (1 − correlation)`. Cutting this tree produces OCUs; an OCU abundance is the **sum** of its member taxa's abundances in each sample. Selected clustering levels range from individual taxa toward three OCUs; the sampling step may skip the endpoint.

At a level with K OCUs, let x[s,j] > 0 be the processed abundance of OCU j in sample s. Natural logarithms give:

- **CLR:** `g[s] = exp((1/K) × Σ_j ln(x[s,j]))`; candidate j is `ln(x[s,j]/g[s]) = ln(x[s,j]) − (1/K) × Σ_j ln(x[s,j])`. The geometric mean includes all K OCUs, including j.
- **pairs (pairwise log-ratio / SLR):** candidate (i,j) is `ln(x[s,i]/x[s,j])`. Reversing numerator and denominator reverses the sign. It does not change absolute correlation.

These formulas apply to positive OCU abundances after preprocessing; taking logarithms of raw zeros is undefined. Summing taxa into an OCU is different from taking their geometric mean. The multi-taxon balance used to generate the synthetic responses is not a claim that the search considers every possible multi-taxon partition.

For each response, method and level, CompoRes chooses the candidate with the greatest absolute correlation to the response. Direction is relative to the stated log-ratio and its numerator/denominator. Selected scatter plots and fitted lines describe the same data used for selection; independent validation is needed for predictive claims.

## Statistical interpretation

**Empirical permutation estimate.** Each shuffle breaks microbiome–response alignment according to `SHUFFLE` and repeats the best-balance search. For one response, method and OCU level, let `T_obs` be the observed maximum absolute correlation and `T_b` the shuffled maxima. With n shuffles and k values satisfying `T_b ≥ T_obs`, the empirical estimate is `(k + 1)/(n + 1)`. Output labels such as `bootstrap` and `p_value_bootstrap` refer to this permutation estimate, not ordinary resampling of observations with replacement.

This interpretation requires a valid null shuffling scheme and exchangeable samples under that null. Unrestricted shuffling may be inappropriate for repeated measures, paired samples, batches or other dependent designs. The metadata coloring file does not by itself address these issues. Ten shuffles give a minimum empirical p-value of 1/11; fifty give 1/51. These demo budgets illustrate operation, not precise tail estimation.

**GEV estimate.** CompoRes Fisher-transforms best absolute correlations with `arctanh`, fits a generalized extreme value distribution to the transformed shuffled maxima, and estimates the upper-tail probability `1 − F_GEV(arctanh(T_obs))`. This is a model-based extrapolation, not additional empirical permutation evidence. GEV tail accuracy depends on fit adequacy and shuffle count; a small demo sample of maxima cannot substantiate extremely small tail probabilities. Inspect fit diagnostics where available, increase shuffles, and assess stability before drawing strong conclusions. Handling exact correlations of one requires numerical safeguards in the implementation.

GEV p-values are floored at `1e-11` (maximum `−ln(p)` about 25.33). A value at the floor is capped, not evidence that its true probability equals `1e-11`. The code also holds experimental alternatives for this tail case, which cannot be selected in the configuration; see the [reference](docs/reference.md#p-value-estimation-and-interpretation).

**Multiplicity scope.** Repeating the maximum search in each shuffle accounts for selection among candidate balances for that response, method and level, conditional on the pipeline and valid null. It does not, by itself, establish family-wise or false-discovery control across responses, OCU levels, CLR versus pairs, taxon traces, datasets or later choices of which results to report. The response association rankings do not apply an across-response correction. Define the inferential family and use a justified analysis of valid p-values if formal error control is needed.

**Aggregate scores.** The response ranking is `S = (1/L) × Σ_l [−ln(p_l)]` over L processed levels, separately for empirical and GEV estimates. The levels share data and overlapping taxa, so their evidence is dependent. S is a ranking statistic, **not a calibrated global p-value**; `exp(−S)` is a geometric mean of level-wise estimates, not a combined significance test. Agreement between methods and a large ranking gap may guide follow-up but do not supply a controlled significance threshold.

OTU tracing further distributes and averages balance scores across taxa and levels. Files historically labelled `estimated_p_value` contain `exp(−score)`; these are **not calibrated OTU-level or global p-values**, and the best-balance permutation calibration must not be assumed to transfer to every traced balance or taxon. See the [tutorial](docs/tutorial.md#5-per-taxon-ranking) and [reference](docs/reference.md#otu-wise-cumulative-p-value-tracing-and-analysis).

## Runtime, resuming and reproducibility

More levels, responses, candidate pairs and shuffles increase work. Pairwise searches grow quadratically with OCU count. For an initial check, use `--preprocess` or the quick synthetic configuration; increase `OCU_SAMPLING_RATE` to reduce levels and respect `MAX_OCU` for pairs. CompoRes parallelizes core work across available CPUs and processes responses in batches of 100. See [resource examples and runtime tips](docs/reference.md#computational-requirements); the examples are observations, not minimum requirements or resource guarantees.

Runs resume from saved state and skip completed steps, including on repeat demo runs. A rapid repeat may only report stored results. To rerun from scratch, preserve old results and select a fresh output destination, or remove the intended run's output directory after backing it up. Do not assume changed data or configuration automatically invalidates cached steps.

For an auditable analysis, save the exact command, effective configuration and resolved paths, input files or checksums, sample/taxon filtering counts, repository commit or package version, environment/dependency versions, shuffle settings, completed cycle count, logs and state files. Record whether results were resumed and the hardware/elapsed runtime. Shuffles are random; exact scores may differ between fresh runs. There is currently no CLI or configuration option for setting a random seed. `report.html` records the CompoRes version and when it was written; save the other metadata above yourself where it is not captured in outputs. Ranking stability across cycles is useful but does not validate the null model or GEV tail calibration.

## Advanced features and documentation

```bash
# Add directional OTU tracing; omit --response to trace all responses
compores --config /path/to/my_config.yaml --include_otu_trace_analysis --response response_1

# Synthetic binary classification analysis; omit --response to select the top-ranked response
compores --config /path/to/my_config.yaml --include_classification_power_analysis --response response_1
```

The `response_{index}` prefix is sufficient for selection. OTU tracing supports CLR, condensed pairs and OTU-pair matrices. Synthetic power analysis regenerates responses from fitted models and reruns CompoRes; it can be much slower and describes model-based synthetic performance, not independent clinical or biological validation.

- [Synthetic tutorial](docs/tutorial.md): both demo configurations, all five output walkthroughs, figures and numerical examples.
- [Reference](docs/reference.md): complete installation/uninstallation, resource table, preprocessing/imputation, clustering, naming conventions, advanced tracing, classification constants and SLURM examples, and logging.

To run the automated tests, run `pytest` from the repository root. It runs the fast tests, the end-to-end quick test `tests/test_quick_test_run.py` included, as CI does for pushes and pull requests to `main` that change the code. The optional, much slower power-analysis test runs only with `pytest -m slow`, and in CI only when the Slow Tests workflow is started manually.

## License

CompoRes is distributed under the Yissum Software Research License, for non-commercial research use only; see [LICENSE](LICENSE). For other uses, contact Yissum at software@yissum.co.il.
