import argparse

from .compores_main import ComporesMain, CONFIG_FILE_PATH

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""`CompoRes` is an algorithmic package designed to analyze
    correlation between changes in the host's microbiota and the host's gene expression. It addresses two
    significant analysis challenges: compositional statistics in the microbiota and multiple hypothesis
    testing, which can be a source of spurious correlations between microbiota and host.""")
    parser.add_argument(
        '--config',
        nargs='?',
        type=str,
        default=CONFIG_FILE_PATH,
        help='The path to the config file; defaults to test `config.yaml`.'
    )
    parser.add_argument(
        '--include_otu_trace_analysis',
        nargs='?',
        type=bool,
        default=False,
        help='True for a run with further OTU p-value tracing and analysis, False without it.'
    )
    parser.add_argument(
        '--include_classification_power_analysis',
        nargs='?',
        type=bool,
        default=False,
        help='True for a run with further derived synthetic response generation and classification, False without it.'
    )
    parser.add_argument(
        '--response',
        nargs='?',
        type=str,
        default=None,
        help='The response variable to apply synthetic power analysis to.'
    )
    parser.add_argument(
        '--no_plotting',
        action='store_true',
        help='Do not plot the results.'
    )
    parser.add_argument(
        '--ocu_case',
        nargs='?',
        type=int,
        default=None,
        help='The OCU case to analyze; used for Synthetic Power Analysis.'
    )
    args = parser.parse_args()

    runner = ComporesMain(args.config, ocu_case=args.ocu_case)
    if args.no_plotting:
        runner.switch_off_plotting()
    try:
        runner.run()
    finally:
        runner.close()

    if args.include_otu_trace_analysis:
        runner.generate_otu_p_value_summary_data(args.response)

    if args.include_classification_power_analysis:
        runner.add_synthetic_data_analysis(args.response)
