class OutlierCheckFailed(Exception):
    def __init__(self, message="Outlier check failed."):
        super().__init__(message)


class DuplicatedIndices(Exception):
    def __init__(self, message="Duplicated sample tag names", file_name=None):
        if file_name:
            message = (
                f"{message}: provide unique sample identifiers in `{file_name}` or add the `--deduplicate True` "
                "argument to automate deduplication (the first duplicated sample tag will be used)."
            )
        super().__init__(message)


class NoResponseLabelFound(Exception):
    def __init__(self, message="No response label found in the response index."):
        super().__init__(message)


class MisMatchFiles(Exception):
    def __init__(self, message="The input files have mismatching rows."):
        super().__init__(message)


class NonNumericDataFrameError(Exception):
    def __init__(self, message="The input data contains non-numeric values."):
        super().__init__(message)


class NegativeValuesDataFrameError(Exception):
    def __init__(self, message="The input microbiome data contains negative values."):
        super().__init__(message)


class EmptyDataFrame(Exception):
    def __init__(self, message="The resulting DataFrame is empty."):
        super().__init__(message)


class MinDataFrame(Exception):
    def __init__(
            self,
            message="The resulting DataFrame should have at least 3 rows and 3 columns for OTU, "
                    "and at least 3 rows and 1 column for response.",
            file_name=None
    ):
        if file_name:
            message = f"{message}: provide a valid file in {file_name}"

        super().__init__(message)
