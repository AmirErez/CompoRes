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
