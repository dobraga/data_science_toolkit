from dstools.preprocess.utils import BasePreproc

import pandas as pd
import numpy as np


class TransformBinary(BasePreproc):
    """
    This class wil be used to create a new binarized columns using a most relevant value

    Parameters
    ----------

    to_bin: Force binarize columns

    auto_binary: Search columns with rate greater than :threshold_min to binarize

    drop: Drop columns with a rate of class greater than :threshold_max and original columns
    """

    def __init__(self, threshold_min=0.5, threshold_max=0.95):
        super().__init__()
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max

    def _fit(self, X: pd.DataFrame, y: pd.DataFrame = None):
        self.cols_bin_ = []
        self._to_bin_ = {}

        for col in X.select_dtypes(include="object").columns:
            value_counts = X[col].value_counts(normalize=True)
            most_freq = value_counts.index[0]
            value = value_counts[most_freq]

            if self.threshold_min < value < self.threshold_max:
                self._to_bin_[col] = most_freq
                self.cols_bin_.append("bin_" + col + "_" + str(most_freq))

        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        new_X = np.zeros((X.shape[0], len(self._to_bin_)))

        for i, (col, most_freq) in enumerate(self._to_bin_.items()):
            new_X[:, i] = X[col].apply(lambda x: 1 if x == most_freq else 0)

        return new_X

    def get_feature_names(self, input_features=None):
        return self.cols_bin_


if __name__ == "__main__":
    TransformBinary()
