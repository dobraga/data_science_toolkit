from dstools.preprocess.utils import BasePreproc

import pandas as pd


class TransformOthers(BasePreproc):
    """
    This class will be used to transform very granular fields

    :threshold: Define threshold to transform the value 'Others'
    :not_transform_cols: Not transform this columns
    """

    def __init__(self, threshold: float = 0.01):
        super().__init__()
        self.threshold = threshold

    def _fit(self, X: pd.DataFrame, y: pd.DataFrame = None):
        self.relevant_values_ = {}

        for col in X.select_dtypes(include=["object"]):
            aux = X[col].value_counts(normalize=True, dropna=False)
            aux = aux >= self.threshold

            if any(aux):
                self.relevant_values_[col] = list(aux[aux].index)

        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        for col, relevant_values in self.relevant_values_.items():
            X.loc[~X[col].isin(relevant_values), col] = "Others"

        return X[self.relevant_values_.keys()]

    def get_feature_names(self, input_features=None):
        return list(self.relevant_values_.keys())
