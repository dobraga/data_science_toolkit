from dstools.preprocess.utils import BasePreproc


from typing import Any, Dict
import pandas as pd
import numpy as np


class TransformImputer(BasePreproc):
    """
    This class will be used to input non observed data

    mapping = {
        'colA': 'ValueA',
        'colB': 0,
        'colC': np.mean,
        'LotFrontage': ('Neighborhood', np.mean)
    }
    """

    def __init__(
        self,
        mapping: Dict[str, Any] = {},
        auto_input: bool = False,
        numeric_input=np.mean,
    ):
        if not mapping:
            Exception
        super().__init__()
        self.mapping = mapping
        self.auto_input = auto_input
        self.numeric_input = numeric_input

    def _fit(self, X: pd.DataFrame, y: pd.DataFrame = None):
        self.mapping_ = self.mapping
        if self.auto_input:
            for col in X._get_numeric_data().columns:
                if col not in self.mapping_.keys():
                    self.mapping_[col] = self.numeric_input

        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        for col, transformer in self.mapping_.items():
            if callable(transformer):
                X[col] = X[col].fillna(transformer(X[col]))

            elif isinstance(transformer, tuple):
                group = transformer[0]
                func = transformer[1]
                X[col] = X.groupby(group)[col].transform(func)

            elif isinstance(transformer, str):
                X[col] = X[col].fillna(transformer)

        return X[self.mapping_.keys()]

    def get_feature_names(self, input_features=None):
        return list(self.mapping_.keys())
