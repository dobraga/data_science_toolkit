from dstools.preprocess.utils import BasePreproc

import pandas as pd
import numpy as np


class TransformColumn(BasePreproc):
    """
    This class wil be used to create a new columns or just transform columns in dataset

    tnc = TransformNewColumn({
        'TotalBath': 'X["BsmtFullBath"] + 0.5*X["BsmtHalfBath"] + X["FullBath"] + 0.5*X["HalfBath"]',
        'Test': 'X["SalePrice"].apply(log1p)'
    })

    To use non-basic functions like `to_numeric`

    tnc.fit(df, env={"log1p": np.log1p})

    tnc.transform(X)

    Because `to_numeric` is not defined in this class
    """

    def __init__(self, mapping: dict = {}):
        super().__init__()
        self.mapping = mapping

    def _fit(self, X: pd.DataFrame, y: pd.DataFrame, **fit_params):
        self.env_ = {}
        if "env" in fit_params.keys():
            self.env_ = fit_params["env"]
        return self

    def add(self, mapping: dict):
        self.mapping = dict(self.mapping, **mapping)

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        env = dict(locals(), **self.env_)

        new_X = np.zeros((X.shape[0], len(self.mapping)))

        for i, (_, command) in enumerate(self.mapping.items()):
            command = command.replace("\n", "").replace("\t", "")
            new_X[:, i] = eval(command, env)

        return new_X

    def get_feature_names(self, input_features=None):
        return []  # list(self.mapping.keys())
