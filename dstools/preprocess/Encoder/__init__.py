from ..utils import BasePreproc
import numpy as np
import pandas as pd


class OrdinalEncoder(BasePreproc):
    def __init__(self, dict_ordinal: dict, nan_value: str = "NAN"):
        """
        ordinal_cols = {
            'BsmtCond':['Ex','Gd','TA','Fa','Po','NAN'],
            'MSZoning':['A','C (all)','FV','I','RH','RL','RP','RM', 'NAN'],
            'Street':['Grvl','Pave']
        }
        """
        self.nan_value = nan_value
        self.mapping_ = dict_ordinal

    def _transform_value(self, value: str, values: list) -> int:
        value = self.nan_value if pd.isna(value) else value
        return int(np.argwhere([value == obj for obj in values]))

    def transform_col(self, X: pd.DataFrame, col: str) -> pd.DataFrame:
        values = self.mapping_[col]
        return X[col].apply(lambda value: self._transform_value(value, values))

    def _fit(self, X: pd.DataFrame):
        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        for col in self.mapping_.keys():

            if col in X.columns:
                try:
                    X[col] = self.transform_col(X, col)
                except Exception as e:
                    print(
                        "The column {} was not converted, verify dict [{}]".format(
                            col, e
                        )
                    )

            else:
                print("The column {} was not found in dataset".format(col))

        return X
