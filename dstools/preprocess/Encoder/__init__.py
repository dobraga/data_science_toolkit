from ..utils import BasePreproc
import numpy as np
import pandas as pd


class OrdinalEncoder(BasePreproc):
    def __init__(self, dict_ordinal):
        """
        ordinal_cols = {
            'BsmtCond':['Ex','Gd','TA','Fa','Po','NAN'],
            'MSZoning':['A','C (all)','FV','I','RH','RL','RP','RM', 'NAN'],
            'Street':['Grvl','Pave']
        }
        """
        self.mapping_ = dict_ordinal

    def _transform_line(self, value):
        return int(
            np.argwhere(
                [("NAN" if pd.isna(value) else value) == obj for obj in self.values]
            )
        )

    def transform_col(self, X, col):
        self.values = self.mapping_[col]
        return X[col].apply(lambda row: self._transform_line(row))

    def _fit(self, X):
        return self

    def _transform(self, X):
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

    def inverse_transform(self, X):
        return True
