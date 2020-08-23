from sklearn.preprocessing import PowerTransformer
from ..utils import BasePreproc
import pandas as pd
import numpy as np
import re


class TransformNewColumn(BasePreproc):
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

    def __init__(self, mapping={}):
        super().__init__()
        self.mapping = mapping

    def _fit(self, X, **fit_params):
        self.env_ = {}
        if "env" in fit_params.keys():
            self.env_ = fit_params["env"]
        return self

    def add(self, mapping={}):
        self.mapping = dict(self.mapping, **mapping)

    def _transform(self, X):
        env = dict(locals(), **self.env_)

        for namecol, command in self.mapping.items():
            X[namecol] = eval(command, env)

        return X


class TransformBinary(BasePreproc):
    """
    This class wil be used to create a new binarized columns using a most relevant value
    
    :to_bin: Force binarize columns
    
    :auto_binary: Search columns whith rate greater than :threshold_min to binarize
    :drop: Drop columns with a rate of class greater than :threshold_max and original columns
    """

    def __init__(
        self,
        drop=True,
        auto_binary=True,
        threshold_min=0.5,
        threshold_max=0.95,
        to_bin=[],
    ):
        super().__init__()
        self._drop = drop
        self.to_bin = to_bin
        self.auto_binary = auto_binary
        self._threshold_min = threshold_min
        self._threshold_max = threshold_max

    def _fit(self, X):
        self.cols_bin_ = []
        self.cols_drop_ = []
        self._to_bin_ = {}

        for col in X.select_dtypes(include="object").columns:
            value_counts = X[col].value_counts(normalize=True)
            most_freq = value_counts.index[0]
            value = value_counts[most_freq]

            if self.auto_binary:
                if (value > self._threshold_min) and (value < self._threshold_max):
                    self._to_bin_[col] = most_freq
                    self.cols_bin_.append("bin_" + col + "_" + str(most_freq))

            if col in self.to_bin:
                self._to_bin_[col] = most_freq
                self.cols_bin_.append("bin_" + col + "_" + str(most_freq))

            if value >= self._threshold_max:
                self.cols_drop_.append(col)
        return self

    def _transform(self, X):
        for col in self._to_bin_.keys():
            most_freq = self._to_bin_[col]
            X["bin_" + col + "_" + str(most_freq)] = X[col].apply(
                lambda x: 1 if x == most_freq else 0
            )

        if self._drop:
            X = X.drop(columns=self.cols_drop_ + list(self._to_bin_.keys()))

        return X


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

    def __init__(self, mapping=None, numeric_input=np.min, not_input=["target"]):
        if not mapping:
            Exception
        super().__init__()
        self.mapping = mapping
        self.numeric_input = numeric_input
        self.not_input = not_input

    def _fit(self, X):
        self.mapping_ = self.mapping
        for col in X._get_numeric_data().columns:
            if (col not in self.mapping_.keys()) and (col not in self.not_input):
                self.mapping_[col] = self.numeric_input
        return self

    def _transform(self, X):
        mapping = self.mapping_.copy()
        tuples = []

        for col in mapping.keys():
            if callable(mapping[col]):
                mapping[col] = mapping[col](X[col])

            elif isinstance(mapping[col], tuple):
                group = mapping[col][0]
                func = mapping[col][1]
                X[col] = X.groupby(group)[col].transform(lambda x: x.fillna(func(x)))
                tuples.append(col)

        for tup in tuples:
            del mapping[tup]

        return X.fillna(mapping)


class TransformOthers(BasePreproc):
    """
    This class will be used to transform very granular fields
    
    :threshold: Define threshold to transform the value 'Others'
    :not_transform_cols: Not transform this columns
    """

    def __init__(self, threshold=0.01, not_transform_cols=[]):
        super().__init__()
        self.threshold = threshold
        self.not_transform_cols = not_transform_cols

    def _fit(self, X):
        self.relevant_values_ = {}

        for col in X.select_dtypes(include=["object"]):
            if col not in self.not_transform_cols:
                aux = X.groupby(col)[[col]].count() / len(X)
                aux = aux >= self.threshold

                if aux[col].max:
                    self.relevant_values_[col] = list(aux[aux[col]].index)
        return self

    def _transform(self, X):
        for col in self.relevant_values_.keys():
            X.loc[~X[col].isin(self.relevant_values_[col]), col] = "Others"

        return X


class TransformColumn(BasePreproc):
    """
    This class will be used to standardize the transformations
    
    mapping = {
        'SalePrice': np.log1p
    }
    
    tc = TransformColumn(mapping)
    
    tc.add({'PoolArea': np.log1p})
    """

    def __init__(self, mapping={}):
        super().__init__()
        self.mapping_ = mapping

    def add(self, mapping):
        self.mapping_ = {**self.mapping_, **mapping}

    def _fit(self, X):
        return self

    def _transform(self, X):
        for col, func in self.mapping_.items():
            X[col] = func(X[col])

        return X


class TransformPower(BasePreproc):
    def __init__(self, not_transform="Target"):
        super().__init__()
        self.cols_not_transform = (
            not_transform if type(not_transform) == list else [not_transform]
        )
        self.pt = PowerTransformer()

    def _fit(self, X):
        self.cols_transform_ = []
        self.cols_transform_ = [
            col
            for col in X._get_numeric_data().columns
            if col not in self.cols_not_transform
        ]
        self.pt.fit(X[self.cols_transform_])
        return self

    def _transform(self, X):
        X[self.cols_transform_] = self.pt.transform(X[self.cols_transform_])
        return X
