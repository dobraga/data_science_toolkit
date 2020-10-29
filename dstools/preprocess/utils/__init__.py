from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd


class BasePreproc(TransformerMixin, BaseEstimator):
    def __init__(self):
        super().__init__()

    def fit(self, X: pd.DataFrame, **fit_params):
        return self._fit(X, **fit_params)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._transform(X.copy())
