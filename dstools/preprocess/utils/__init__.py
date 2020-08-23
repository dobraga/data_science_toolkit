from sklearn.base import TransformerMixin, BaseEstimator


class BasePreproc(TransformerMixin, BaseEstimator):
    def __init__(self):
        super().__init__()

    def fit(self, X, **fit_params):
        return self._fit(X, **fit_params)

    def transform(self, X):
        return self._transform(X.copy())
