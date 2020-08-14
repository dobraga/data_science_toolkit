from sklearn.base import TransformerMixin


class BasePreproc(TransformerMixin):
    def __init__(self):
        super().__init__()
        self.fitted = False

    def fit(self, X, **fit_params):
        r = self._fit(X, **fit_params)
        self.fitted = True
        return r

