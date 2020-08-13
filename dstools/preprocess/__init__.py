from .Encoder import OrdinalEncoder
from .Transform import (
    TransformBinary,
    TransformColumn,
    TransformImputer,
    TransformNewColumn,
    TransformOthers,
    TransformPower,
)


class Pipeline:
    def __init__(self):
        self.transformations = []
        self.is_fitted = None

    def _is_fitted(self):
        if len(self.transformations) > 0:
            fitted = [t.fitted for t in self.transformations]
            self.is_fitted = min(fitted)

    def fit(self, X):
        return True

    def transform(self, X):
        df = X.copy()

        for t in self.transformations:
            df = t.transform(df)

        return df

    def add(self, transform):
        self.transformations.append(transform)
        self._is_fitted()
