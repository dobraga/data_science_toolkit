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

    def transform(self, X):
        df = X.copy()

        for t in self.transformations:
            df = t.transform(df)

        return df

    def add(self, transform):
        if transform.fitted:
            self.transformations.append(transform)
        else:
            raise "Transformation not fitted"
