import pickle
from sklearn.utils.validation import check_is_fitted
from .utils import BaseEstimator
from .Encoder import OrdinalEncoder
from .Transform import (
    TransformBinary,
    TransformColumn,
    TransformImputer,
    TransformNewColumn,
    TransformOthers,
    TransformPower,
)


class Pipeline(BaseEstimator):
    def __init__(self, steps=[]):
        self.steps = []

        if steps:
            self.append(steps)

    def __len__(self):
        return len(self.steps)

    def __getitem__(self, ind):
        return self.steps[ind]

    def transform(self, X):
        df = X.copy()

        for t in self.steps:
            df = t.transform(df)

        return df

    def append(self, steps):
        if not isinstance(steps, list):
            steps = [steps]

        for step in steps:
            check_is_fitted(step)
            self.steps.append(step)

    def save(self, file):
        with open(file, "wb") as f:
            pickle.dump(self.steps, f, pickle.HIGHEST_PROTOCOL)

    def load(self, file):
        with open(file, "rb") as f:
            self.steps = pickle.load(f)

