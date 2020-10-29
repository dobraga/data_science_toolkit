import dill
import pandas as pd
from sklearn.utils.validation import check_is_fitted
from .utils import BaseEstimator
from .Encoder import OrdinalEncoder
from .Transform import (
    TransformBinary,
    TransformColumn,
    TransformImputer,
    TransformOthers,
)


class Pipeline(BaseEstimator):
    def __init__(self, pre_process: list = [], model=None, pos_process: list = []):
        self.pre_process = []
        self.model = None
        self.pos_process = []

        if pre_process:
            self.append(pre_process, "pre_process")
        if model:
            self.append(model, "model")
        if pos_process:
            self.append(pos_process, "pos_process")

    def __len__(self):
        return len(self.pre_process) + len(self.pos_process)

    def __getitem__(self, ind, type: str = "pre_process"):
        if type == "pre_process":
            return self.pre_process[ind]
        elif type == "pos_process":
            return self.pos_process[ind]

    def __call__(self, X: pd.DataFrame) -> tuple:
        return self.transform(X)

    def transform(self, X: pd.DataFrame) -> tuple:
        df = X.copy()
        y = None

        for s in self.pre_process:
            if hasattr(s, "__call__"):
                df = s(df)
            else:
                df = s.transform(df)

        if self.model:
            y = self.model.predict(df)

        if self.pos_process:
            for f in self.pos_process:
                y = f(y)

        return df, y

    def append(self, steps, type: str = "pre_process"):
        if type in ["pre_process", "pos_process"]:
            if not isinstance(steps, list):
                steps = [steps]

            for step in steps:
                if type == "pre_process":
                    if not hasattr(step, "__call__"):
                        check_is_fitted(step)
                    self.pre_process.append(step)

                else:
                    assert hasattr(step, "__call__"), f"{step} is a not callable"
                    self.pos_process.append(step)

        elif type == "model":
            assert hasattr(steps, "predict"), f"{steps} not have a predict method"
            self.model = steps

        return self

    def save(self, file: str):
        with open(file, "wb") as f:
            dill.dump(self.__dict__, f)

    @staticmethod
    def load(file: str):
        with open(file, "rb") as f:
            pip = Pipeline()
            pip.__dict__.update(dill.load(f))
        return pip
