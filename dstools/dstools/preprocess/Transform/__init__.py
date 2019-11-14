from sklearn.preprocessing import PowerTransformer, OneHotEncoder
from sklearn.base import BaseEstimator
import pandas as pd
import numpy as np
import re

class TransformNewColumn(BaseEstimator):
    def __init__(self,mapping = {}):
        '''
        mapping = {'TotalBath': 'BsmtFullBath + 0.5*BsmtHalfBath + FullBath + 0.5*HalfBath'}
        '''
        self.mapping = mapping
        self.opperators = '\\+|\\*'

    def fit(self):
        return True

    def _adjust_command(self, command):
        command = command.replace(' ','')
        reg1 = re.compile(self.opperators)
        for column in reg1.split(command):
            if column in self.columns:
                reg = re.compile('(^{col}({op}))|(({op}){col}({op}))|((({op}){col}))$'.format(col = column, op = self.opperators))
                position = reg.search(command)
                if position:
                    position = list(position.span())
                    position[0] = position[0] + 1 if position[0] > 0 else position[0]
                    position[1] = len(command) + 1 if position[1] >= len(command) else position[1]
                    command = command[:position[0]] + "df['{col}']".format(col = column) + command[position[1]-1:]

        return command

    def transform(self, X):
        df = X.copy()

        self.columns = df.columns

        for namecol, command in self.mapping.items():
            command = self._adjust_command(command)

            df[namecol] = eval(command)

        return df


class TransformBinary(BaseEstimator):
    def __init__(self, to_bin = None, drop = True, auto_binary = True, threshold_min = 0.5, threshold_max = 0.95):
        if not to_bin and not auto_binary:
            Exception

        self._drop = drop
        self._auto_binary = auto_binary
        self._threshold_min = threshold_min
        self._threshold_max = threshold_max
        
        self.cols_bin = []

        if to_bin:
            self._to_bin = to_bin
        else:
            self._to_bin = [] 

        self._to_drop = []

    def fit(self, X):
        if self._auto_binary:
            for col in X._get_numeric_data().columns:
                value = (X[col] == 0).mean()

                if (value > self._threshold_min) and (value < self._threshold_max):
                    self._to_bin.append(col)
                    self.cols_bin.append('bin_'+col)

                elif value >= self._threshold_max:
                    self._to_drop.append(col)

    def transform(self, X):
        df = X.copy()

        for col in self._to_bin:
            df['bin_'+col] = df[col].apply(lambda x: 1 if x>0 else 0)

        if self._drop:
            df = df.drop(columns = self._to_bin + self._to_drop)

        return df

    def fit_transform(self, X):
        self.fit(X)
        df = self.transform(X)
        return df


class TransformImputer(BaseEstimator):
    def __init__(self, mapping = None, numeric_input = np.min, not_input = ['target']):
        '''
        mapping = {
            'colA': 'ValueA',
            'colB': 0,
            'colC': np.mean,
            'LotFrontage': ('Neighborhood', np.mean)
        }
        '''

        if not mapping:
            Exception

        self.mapping = mapping
        self.numeric_input = numeric_input
        self.not_input = not_input

    def fit(self, X):
        for col in X._get_numeric_data().columns:
            if (col not in self.mapping.keys()) and (col not in self.not_input):
                self.mapping[col] = self.numeric_input

    def transform(self, X):
        df = X.copy()
        mapping = self.mapping.copy()
        tuples = []

        for col in mapping.keys():
            if callable(mapping[col]):
                mapping[col] = mapping[col](df[col])

            elif isinstance(mapping[col], tuple):
                group = mapping[col][0]
                func = mapping[col][1]
                df[col] = df.groupby(group)[col].transform(lambda x: x.fillna(func(x)))
                tuples.append(col)
                
        for tup in tuples:
            del mapping[tup]

        return df.fillna(mapping)

    def fit_transform(self, X):
        self.fit(X)
        df = self.transform(X)

        return df


class TransformOthers(BaseEstimator):
    def __init__(self, threshold = 0.01, not_use_cols = []):
        self.threshold = threshold
        self.relevant_values = {}
        self.not_use_cols = not_use_cols

    def fit(self, X):
        for col in X.select_dtypes(include=['object']):
            if col not in self.not_use_cols:
                aux = X.groupby(col)[[col]].count()/len(X)
                aux = aux>=self.threshold

                if aux[col].max:
                    self.relevant_values[col] = list(aux[aux[col]].index)

    def transform(self, X):
        df = X.copy()

        for col in self.relevant_values.keys():
            df.loc[~df[col].isin(self.relevant_values[col]), col] = 'Others'

        return df

    def fit_transform(self, X):
        self.fit(X)
        df = self.transform(X)

        return df


class TransformColumn(BaseEstimator):
    def __init__(self, mapping = {}):
        '''
        mapping = {
            'SalePrice': np.log1p
        }
        '''

        self.mapping = mapping

    def add(self, mapping):
        self.mapping = {**self.mapping, **mapping}
        
    def fit(self, X):
        return True

    def transform(self, X):
        df = X.copy()

        for col, func in self.mapping.items():
            df[col] = func(df[col])

        return df
    
    
class TransformPower(BaseEstimator):
    def __init__(self, not_transform = 'Target'):
        self.cols_not_transform = not_transform if type(not_transform) == list else [not_transform]
        self.cols_transform = []
        self.pt = PowerTransformer()
        
    def fit(self, X):
        self.cols_transform = [col for col in X._get_numeric_data().columns if col not in self.cols_not_transform]
        self.pt.fit(X[self.cols_transform])
        return True
    
    def transform(self, X):
        df = X.copy()
        df[self.cols_transform] = self.pt.transform(df[self.cols_transform])
        
        return df
    
    def fit_transform(self, X):
        self.fit(X)
        df = self.transform(X)

        return df
    
    
class TransformOneHot(BaseEstimator):
    def __init__(self)
        