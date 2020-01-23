from sklearn.preprocessing import PowerTransformer, OneHotEncoder
from sklearn.base import BaseEstimator
import pandas as pd
import numpy as np
import re

class TransformNewColumn(BaseEstimator):
    '''
        This class wil be used to create a new columns in dataset
        
        mapping = {'TotalBath': 'BsmtFullBath + 0.5*BsmtHalfBath + FullBath + 0.5*HalfBath'}
    '''
    def __init__(self,mapping = {}):
        self.mapping = mapping
        self.opperators = '[\+|\-|\*|\/|\%]+'
        self.mapping_commands = {}

    def fit(self):
        return True

    def _adjust_command(self, command):
        regxp_opperators = re.compile(self.opperators)
        variables = regxp_opperators.split(command)
        opp = regxp_opperators.findall(command)
        
        command = ''
        for i, var in enumerate(variables):
            var_adj = var.strip()
            if var_adj:
                if var_adj in self.columns:
                    command += 'X["' + var_adj + '"]'
                else:
                    command += var_adj

            if i < len(opp):
                command += opp[i]

        return command

    def transform(self, X):
        df = X.copy()

        self.columns = df.columns

        for namecol, command in self.mapping.items():
            command = self._adjust_command(command)

            df[namecol] = eval(command)
            
            self.mapping_commands[namecol] = command

        return df


class TransformBinary(BaseEstimator):
    '''
        This class wil be used to create a new binarized columns using a most relevant value
        
        :to_bin: Force binarize columns
        
        :auto_binary: Search columns whith rate greater than :threshold_min to binarize
        :drop: Drop columns with a rate of class greater than :threshold_max and original columns
    '''
    def __init__(self, drop = True, auto_binary = True, threshold_min = 0.5, threshold_max = 0.95, to_bin = []):
        self._drop = drop
        self.to_bin = to_bin
        self.auto_binary = auto_binary
        self._threshold_min = threshold_min
        self._threshold_max = threshold_max
        
        self.cols_bin = []
        self.cols_drop = []
        self._to_bin = {}

    def fit(self, X):
        for col in X.select_dtypes(include='object').columns:
            value_counts = X[col].value_counts(normalize=True)
            most_freq = value_counts.index[0]
            value = value_counts[most_freq]

            if self.auto_binary:
                if (value > self._threshold_min) and (value < self._threshold_max):
                    self._to_bin[col] = most_freq
                    self.cols_bin.append('bin_'+col+'_'+str(most_freq))

            if col in self.to_bin:
                self._to_bin[col] = most_freq
                self.cols_bin.append('bin_'+col+'_'+str(most_freq))

            if value >= self._threshold_max:
                self.cols_drop.append(col)

    def transform(self, X):
        df = X.copy()

        for col in self._to_bin.keys():
            most_freq = self._to_bin[col]
            df['bin_'+col+'_'+str(most_freq)] = df[col].apply(lambda x: 1 if x == most_freq else 0)

        if self._drop:
            df = df.drop(columns = self.cols_drop + list(self._to_bin.keys()))

        return df

    def fit_transform(self, X):
        self.fit(X)
        df = self.transform(X)
        return df


class TransformImputer(BaseEstimator):
    '''
        This class will be used to input non observed data
        
        mapping = {
            'colA': 'ValueA',
            'colB': 0,
            'colC': np.mean,
            'LotFrontage': ('Neighborhood', np.mean)
        }
    '''
    def __init__(self, mapping = None, numeric_input = np.min, not_input = ['target']):
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
    '''
        This class will be used to transform very granular fields
        
        :threshold: Define threshold to transform the value 'Others'
        :not_transform_cols: Not transform this columns
    '''
    def __init__(self, threshold = 0.01, not_transform_cols = []):
        self.threshold = threshold
        self.relevant_values = {}
        self.not_transform_cols = not_transform_cols

    def fit(self, X):
        for col in X.select_dtypes(include=['object']):
            if col not in self.not_transform_cols:
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
    '''
        This class will be used to standardize the transformations
        
        mapping = {
            'SalePrice': np.log1p
        }
        
        tc = TransformColumn(mapping)
        
        tc.add({'PoolArea': np.log1p})
    '''
    def __init__(self, mapping = {}):
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