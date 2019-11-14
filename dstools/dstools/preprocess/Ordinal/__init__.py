from sklearn.base import BaseEstimator
import numpy as np
import pandas as pd

class OrdinalEncoder(BaseEstimator):
    def __init__(self, dict_ordinal):
        '''
        ordinal_cols = {
            'BsmtCond':['Ex','Gd','TA','Fa','Po','NAN'],
            'MSZoning':['A','C (all)','FV','I','RH','RL','RP','RM', 'NAN'],
            'Street':['Grvl','Pave']
        }
        '''
        self.ordinal = dict_ordinal
            
    def _transform(self, value):
        return int(np.argwhere([('NAN' if pd.isna(value) else value) == obj for obj in self.values]))
    
    def transform_col(self, df, col):
        self.values = self.ordinal[col]
        return df[col].apply(lambda row: self._transform(row))

    def fit(self):
        return True
    
    def transform(self, X):
        df = X.copy()

        for col in self.ordinal.keys():

            if col in df.columns:
                try:
                    df[col] = self.transform_col(df, col)
                except Exception as e:
                    print('The column {} was not converted, verify dict [{}]'.format(col, e))
                    
            else:
                print('The column {} was not found in dataset'.format(col))

        return df

    def inverse_transform(self, df):
        return True
