import numpy as np
import pandas as pd

class Corr:
    def __init__(self, df):
        df_corr = df._get_numeric_data().corr().stack().reset_index()
        df_corr = df_corr[df_corr['level_0'] != df_corr['level_1']]
        df_corr['abs'] = np.abs(df_corr[0])
        df_corr = df_corr.sort_values('abs', ascending = False)
        df_aux = df_corr.copy()

        cond = df_corr['level_0'] > df_corr['level_1']
        df_corr.loc[cond, 'level_1'] = df_aux.loc[cond, 'level_0']
        df_corr.loc[cond, 'level_0'] = df_aux.loc[cond, 'level_1']

        self.df_corr = df_corr.drop_duplicates().reset_index(drop = True).drop(columns = 'abs')
        self.df_corr.columns = ['var1', 'var2', 'corr']

    def lowest_correlation(self, var = 'target', threshold = .9):
        col_var = 'lowest_correlation_'+var
        self.df_corr[col_var] = ''

        for i in range(self.df_corr.shape[0]):
            line = self.df_corr.iloc[i,:]
            
            if ((line.var1 != var) & (line.var2 != var)):
            
                cor1 = self.df_corr.loc[(self.df_corr['var1'].isin([line.var1, var]) & self.df_corr['var2'].isin([line.var1, var])), 'corr'].values
                cor2 = self.df_corr.loc[(self.df_corr['var1'].isin([line.var2, var]) & self.df_corr['var2'].isin([line.var2, var])), 'corr'].values

                self.df_corr.loc[i, col_var] = line.var1 if np.absolute(cor1) < np.absolute(cor2) else line.var2

        return list(set(self.df_corr.loc[self.df_corr['corr'] >= .9, col_var]))