import numpy as np
import pandas as pd

def corr(df, columns = None):
    if columns:
        df_corr = df[columns].corr().stack().reset_index()
    else:
        df_corr = df._get_numeric_data().corr().stack().reset_index()
    df_corr = df_corr[df_corr['level_0'] != df_corr['level_1']]
    df_corr['abs'] = np.abs(df_corr[0])
    df_corr = df_corr.sort_values('abs', ascending = False)
    df_aux = df_corr.copy()

    cond = df_corr['level_0'] > df_corr['level_1']
    df_corr.loc[cond, 'level_1'] = df_aux.loc[cond, 'level_0']
    df_corr.loc[cond, 'level_0'] = df_aux.loc[cond, 'level_1']
    return df_corr.drop_duplicates().reset_index(drop = True).drop(columns = 'abs')

