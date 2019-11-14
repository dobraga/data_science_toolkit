import pandas as pd
from os.path import dirname, join

module_path = dirname(__file__)

def load_house_prices():
    """https://www.kaggle.com/c/house-prices-advanced-regression-techniques"""
    
    train = join(module_path, 'data', 'house_prices/train.csv').drop(columns = 'Id')
    test = join(module_path, 'data', 'house_prices/train.csv')
        
    return train, test
