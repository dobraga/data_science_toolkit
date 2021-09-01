import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.compose import TransformedTargetRegressor, ColumnTransformer

from dstools.dataset import load_house_prices
from dstools.preprocess import (
    OrdinalEncoder,
    TransformColumn,
    TransformImputer,
    TransformOthers,
)


def test_pipeline():
    train, _ = load_house_prices()

    X_train, X_test, y_train, y_test = train_test_split(
        train.drop(columns="SalePrice"), train["SalePrice"], train_size=0.75
    )

    ordinal = {
        "Street": ["Grvl", "Pave"],
        "LotShape": ["Reg", "IR1", "IR2", "IR3"],
        "Utilities": ["AllPub", "NoSewr", "NoSeWa", "ELO", "NAN"],
        "LandSlope": ["Gtl", "Mod", "Sev"],
        "BldgType": ["1Fam", "2fmCon", "Duplex", "Twnhs", "TwnhsE"],
        "ExterQual": ["Ex", "Gd", "TA", "Fa", "Po"],
        "ExterCond": ["Ex", "Gd", "TA", "Fa", "Po"],
        "BsmtQual": ["Ex", "Gd", "TA", "Fa", "Po", "NAN"],
        "BsmtCond": ["Ex", "Gd", "TA", "Fa", "Po", "NAN"],
        "BsmtExposure": ["Gd", "Av", "Mn", "No", "NAN"],
        "BsmtFinType1": ["GLQ", "ALQ", "BLQ", "Rec", "LwQ", "Unf", "NAN"],
        "BsmtFinType2": ["GLQ", "ALQ", "BLQ", "Rec", "LwQ", "Unf", "NAN"],
        "HeatingQC": ["Ex", "Gd", "TA", "Fa", "Po"],
        "CentralAir": ["N", "Y"],
        "KitchenQual": ["Ex", "Gd", "TA", "Fa", "Po", "NAN"],
        "FireplaceQu": ["Ex", "Gd", "TA", "Fa", "Po", "NAN"],
        "GarageFinish": ["Fin", "RFn", "Unf", "NAN"],
        "GarageQual": ["Ex", "Gd", "TA", "Fa", "Po", "NAN"],
        "GarageCond": ["Ex", "Gd", "TA", "Fa", "Po", "NAN"],
        "PavedDrive": ["Y", "P", "N"],
        "PoolQC": ["Ex", "Gd", "TA", "Fa", "NAN"],
        "Fence": ["GdPrv", "MnPrv", "GdWo", "MnWw", "NAN"],
    }

    # new_cols = {
    #     "TotalBath": 'X["BsmtFullBath"] + 0.5*X["BsmtHalfBath"] + X["FullBath"] + 0.5*X["HalfBath"]',
    #     "TotalSqrFootage": 'X["BsmtFinSF1"] + X["BsmtFinSF2"] + X["1stFlrSF"] + X["2ndFlrSF"]',
    #     "TotalPorch": 'X["OpenPorchSF"] + X["3SsnPorch"] + X["EnclosedPorch"] + X["ScreenPorch"] + X["WoodDeckSF"]',
    # }

    nominal = ["MSZoning", "Alley", "LandContour", "LotConfig", "Neighborhood"]
    nominal += ["Condition1", "Condition2", "HouseStyle", "RoofStyle", "RoofMatl"]
    nominal += ["Exterior1st", "Exterior2nd", "MasVnrType", "Foundation", "Heating"]
    nominal += ["Electrical", "Functional", "GarageType", "MiscFeature", "SaleType"]
    nominal += ["SaleCondition"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("ordinal", OrdinalEncoder(ordinal), list(ordinal.keys())),
            (
                "nominal",
                make_pipeline(TransformOthers(threshold=0.1), OneHotEncoder()),
                nominal,
            ),
            # ("new_cols", TransformColumn(new_cols), list(new_cols.keys())),
        ]
    )

    reg = Pipeline(
        [("preprocessor", preprocessor), ("regressor", DecisionTreeRegressor())]
    )

    treg = TransformedTargetRegressor(reg, func=np.log1p, inverse_func=np.expm1)

    treg.fit(X_train, y_train).score(X_test, y_test)
