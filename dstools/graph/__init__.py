import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def _distplot_group(df, column="MSZoning", target="LogSalePrice"):
    types = df[column].drop_duplicates().values

    for typ in types:
        if pd.isnull(typ):
            subset = df[df[column].isnull()]

        else:
            subset = df[df[column] == typ]

        sns.distplot(
            subset[target],
            hist=False,
            kde=True,
            kde_kws={"shade": True, "linewidth": 1},
            label=typ,
        )

    plt.legend(prop={"size": 16}, title=column)
    plt.title("Density Plot with Multiple {}".format(column))
    plt.xlabel(target)
    plt.ylabel("Density")


def distplots_group(df, columns=["MSZoning"], target="LogSalePrice"):
    for i, col in enumerate(columns):
        plt.figure(i)
        _distplot_group(df, column=col, target=target)

    plt.show()
