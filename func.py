import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import sigmaclip


def qualitative_graph(name, col, x_name, y_name):
    fig, (ax_bar, ax_pie) = plt.subplots(1, 2, constrained_layout=True,
                                         figsize=(10, 4))
    fig.suptitle(name, fontsize=20)
    value_c = col.value_counts()
    ax_bar.bar(value_c.index, value_c.values)
    ax_bar.set_xlabel(x_name)
    ax_bar.set_ylabel(y_name)
    ax_pie.pie(value_c.values, labels=value_c.index, autopct='%1.1f%%')
    plt.show()


def quantitative_graph(name, col, x_name, y_name):
    fig, (ax_hist, ax_moustache) = plt.subplots(1, 2, constrained_layout=True,
                                                figsize=(10, 4))
    fig.suptitle(name, fontsize=20)
    ax_hist.set_xlabel(x_name)
    ax_hist.set_ylabel(y_name)
    sns.histplot(data=col, kde=True, ax=ax_hist)
    sns.boxplot(data=col, ax=ax_moustache, orient='h')
    plt.show()


def repair_str(cat):
    return cat.replace({" ": None, "-": None}).replace({np.nan: ""}).apply(
        lambda x: x.capitalize()).replace({"": np.nan})


def quartile_meth(df_quant, num):

    Q1 = df_quant[num].quantile(0.25)
    Q3 = df_quant[num].quantile(0.75)

    iqr = Q3 - Q1
    bottom_bound_q = Q1 - 1.5 * iqr
    top_bound_q = Q3 + 1.5 * iqr

    df_quant = df_quant[~((df_quant[num] < bottom_bound_q) | (
            df_quant[num] > top_bound_q)).any(axis=1)]

    df_quant = df_quant.reset_index()
    df_quant.pop('index')
    return df_quant


def sigma_meth(df_sigma, num):
    for col in num:
        data = df_sigma[col].dropna()
        clean_data, low, high = sigmaclip(data, low=3, high=3)
        df_sigma = df_sigma.loc[
            (df_sigma[col].isin(clean_data)) | (df_sigma[col].isna())]

    df_sigma = df_sigma.reset_index()
    df_sigma.pop('index')
    return df_sigma
