from collections import namedtuple
import pandas as pd
import polars as pl

import config
import utils

TabularDataSet = namedtuple(
    "DataSet", ["X", "y", "variables", "numeric_vars", "categorical_vars"]
)


def load_adult(subsample_frac: float=1.0):
    df = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        header=None,
        sep=r'\s*,\s*', 
        engine='python',
        na_values="?"
    )
    df.columns = [
        "Age",
        "WorkClass",
        "fnlwgt",
        "Education",
        "EducationNum",
        "MaritalStatus",
        "Occupation",
        "Relationship",
        "Race",
        "Gender",
        "CapitalGain",
        "CapitalLoss",
        "HoursPerWeek",
        "NativeCountry",
        "Income",
    ]

    label_var = 'Income'
    categorical_vars = [
        'WorkClass', 'Education', 'EducationNum', 'MaritalStatus', 'Occupation', 
        'Relationship', 'Race', 'Gender', 'NativeCountry'
    ]
    numeric_vars = df.columns.drop(categorical_vars + [label_var]).to_list()

    labels = df[label_var].values
    cat = pd.get_dummies(df[categorical_vars])
    categorical_vars = cat.columns.to_list()
    df = pd.concat([df[numeric_vars].astype(float), cat], axis=1)

    variables = numeric_vars + categorical_vars

    if subsample_frac < 1.0:
        df = df.sample(frac=subsample_frac)
    X = df[variables].values
    y = (labels == '>50K').astype(int)

    data = TabularDataSet(X, y, variables, numeric_vars, categorical_vars)
    return data



def get_physionet2012_stats():
    features = config.PHYSIONET2012_FEATURES
    group_vars = ["variable"]
    value_var = "value"
    stats = (
        pl.read_parquet(
            config.PHYSIONET2012_TIMESERIES, ["RecordID"] + features, use_pyarrow=True
        )
        .lazy()
        .pipe(utils.wide2tall, value_vars=features, id_vars=["RecordID"])
        .groupby(group_vars)
        .agg(
            utils.calculate_stats_aggs(value_var)
            + utils.calculate_quantiles_aggs(value_var)
        )
        .with_columns((pl.col("Q75") - pl.col("Q25")).round(4).alias("IQR"))
        .rename({"variable": "Variable"})
        .sort("Variable")
        .collect()
    )
    return stats


def load_physionet2012():
    features = config.PHYSIONET2012_FEATURES
    df = (
        pl.read_parquet(
            config.PHYSIONET2012_TIMESERIES, ["RecordID"] + features, use_pyarrow=True
        )
        .lazy()
        .groupby(["RecordID"])
        .agg([utils.finite(f).mean() for f in features])
        .collect()
    ).to_numpy()[:, 1:]

    return df