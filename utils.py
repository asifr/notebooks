from typing import (
    List,
    Union,
    TypeVar,
    Any,
    Dict,
    Optional,
    cast,
    Sequence,
    Callable,
    Tuple,
)
import os
import io
import sys
import math
import shutil
import warnings
from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

T = TypeVar("T")
PathLike = TypeVar("PathLike", str, Path, None)


def is_sequence(obj):
    return isinstance(obj, Sequence) and not isinstance(obj, (str, bytes))


def is_listish(obj):
    """Check if something quacks like a list."""
    if isinstance(obj, (list, tuple, set)):
        return True
    return is_sequence(obj)


def ensure_list(obj):
    """Make the returned object a list, otherwise wrap as single item."""
    if obj is None:
        return []
    if is_listish(obj):
        return [o for o in cast(Sequence[T], obj)]
    return [cast(T, obj)]


# ----------------------------------------------------------------------
# Reading and writing PyArrow datasets


def read(
    file: PathLike,
    columns: List[str] = None,
    filter: ds.Expression = None,
    indices: Union[pa.Array, List[int]] = None,
    polars=False,
):
    """
    Read a Parquet dataset.

    Parameters
    ----------
    file : PathLike
    columns : List[str]
    filter : pyarrow.dataset.Expression, optional
        Use predicate pushdown to filter the dataset. Use pyarrow.Field
        expressions to refer to columns by name and compose conditional
        statements.
    indices : Union[pa.Array, List[int]]
        Filter rows by indices.
    polars : bool, optional
        If True, return a Polars DataFrame. If False, return a PyArrow table.

    Returns
    -------
    table : pa.Table
    """
    columns = ensure_list(columns)
    dataset = ds.dataset(file)
    if indices is not None:
        if not isinstance(indices, pa.Array):
            indices = pa.array(indices)
        table = dataset.take(indices, columns=columns, filter=filter)
    else:
        table = dataset.to_table(columns=columns, filter=filter)

    if polars:
        return pl.from_arrow(table)
    else:
        return table


def write(
    file: PathLike,
    table: Union[pa.Table, pd.DataFrame],
    column_names: List[str] = None,
    partition_cols: Optional[List[str]] = None,
    create_metadata=False,
    **kwargs,
) -> None:
    """
    Write a PyArrow Table, Pandas DataFrame, or numpy array to parquet dataset.

    Writing metadata on partitioned dataset (i.e. with `partition_cols`) is
    not supported by PyArrow because the row group schema changes across
    partitioned Parquet files.
    See issue: https://issues.apache.org/jira/browse/ARROW-13269

    When `index_cols` is given, the `_index.pkl` file will be created. This
    is useful for reading specific rows of the dataset back in. The index
    is a dictionary of keys and values, where keys are string/int
    (or tuples for multi-column indexing) and values are numpy arrays of
    indices: `{1: np.array([0,2])}` for single column index and
    `{(1,2): np.array([0,2])}` for a multi-column index.

    Set `data_page_size=268435456` to store 256MB row groups.

    Set `row_group_size=10000` to store 10000 rows per row group.

    Storing data in sorted order can improve data access and predicate
    evaluation performance. Sort table columns based on the likelihood of
    their occurrence in query predicates; columns that most frequently occur
    in comparison or range predicates should be sorted first.

    Parameters
    ----------
    file : PathLike
    table : pyarrow.Table or pandas.DataFrame
    column_names : List[str], optional
        Column names if the table is a numpy ndarray
    partition_cols : List[str], optional
        Parttion the dataset by the specified columns. PyArrow does not
        support metadata on partitioned datasets so a warning will be raised.
    **kwargs : dict
        Additional kwargs for write_table function. See docstring for
        `write_table` or `ParquetWriter` for more information.
        https://arrow.apache.org/docs/python/generated/pyarrow.parquet.write_table.html
    """
    metadata_collector = []
    # pandas.DataFrame to pyarrow.Table
    if isinstance(table, pd.DataFrame):
        table = pa.Table.from_pandas(table)
    # numpy.ndarray to pyarrow.Table
    if isinstance(table, np.ndarray):
        if table.ndim == 1:
            table = table[:, np.newaxis]
        elif table.ndim > 2:
            raise ValueError("Only 1D or 2D arrays are supported")
        table = numpy_to_table(table, column_names)
    # pyarrow.Array to pyarrow.Table
    if isinstance(table, pa.Array):
        table = pa.Table.from_arrays(
            [table], column_names if column_names is not None else ["0"]
        )
    # delete existing dataset and create new one
    delete(file)
    pq.write_to_dataset(
        table,
        root_path=file,
        partition_cols=partition_cols,
        metadata_collector=metadata_collector,
        **kwargs,
    )
    if create_metadata:
        # The _common_metadata file doesn't hold any information about file
        # (no row group info) but just the schema
        pq.write_metadata(table.schema, file / "_common_metadata")
        # The _metadata file holds information about row groups
        if partition_cols is None:
            pq.write_metadata(
                table.schema, file / "_metadata", metadata_collector=metadata_collector
            )
        else:
            warnings.warn(
                "Writing metadata on partitioned dataset is not supported by"
                "PyArrow. Results in a schema mismatch"
                "(https://issues.apache.org/jira/browse/ARROW-13269).",
                RuntimeWarning,
            )


def delete(file: PathLike) -> None:
    """
    Delete a file or directory.

    Parameters
    ----------
    name : str
    """
    if os.path.isdir(file):
        shutil.rmtree(file)
    else:
        if os.path.exists(file):
            os.remove(file)


def numpy_to_table(X: np.ndarray, names: Optional[List[str]] = None) -> pa.Table:
    """
    Convert a numpy array to a Parquet table.

    Parameters
    ----------
    X : np.ndarray
    names : List[str], optional
        Column names, defaults to a list of integers starting at 0

    Returns
    -------
    table : pa.Table
    """
    n_cols = X.shape[1]
    if names is None:
        names = [str(i) for i in range(n_cols)]
    arr = [pa.array(X[:, i]) for i in range(n_cols)]
    table = pa.Table.from_arrays(arr, names=names)
    return table


def table_to_numpy(table: pa.Table) -> np.ndarray:
    """
    Convert a Parquet table to a numpy array.

    Parameters
    ----------
    table : pa.Table or pl.DataFrame

    Returns
    -------
    np.ndarray
    """
    if safe_isinstance(table, "polars.internals.frame.DataFrame"):
        return table.to_numpy()

    arr = np.zeros((table.num_rows, table.num_columns))
    for i, col in enumerate(table.columns):
        arr[:, i] = col.to_numpy()
    return arr


def pyarrow_to_dict(arr: pa.Array):
    """
    Cast a pyarrow array to dict.

    Parameters
    ----------
    arr : pyarrow.Array
    """
    return arr.cast(arr.type).dictionary_encode()


def table_columns(file: PathLike) -> List[str]:
    """
    Returns the columns of a Parquet dataset.

    Parameters
    ----------
    name : str

    Returns
    -------
    columns : List[str]
    """
    dataset = pq.ParquetDataset(file)
    return dataset.schema.names


def table_index(
    file: PathLike,
    columns: List[str] = None,
    filter: ds.Expression = None,
    indices: Union[pa.Array, List[int]] = None,
) -> Dict[Any, np.ndarray]:
    """
    Read a Parquet dataset and return a dictionary mapping ids to row indices.

    Parameters
    ----------
    file : PathLike
    columns : List[str], optional
    filter : ds.Expression, optional
    indices : pa.Array, optional

    Returns
    -------
    groups : Dict[str, Union[List[int], np.ndarray, None]]
        Mapping of unique ids to row indices
    """
    if columns is None:
        raise ValueError("indexing columns must be specified")
    columns = ensure_list(columns)
    df = read(file, columns=columns, filter=filter, indices=indices).to_pandas()
    return df.groupby(columns).indices


# ----------------------------------------------------------------------
# Reading and writing data


def fread(filename):
    with io.open(filename, "r", encoding="utf8") as f:
        return f.read()


def fwrite(filename, data):
    with io.open(filename, "w", encoding="utf8") as f:
        f.write(data)


def to_json(filename, data):
    fwrite(filename, json.dumps(data, indent=2))


def from_json(filename, default=[]):
    if os.path.exists(filename):
        with open(filename) as f:
            data = json.load(f)
    else:
        data = default
    return data


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        from pathlib import WindowsPath, PosixPath, PurePath

        # Pandas dataframes have a to_json() method
        if hasattr(obj, "to_json"):
            return json.loads(obj.to_json())
        # Pathlib objects to a string representation
        if isinstance(obj, (WindowsPath, PosixPath)) or isinstance(obj, PurePath):
            return str(obj)
        return json.JSONEncoder.default(self, obj)


def save_pickle(obj, path: PathLike):
    """
    Save a Python object as a pickle file.

    Parameters
    ----------
    obj : dictionary or dataclass or namedtuple
    path : PathLike

    Notes
    -----
    This function will make some transformation on the object based on the type
    before saving it. This is to ensure that data can be loaded back with as few
    pre-requisite modules as possible (e.g. pandas or dataclasses). For example,
    pandas dataframes will be converted to a dictionary and dataclasses will 
    also be converted to a dictionary. You don't need to import pandas or define
    the dataclass before loading the data.

    - np.ndarray will be saved as a numpy array
    - pd.DataFrame and pd.Series will be saved as a dictionary
    - any object with a to_numpy() method will be saved as a numpy array
    - dataclasses will be saved as a dictionary
    - namedtuples will be saved as a dictionary
    - torch modules and optimizers will be save the state_dict(), if there is a 
      cpu() method, it will be called to ensure the state is on the CPU
    """
    # obj is a namedtuple
    if hasattr(obj, "_asdict"):
        obj = obj._asdict()
    # obj is a dataclass
    if hasattr(obj, "__dataclass_fields__"):
        obj = obj.__dict__

    new_obj = {}
    for key, value in obj.items():
        # numpy array
        if safe_isinstance(value, "numpy.ndarray"):
            new_obj[key] = value
        # pandas dataframe
        elif safe_isinstance(value, "pandas.core.frame.DataFrame") | safe_isinstance(value, "pandas.core.series.Series") | hasattr(obj, "to_dict"):
            new_obj[key] = value.to_dict()
        # has a to_numpy method
        elif hasattr(value, "to_numpy"):
            new_obj[key] = value.to_numpy()
        # dataclasses
        elif hasattr(value, "__dataclass_fields__"):
            new_obj[key] = value.__dict__
        # namedtuples
        elif hasattr(value, "_asdict"):
            new_obj[key] = value._asdict()
        # torch model or optimizer
        elif hasattr(value, "state_dict"):
            new_obj[key] = value.cpu().state_dict() if hasattr(value, "cpu") else value.state_dict()
        else:
            new_obj[key] = value

    with open(path, "wb") as f:
        pickle.dump(new_obj, f, protocol=pickle.HIGHEST_PROTOCOL)


# ----------------------------------------------------------------------
# Pandas Dataframe utilities


def impute(
    df: pd.DataFrame,
    method: str,
    reference_lower: pd.Series = None,
    reference_upper: pd.Series = None,
    impute_values: pd.Series = None,
):
    """Apply imputation to a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Wide table with missing values
    method : str, optional
        Imputation methods: mean, median, mode, constant, midpoint, sample
    reference_lower : pd.Series, optional
        Lower reference range, required for midpoint and sample imputation,
        by default None
    reference_upper : pd.Series, optional
        Upper reference range, required for midpoint and sample imputation,
        by default None
    impute_values : pd.Series, optional
        Imputation values for mean, median, or constant method, by default None

    Returns
    -------
    df : pd.DataFrame
        Dataframe with imputed values, inplace operations modify the original
        dataframe
    """
    if method == "mean":
        if impute_values is None:
            impute_values = df.mean()
        df.fillna(impute_values, inplace=True)
    if method == "median":
        if impute_values is None:
            impute_values = df.median()
        df.fillna(impute_values, inplace=True)
    if method == "mode":
        if impute_values is None:
            impute_values = df.mode().iloc[0]
        df.fillna(impute_values, inplace=True)
    if method == "constant":
        if impute_values is None:
            impute_values = 0
        df.fillna(impute_values, inplace=True)
    if (method == "midpoint") | (method == "sample"):
        assert reference_lower is not None, "reference_lower must be provided"
        assert reference_upper is not None, "reference_upper must be provided"
    if method == "midpoint":
        df.fillna(
            (reference_lower.loc[df.columns] + reference_upper.loc[df.columns]) / 2,
            inplace=True,
        )
    if method == "sample":

        def _sample_imputation(series, lower, upper):
            """Impute missing values by sampling from a normal distribution
            centered at the midpoint of the reference range and with standard
            deviation (upper-lower) * 0.15.
            """
            missing = np.isnan(series)
            N = missing.sum()
            if (lower == upper) | np.isnan(upper):
                series[missing] = lower
            else:
                mid = (lower + upper) / 2.0
                sd = (upper - lower) * 0.15
                vals = mid + np.random.randn(N) * sd
                series[missing] = vals
            return series

        for col in df.columns:
            assert (
                col in reference_lower.index and col in reference_upper.index
            ), f"{col} is missing from reference_lower or reference_upper"
            df[col] = _sample_imputation(
                df[col], reference_lower[col], reference_upper[col]
            )

    return df


def inner_join(left, right, on):
    return left.merge(right, left_on=on, right_on=on, how="inner")


def outer_join(left, right, on):
    return left.merge(right, left_on=on, right_on=on, how="outer")


def left_join(left, right, on):
    return left.merge(right, left_on=on, right_on=on, how="left")


def forward_fill_imputation(wide, feature_cols, id_col, time_col, time_suffix="_t"):
    """
    Forward fill values and create a time column for each feature.

    Parameters
    ----------
    wide : pd.DataFrame
        Wide table with missing values
    feature_cols : List[str]
        List of feature columns
    id_col : str
        Id column
    time_col : str
        Time column
    time_suffix : str, optional
        Suffix for time column, by default "_t"

    Returns
    -------
    wide : pd.DataFrame
        Wide table with imputed values and time columns
    """
    tvec = wide.loc[:, time_col].values
    times = (~wide.loc[:, feature_cols].isna()).astype(float)
    times[times == 0] = np.nan
    times = times * tvec[:, None]
    times.loc[:, id_col] = wide.loc[:, id_col]
    times.loc[:, feature_cols] = (
        times.loc[:, [id_col] + feature_cols].groupby(id_col).ffill()
    )
    times = times.loc[:, feature_cols].add_suffix(time_suffix)
    wide.loc[:, feature_cols] = (
        wide.loc[:, [id_col] + feature_cols].groupby(id_col).ffill()
    )
    wide = pd.concat([wide, times], axis=1)
    return wide


class DataFrameSummary:
    def __init__(self, file, id_col=None, time_col=None, feature_names=None) -> None:
        self.file = file
        self.columns = []
        self.stats = {}
        self.indices = {}
        self.id_col = id_col
        self.time_col = time_col
        self.feature_names = feature_names
        self.categorical_columns = {}
        self.numeric_columns = {}

    def describe(self):
        self.get_columns()
        if self.id_col is not None:
            self.calculate_group_indices(self.id_col)
        self.calculate_stats()

    def get_columns(self):
        self.columns = table_columns(self.file)

    def calculate_stats(self):
        print("Calculating stats")
        q = [
            0,
            1,
            2.5,
            5,
            10,
            20,
            25,
            30,
            40,
            50,
            60,
            70,
            75,
            80,
            90,
            95,
            97.5,
            99,
            100,
        ]
        n = len(self.columns)
        for i, col in enumerate(self.columns):
            print(f"{i}/{n}: {col}")
            x = read(self.file, col)[col].to_numpy().ravel()
            xfinite = x[~np.isnan(x)]
            self.stats[col] = {
                "mean": np.mean(xfinite),
                "std": np.std(xfinite),
                "min": np.min(xfinite),
                "max": np.max(xfinite),
                "n": len(x),
                "missing": np.isnan(x).sum(),
                "n_unique": len(np.unique(xfinite)),
                "value_counts": pd.Series(xfinite).value_counts().to_dict(),
            }
            self.stats[col]["percentiles"] = dict(zip(q, np.percentile(xfinite, q)))
            self.stats[col]["histogram_bin_edges_5"] = np.histogram_bin_edges(
                xfinite, bins=5
            )
            self.stats[col]["histogram_bin_edges_10"] = np.histogram_bin_edges(
                xfinite, bins=10
            )

    def calculate_group_indices(self, index_col):
        print("Summarizing group indices")
        self.indices = table_index(self.file, index_col)

    def save(self, file):
        """
        Save file description to pickle
        """
        out = {
            "id_col": self.id_col,
            "time_col": self.time_col,
            "columns": self.columns,
            "stats": self.stats,
            "indices": self.indices,
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "feature_names": self.feature_names,
        }
        with open(file, "wb") as f:
            pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, file):
        """
        Load file description from pickle
        """
        with open(file, "rb") as f:
            data = pickle.load(f)
            self.id_col = data["id_col"]
            self.time_col = data["time_col"]
            self.columns = data["columns"]
            self.stats = data["stats"]
            self.indices = data["indices"]
            self.numeric_columns = data["numeric_columns"]
            self.categorical_columns = data["categorical_columns"]
            self.feature_names = data["feature_names"]


# ----------------------------------------------------------------------
# Numpy utilities


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance

    Parameters
    ----------
    seed : None, int or instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, int):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(
        "%r cannot be used to seed a numpy.random.RandomState instance" % seed
    )


def take(X, indices, *, axis=0):
    """Return rows, items or columns of X using indices.

    Parameters
    ----------
    X : array-like, sparse-matrix, list, pandas.DataFrame, pandas.Series
        Data from which to sample rows, items or columns. `list` are only
        supported when `axis=0`.
    indices : bool, int, str, slice, array-like
        - If `axis=0`, boolean and integer array-like, integer slice,
          and scalar integer are supported.
        - If `axis=1`:
            - to select a single column, `indices` can be of `int` type for
              all `X` types and `str` only for dataframe. The selected subset
              will be 1D, unless `X` is a sparse matrix in which case it will
              be 2D.
            - to select multiples columns, `indices` can be one of the
              following: `list`, `array`, `slice`. The type used in
              these containers can be one of the following: `int`, 'bool' and
              `str`. However, `str` is only supported when `X` is a dataframe.
              The selected subset will be 2D.
    axis : int, default=0
        The axis along which `X` will be subsampled. `axis=0` will select
        rows while `axis=1` will select columns.

    Returns
    -------
    subset : array-like, sparse-matrix, list, pandas.DataFrame, pandas.Series
        Subset of X on axis 0 or 1.
    """
    if indices is None:
        return X

    if axis not in (0, 1):
        raise ValueError(
            "'axis' should be either 0 (to index rows) or 1 (to index "
            " column). Got {} instead.".format(axis)
        )

    indices_dtype = None
    if isinstance(indices, str):
        indices_dtype = "str"
    elif isinstance(indices, (np.ndarray, list, tuple)):
        if len(indices) > 0:
            if isinstance(indices[0], str):
                indices_dtype = "str"

    if axis == 0 and indices_dtype == "str":
        raise ValueError("String indexing is not supported with 'axis=0'")

    if axis == 1 and X.ndim != 2:
        raise ValueError(
            "'X' should be a 2D NumPy array, 2D sparse matrix or pandas "
            "dataframe when indexing the columns (i.e. 'axis=1'). "
            "Got {} instead with {} dimension(s).".format(type(X), X.ndim)
        )

    if axis == 1 and indices_dtype == "str" and not hasattr(X, "loc"):
        raise ValueError(
            "Specifying the columns using strings is only supported for "
            "pandas DataFrames"
        )

    if hasattr(X, "iloc"):
        if indices_dtype != "str" and not (
            isinstance(indices, slice) or np.isscalar(indices)
        ):
            return X.take(indices, axis=axis)
        else:
            indexer = X.iloc if indices_dtype != "str" else X.loc
            return indexer[:, indices] if axis else indexer[indices]
    elif hasattr(X, "shape"):
        return X[indices] if axis == 0 else X[:, indices]
    else:
        return [X[idx] for idx in indices]


def pad_sequences(
    sequences, maxlen=None, dtype="float32", padding="pre", truncating="pre", value=0.0
):
    if not hasattr(sequences, "__len__"):
        raise ValueError("`sequences` must be iterable.")
    lengths = []
    for x in sequences:
        if not hasattr(x, "__len__"):
            raise ValueError(
                "`sequences` must be a list of iterables. "
                "Found non-iterable: " + str(x)
            )
        lengths.append(len(x))

    num_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((num_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == "pre":
            trunc = s[-maxlen:]
        elif truncating == "post":
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" ' "not understood" % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError(
                "Shape of sample %s of sequence at position %s "
                "is different from expected shape %s"
                % (trunc.shape[1:], idx, sample_shape)
            )

        if padding == "post":
            x[idx, : len(trunc)] = trunc
        elif padding == "pre":
            x[idx, -len(trunc) :] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


def path_to_string(path):
    """
    Convert `PathLike` objects to their string representation.
    If given a non-string typed path object, converts it to its string
    representation.

    If the object passed to `path` is not among the above, then it is
    returned unchanged. This allows e.g. passthrough of file objects
    through this function.

    Parameters
    ----------
    path : PathLike
        `PathLike` object that represents a path

    Returns
    -------
    path : str
        A string representation of the path argument, if Python support exists.
    """
    if isinstance(path, os.PathLike):
        return os.fspath(path)
    return path


def safe_isinstance(obj, class_path_str):
    """
    Acts as a safe version of isinstance without having to explicitly
    import packages which may not exist in the users environment.
    Checks if obj is an instance of type specified by class_path_str.

    Parameters
    ----------
    obj: Any
        Some object you want to test against
    class_path_str: str or list
        A string or list of strings specifying full class paths
        Example: `sklearn.ensemble.RandomForestRegressor`

    Returns
    --------
    bool : True if isinstance is true and the package exists, False otherwise
    """
    if isinstance(class_path_str, str):
        class_path_strs = [class_path_str]
    elif isinstance(class_path_str, list) or isinstance(class_path_str, tuple):
        class_path_strs = class_path_str
    else:
        class_path_strs = [""]

    # try each module path in order
    for class_path_str in class_path_strs:
        if "." not in class_path_str:
            raise ValueError(
                "class_path_str must be a string or list of strings specifying a full \
                module path to a class. Eg, 'sklearn.ensemble.RandomForestRegressor'"
            )

        # Splits on last occurence of "."
        module_name, class_name = class_path_str.rsplit(".", 1)

        # here we don't check further if the model is not imported, since we shouldn't have
        # an object of that types passed to us if the model the type is from has never been
        # imported. (and we don't want to import lots of new modules for no reason)
        if module_name not in sys.modules:
            continue

        module = sys.modules[module_name]

        # Get class
        _class = getattr(module, class_name, None)

        if _class is None:
            continue

        if isinstance(obj, _class):
            return True

    return False


def to_numpy(
    X: Union[np.ndarray, pl.DataFrame, pd.DataFrame, pa.Table, pl.Series, pd.Series]
) -> np.ndarray:
    """
    Transform an input to a numpy array.
    """
    if safe_isinstance(X, "numpy.ndarray"):
        return X
    if hasattr(X, "to_numpy"):
        return X.to_numpy()
    if safe_isinstance(X, "pandas.core.frame.DataFrame"):
        return X.values
    if safe_isinstance(X, "pyarrow.lib.Table"):
        return table_to_numpy(X)
    if safe_isinstance(X, "polars.internals.frame.DataFrame"):
        return X.to_numpy()
    return X


# ----------------------------------------------------------------------
# Polars utilities


def forward_fill(df, group_col, sort_by=None):
    if safe_isinstance(df, "pyarrow.lib.Table"):
        df = pl.from_arrow(df)

    dx = (
        df.lazy()
        .sort(by=sort_by if sort_by is not None else [group_col])
        .groupby(group_col)
        .agg([pl.all().exclude(group_col).fill_null("forward")])
        .explode(pl.all().exclude(group_col))
        .collect()
    )
    return dx


def create_index(df: pl.DataFrame, var: str) -> Dict[str, pl.Series]:
    g = df[:, var].groupby(var).groups()
    return dict(zip(g[var], g.groups))


def finite(var: str) -> pl.Expr:
    return pl.col(var).filter(pl.col(var).is_not_null() & pl.col(var).is_not_nan())


def wide2tall(
    df: pl.DataFrame, value_vars: List[str], id_vars: List[str] = []
) -> pl.DataFrame:
    """
    Returns pl.DataFrame with columns `[variable, value]`
    """
    return df.melt(id_vars=id_vars, value_vars=value_vars).filter(
        pl.col("value").is_not_nan() & pl.col("value").is_not_null()
    )


def calculate_stats_aggs(
    value_var: List[str],
    digits=4,
) -> List[pl.Expr]:
    value = finite(value_var)
    aggs = [
        value.mean().alias("Mean").round(digits),
        value.min().alias("Min").round(digits),
        value.max().alias("Max").round(digits),
    ]

    return aggs


def calculate_quantiles_aggs(
    value_var: List[str], quantiles: List[float] = None
) -> List[pl.Expr]:
    if quantiles is None:
        quantiles = [
            0,
            1,
            2.5,
            5,
            10,
            20,
            25,
            30,
            40,
            50,
            60,
            70,
            75,
            80,
            90,
            95,
            97.5,
            99,
            100,
        ]

    value = finite(value_var)
    aggs = []
    for q in quantiles:
        aggs.append(value.quantile(q / 100).alias(f"Q{q}"))

    return aggs


def calculate_quantiles(
    df: pl.DataFrame,
    group_vars: List[str],
    value_var: List[str],
    quantiles: List[float] = None,
) -> pl.DataFrame:
    aggs = calculate_quantiles_aggs(group_vars, value_var, quantiles)
    return df.groupby(group_vars).agg(aggs)


def outlier_robust_sigmoidal_transform(
    X: np.ndarray, medians: np.ndarray, iqr: np.ndarray
):
    return 1 / (1 + (np.exp(-(X - medians) / (1.35 * iqr))))


def get_row_index(
    index: Dict[Any, pl.Series], values: List[Any], errors: str = "raise"
):
    """
    Lookup row numbers from an index and a list of values.
    """
    inds = []
    for v in values:
        if v not in index:
            if errors == "raise":
                raise ValueError(f"{v} not in index")
            elif errors == "ignore":
                continue
            else:
                raise ValueError(f"Unknown error handling: {errors}")
        else:
            inds.append(index[v])
    return pl.concat(inds)


def pipe(func: Callable, *args, **kwargs):
    return func(*args, **kwargs)


# ----------------------------------------------------------------------
# Plotting utilities


def fontsize(ax, fz=14):
    """Set fontsizes for title, axis labels, and ticklabels.

    Parameters
    ----------
    ax : axis
        matplotlib axis
    fz : int
        font size
    """
    for item in (
        [ax.title, ax.xaxis.label, ax.yaxis.label]
        + ax.get_xticklabels()
        + ax.get_yticklabels()
    ):
        item.set_fontsize(fz)


def labels(
    ax,
    title=None,
    subtitle=None,
    xlabel=None,
    ylabel=None,
    xticklabels=None,
    yticklabels=None,
    legend_title=None,
    legend_labels=None,
    legend_loc="upper right",
    fz=14,
    grid=True,
):
    """
    Assign titles and labels.

    xticklabels and legend_labels should be lists, all other args are strings.

    Parameters
    ----------
    ax :
        matplotlib axis
    title :
        figure title
    subtitle :
        figure subtitle
    xlabel :
        x-axis label
    ylabel :
        y-axis label
    xticklabels :
        list of labels for x-axis
    yticklabels :
        list of labels for y-axis
    legend_title :
        legend title
    legend_labels :
        list of legend labels
    legend_loc :
        Legend position: upper right|lower right|upper left|lower left
    fz :
        font size
    grid :
        boolean
    """
    if title is not None:
        if subtitle is None:
            ax.set_title(title)
        else:
            ax.set_suptitle(title, y=1, fontsize=fz + 2)
            ax.set_title(subtitle)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels)
    if legend_title is not None and legend_labels is not None:
        handles, ax_legend_labs = ax.get_legend_handles_labels()
        ax.legend(handles, legend_labels, title=legend_title, loc=legend_loc)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if grid:
        ax.grid(linestyle="dotted")
    fontsize(ax, fz)


def plotly_theme(
    fig,
    title=None,
    font_family="Arial",
    font_size=18,
    template="simple_white",
    ygrid=True,
    xgrid=True,
    grid=True,
    hline=None,
    vline=None,
    legend_title=None,
    xlabel=None,
    ylabel=None,
    hovermode="x unified",
    height=None,
    width=None,
):
    if hline is not None:
        fig = fig.add_shape(
            type="line",
            line_color="black",
            line_width=3,
            opacity=1,
            line_dash="solid",
            x0=0,
            x1=1,
            xref="paper",
            y0=hline,
            y1=hline,
            yref="y",
        )
    if vline is not None:
        fig = fig.add_shape(
            type="line",
            line_color="black",
            line_width=3,
            opacity=1,
            line_dash="solid",
            y0=0,
            y1=1,
            yref="paper",
            x0=hline,
            x1=vline,
            xref="y",
        )
    fig = fig.update_layout(
        template=template,
        title=title,
        font_family=font_family,
        font_size=font_size,
        legend=dict(
            title=legend_title,
            orientation="h",
            y=1,
            yanchor="bottom",
            x=0.5,
            xanchor="center",
        ),
        hovermode=hovermode,
        height=height,
        width=width,
    )
    if not grid:
        xgrid = ygrid = False
    fig = fig.update_yaxes(showgrid=ygrid, title=ylabel)
    fig = fig.update_xaxes(showgrid=xgrid, title=xlabel)
    return fig


def arrays_to_df(**kwargs):
    """
    Convert arrays to dataframes.

    Parameters
    ----------
    **kwargs :
        Key-value pairs of arrays to convert to dataframes. Arrays can be
        1D or 2D.

    Returns
    -------
    df :
        pandas dataframe
    """
    keys = list(kwargs.keys())
    x = kwargs[keys[0]]
    if x.ndim > 2:
        raise ValueError("arrays can only be 1D or 2D")
    if x.ndim == 2:
        ids = np.tile(np.arange(x.shape[-1]), (x.shape[0], 1))
    else:
        ids = np.arange(len(x))
    df = pd.DataFrame({k: v.flatten() for k, v in kwargs.items()})
    df.loc[:, "ids"] = ids.flatten()
    return df


# ----------------------------------------------------------------------
# CDF transformation


class CDFTransform:
    """
    CDF transformation to normalize data in range [0, 1]. Maps values to
    quantiles.

    NaNs are treated as missing values: disregarded in fit, and maintained
    in transform.

    Parameters
    ----------
    n_quantiles : int
        Number of quantiles to use in the transformation.
    """

    def __init__(self, n_quantiles=1000):
        self.n_quantiles = n_quantiles
        self._quantiles = []

    def fit(self, X):
        """
        Calculates quantiles for all columns in X.

        Fills in `self._quantiles` list with quantiles for each column in X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data used to calculate the quantiles.
        """
        self._quantiles = []
        if safe_isinstance(X, "pandas.core.frame.DataFrame"):
            X = X.values
        if safe_isinstance(X, "pyarrow.lib.Table"):
            X = table_to_numpy(X)
        if safe_isinstance(X, "polars.internals.frame.DataFrame"):
            X = X.to_numpy()
        n_samples, n_features = X.shape
        n_quantiles_ = max(1, min(self.n_quantiles, n_samples))
        references = np.linspace(0, 1, n_quantiles_, endpoint=True) * 100
        quantiles = []
        for col in X.T:
            quantiles.append(np.nanpercentile(col, references))
        quantiles = np.transpose(quantiles)
        quantiles = np.maximum.accumulate(quantiles)
        self._quantiles = quantiles

    def calculate_col_quantiles(self, col):
        """
        Calculates quantiles for a single column.

        Parameters
        ----------
        col : array-like, shape (n_samples,)
            A single column of data.
        """
        n_samples = len(col)
        n_quantiles_ = max(1, min(self.n_quantiles, n_samples))
        references = np.linspace(0, 1, n_quantiles_, endpoint=True) * 100
        q = np.maximum.accumulate(np.nanpercentile(col, references))
        return q

    def fit_parquet(self, parquet_file: PathLike, columns: List[str]):
        """
        Calculates quantiles for each column in the parquet file.

        Some parquet files can't be read directly into memory so we read one
        column at a time and calculate the quantiles.

        Parameters
        ----------
        parquet_file : PathLike
            Path to parquet file.
        columns : List[str]
            List of columns to calculate quantiles for.
        """
        references = np.linspace(0, 1, self.n_quantiles, endpoint=True) * 100
        quantiles = []
        for col in columns:
            expr = ds.field(col).is_valid()
            x = read(parquet_file, columns=[col], filter=expr)[col].to_numpy()
            quantiles.append(np.nanpercentile(x, references))
        quantiles = np.transpose(quantiles)
        quantiles = np.maximum.accumulate(quantiles)
        self._quantiles = quantiles

    def transform(self, X) -> np.ndarray:
        """
        Normalize values in the range [0,1].

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data to be transformed. Inputs can be a numpy array, pandas
            DataFrame, pyarrow Table, or polars DataFrame.

        Returns
        -------
        X_transformed : array-like, shape (n_samples, n_features)
            Transformed data.
        """
        assert self._quantiles is not None, "Must fit before transforming"
        X = to_numpy(X)
        assert (
            self._quantiles.shape[1] == X.shape[1]
        ), f"Expected {self._quantiles.shape[1]} features, got {X.shape[1]}"
        Xc = np.zeros(X.shape)
        _, n_features = X.shape
        references = np.linspace(0, 1, self.n_quantiles, endpoint=True)
        lower_bound_y, upper_bound_y = 0, 1
        for feature_idx in range(n_features):
            X_col = X[:, feature_idx].copy()
            quantiles = self._quantiles[:, feature_idx]
            lower_bound_x = quantiles[0]
            upper_bound_x = quantiles[-1]
            lower_bounds_idx = X_col == lower_bound_x
            upper_bounds_idx = X_col == upper_bound_x
            isfinite_mask = ~np.isnan(X_col)
            X_col_finite = X_col[isfinite_mask]
            if len(X_col_finite) > 0:
                X_col[isfinite_mask] = 0.5 * (
                    np.interp(X_col_finite, quantiles, references)
                    - np.interp(-X_col_finite, -quantiles[::-1], -references[::-1])
                )
            Xc[:, feature_idx] = X_col
            X_col[upper_bounds_idx] = upper_bound_y
            X_col[lower_bounds_idx] = lower_bound_y
        return Xc

    def digitize(self, X):
        assert self._quantiles is not None, "Must fit before transforming"
        X = to_numpy(X)
        assert (
            self._quantiles.shape[1] == X.shape[1]
        ), f"Expected {self._quantiles.shape[1]} features, got {X.shape[1]}"
        Xc = np.zeros(X.shape)
        _, n_features = X.shape
        for feature_idx in range(n_features):
            X_col = X[:, feature_idx].copy()
            missing = np.isnan(X_col)
            quantiles = self._quantiles[feature_idx]
            Xc[:, feature_idx] = np.digitize(X_col, quantiles, right=True) + 1
            Xc[missing, feature_idx] = np.nan
        return Xc

    def save(self, path):
        np.save(path, self._quantiles)

    def load(self, path):
        self._quantiles = np.load(path)


# ----------------------------------------------------------------------
# Quantile binning transformation


class QuantileBinningTransform:
    """
    Bin values into quantiles.

    NaNs are treated as missing values: disregarded in fit, and maintained
    in transform. Bins are offset by 1 so bins start at 1. Bin 0 is reserved
    for missing values.

    Parameters
    ----------
    n_quantiles : int
        Number of quantiles to use in the transformation.
    """

    def __init__(self, n_quantiles=21):
        self.n_quantiles = n_quantiles
        self._quantiles = []

    def fit(self, X):
        """
        Calculates quantiles for all columns in X.

        Fills in `self._quantiles` list with quantiles for each column in X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data used to calculate the quantiles.
        """
        self._quantiles = []
        X = to_numpy(X)
        n_samples, n_features = X.shape
        n_quantiles_ = max(1, min(self.n_quantiles, n_samples))
        references = np.linspace(0, 1, n_quantiles_, endpoint=True) * 100
        quantiles = []
        for col in X.T:
            quantiles.append(
                np.maximum.accumulate(np.unique(np.nanpercentile(col, references)))
            )
        self._quantiles = quantiles

    def fit_parquet(self, parquet_file: PathLike, columns: List[str]):
        """
        Calculates quantiles for each column in the parquet file.

        Some parquet files can't be read directly into memory so we read one
        column at a time and calculate the quantiles.

        Parameters
        ----------
        parquet_file : PathLike
            Path to parquet file.
        columns : List[str]
            List of columns to calculate quantiles for.
        """
        references = np.linspace(0, 1, self.n_quantiles, endpoint=True) * 100
        quantiles = []
        for col in columns:
            expr = ds.field(col).is_valid()
            col = read(parquet_file, columns=[col], filter=expr)[col].to_numpy()
            quantiles.append(
                np.maximum.accumulate(np.unique(np.nanpercentile(col, references)))
            )
        self._quantiles = quantiles

    def transform(self, X, monotonic=True) -> np.ndarray:
        """
        Normalize values in the range [0,1].

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data to be transformed. Inputs can be a numpy array, pandas
            DataFrame, pyarrow Table, or polars DataFrame.

        Returns
        -------
        X_transformed : array-like, shape (n_samples, n_features)
            Transformed data.
        """
        assert self._quantiles is not None, "Must fit before transforming"
        X = to_numpy(X)
        assert (
            len(self._quantiles) == X.shape[1]
        ), f"Expected {len(self._quantiles)} features, got {X.shape[1]}"
        missing = np.isnan(X)
        Xc = np.zeros(X.shape, dtype=np.int32)
        _, n_features = X.shape
        for feature_idx in range(n_features):
            X_col = X[:, feature_idx].copy()
            quantiles = self._quantiles[feature_idx]
            col = np.digitize(X_col, quantiles, right=False)
            # offset by 1 so bins start at 1 and bin 0 is reserved for
            # missing values
            col = col + 1
            Xc[:, feature_idx] = col

        # bins should be monotonically increasing with each feature column
        if monotonic:
            n_bins = np.cumsum(
                np.array(
                    [0]
                    + [len(self._quantiles[i]) + 1 for i in range(len(self._quantiles))]
                )
            )
            Xc = Xc + n_bins[:-1][np.newaxis, :]

        Xc[missing] = 0
        return Xc

    @property
    def n_bins(self):
        return [len(q) + 1 for q in self._quantiles]

    @property
    def vocab_size(self):
        return sum(self.n_bins) + 1

    def save(self, path):
        # save pickle
        with open(path, "wb") as f:
            pickle.dump(self._quantiles, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path):
        # load pickle
        with open(path, "rb") as f:
            self._quantiles = pickle.load(f)


def init_spark(appName="MyApp", memory=12):
    """
    This function assumes you already have SPARK_HOME and PYSPARK_SUBMIT_ARGS
    environment variables set. Requires apache-spark, pyspark, findspark.

    Install apache-spark on MacOS with homebrew: `brew install apache-spark`.
    Then set the SPARK_HOME environment variable:
    os.environ.setdefault('SPARK_HOME', '/usr/local/Cellar/apache-spark/3.2.0/libexec')
    """
    import os
    import findspark

    def _parse_master(pyspark_submit_args):
        sargs = pyspark_submit_args.split()
        for j, sarg in enumerate(sargs):
            if sarg == "--master":
                try:
                    return sargs[j + 1]
                except:
                    raise Exception("Could not parse master from PYSPARK_SUBMIT_ARGS")
        raise Exception("Could not parse master from PYSPARK_SUBMIT_ARGS")

    if "SPARK_HOME" not in os.environ:
        raise Exception("SPARK_HOME environment variable not set.")
    if "PYSPARK_SUBMIT_ARGS" not in os.environ:
        cmd = f"--master local[12] --driver-memory {memory}g --executor-memory {memory}g pyspark-shell"
        os.environ["PYSPARK_SUBMIT_ARGS"] = cmd
    if "PYSPARK_SUBMIT_ARGS" not in os.environ:
        raise Exception("PYSPARK_SUNBMIT_ARGS environment variable not set.")
    findspark.init(os.environ["SPARK_HOME"])
    spark_master = _parse_master(os.environ["PYSPARK_SUBMIT_ARGS"])

    from pyspark.sql import SparkSession  # type: ignore

    spark = SparkSession.builder.master(spark_master).appName(appName).getOrCreate()
    print(spark.sparkContext.uiWebUrl)
    return spark


def download_url(url: str, out: PathLike, filename: str = None) -> None:
    """
    Download a file from the URL

    Parameters
    ----------
    url : PathLike
    out : PathLike
        output folder
    filename : str
        optional output file or folder name
    """
    import urllib

    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(out, filename)
    os.makedirs(out, exist_ok=True)

    try:
        urllib.request.urlretrieve(url, fpath)
    except (urllib.error.URLError, IOError) as e:
        if url[:5] == "https":
            url = url.replace("https:", "http:")
            urllib.request.urlretrieve(url, fpath)


def unzip(file: PathLike, out: PathLike, password: str = None) -> None:
    """
    Unarchive a .tar.gz, .tar, or .zip file.

    Parameters
    ----------
    file : PathLike
    out : PathLike
        output folder
    """
    import zipfile
    import tarfile

    if file.endswith("tar.gz"):
        tar = tarfile.open(file, "r:gz")
        tar.extractall(path=out)
        tar.close()
    if file.endswith("tar"):
        tar = tarfile.open(file, "r:")
        tar.extractall(path=out)
        tar.close()
    if file.endswith("zip"):
        with zipfile.ZipFile(file, "r") as z:
            if password is not None:
                z.setpassword(password)
            z.extractall(out)


# ------------------------------------------------------------------------------
# Torch


def to_tensor(X, dtype="float", device="cpu"):
    if dtype == "float":
        dtype = torch.float
    elif dtype == "long":
        dtype = torch.long
    else:
        raise ValueError("dtype must be 'float' or 'long'")
    X = to_numpy(X)
    return torch.from_numpy(np.asarray(X)).type(dtype).to(device)


def torch_to_numpy(*args):
    return [x.cpu().detach().numpy() for x in args]

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def squeeze_last_dim(tensor):
    # (B, D, 1) => (B, D).
    if len(tensor.shape) == 3 and tensor.shape[-1] == 1:
        return tensor[..., 0]
    return tensor


def last_timestep(x: torch.Tensor, vlen: torch.Tensor):
    """Grab the last timestep given a set of signal lengths"""
    return x.gather(1, (vlen - 1).view(-1, 1)).squeeze(-1)


def flatten_by_signal_length(x: torch.Tensor, vlen: torch.Tensor):
    """Flatten data given a set of signal lengths"""
    return torch.cat([x[i, :v] for i, v in enumerate(vlen)])


def padding_indicators(X: torch.FloatTensor, seqlens: torch.LongTensor):
    B, _, T = X.size()
    # padding indicator, 1=non-padding, 0=padding
    P = torch.arange(T).unsqueeze(0).expand(B, -1)
    P = (P < seqlens.unsqueeze(-1)).float()
    return P


def create_causal_mask(timesteps):
    mask = np.triu(np.ones((timesteps, timesteps)), k=1)
    mask = (torch.from_numpy(mask) == 0).long()
    return mask


def create_variable_length_mask(input_lengths, maxlen):
    """
    Create a mask for a set of timeseries.
    """
    mask = (torch.arange(maxlen)[None, :] < input_lengths[:, None]).long()
    return mask


def times_to_lags(T):
    """(N x n_step) matrix of times -> (N x n_step) matrix of lags.
    First time is assumed to be zero.
    """
    assert T.ndim == 2, "T must be an (N x n_step) matrix"
    return np.c_[np.diff(T, axis=1), np.zeros(T.shape[0])]


def save_model(model, file):
    torch.save({"state_dict": model.state_dict()}, file)


def load_model(model, file, device="cpu", strict=False):
    model.load_state_dict(
        torch.load(file, map_location=torch.device(device))["state_dict"],
        strict=strict,
    )


def load_to_cpu(path):
    model = torch.load(path, map_location=lambda storage, loc: storage)
    model.cpu()
    return model


def randomly_mask_sequence(L, lm, masking_ratio):
    """
    Randomly create a boolean mask of length `L`, consisting of subsequences of
    average length lm, masking with 0s a `masking_ratio` proportion of the
    sequence L. The length of masking subsequences and intervals follow a
    geometric distribution.

    Parameters
    ----------
    L : int
        length of mask and sequence to be masked
    lm : int
        average length of masking subsequences (streaks of 0s)
    masking_ratio : int
        proportion of L to be masked

    Returns
    -------
    m : bool
        boolean numpy array intended to mask ('drop') with 0s a sequence of
        length L shape (L,)
    """
    keep_mask = np.ones(L, dtype=bool)
    # probability of each masking sequence stopping. parameter of geometric
    # distribution.
    p_m = 1 / lm
    # probability of each unmasked sequence stopping. parameter of geometric
    # distribution.
    p_u = p_m * masking_ratio / (1 - masking_ratio)
    p = [p_m, p_u]
    # Start in state 0 with masking_ratio probability
    # state 0 means masking, 1 means not masking
    state = int(np.random.rand() > masking_ratio)
    for i in range(L):
        # here it happens that state and masking value corresponding to state
        # are identical
        keep_mask[i] = state
        if np.random.rand() < p[state]:
            state = 1 - state
    return keep_mask


def augment(inputs, missing=None, row_prob: float = 0.75, column_prob: float = 0.75):
    """
    Randomly swap values in randomly selected columns. Respects missing values
    by only swapping observed values.

    Parameters
    ----------
    inputs : torch.FloatTensor
    missing : torch.BoolTensor
        binary indicator missing elements = True
    row_prob : float
        swap a fraction of rows, leaving rest untouched
    column_prob : float
        swap a fraction of columns, leaving rest untouched

    Returns
    -------
    augmented : torch.FloatTensor
        augmented tensor after rows and columns have been swapped
    corrupted : torch.FloatTensor
        binary indicator tensor where swapped elements = 1
    """
    observed = 1 * ~(torch.isnan(inputs) if missing is None else missing)
    augmented = torch.clone(inputs)
    corrupted = torch.zeros(inputs.size())
    B, D = inputs.size()
    columns = torch.where(1 * (torch.rand(D) < column_prob))[0]
    for d in columns:
        corrupt = 1 * (torch.rand(B) < row_prob)
        corrupted[:, d] = corrupt & observed[:, d]
        inds = torch.where(corrupted[:, d])[0]
        shuffled_inds = inds[torch.randperm(inds.size(0))]
        augmented[inds, d] = inputs[shuffled_inds, d]
    return augmented, corrupted


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = (
            eta_min
            + (lr - eta_min) * (1 + math.cos(math.pi * epoch / args.max_epochs)) / 2
        )
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / (
            args.warm_epochs * total_batches
        )
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_baseline_prediction(y: np.ndarray):
    proba_positive_class = np.mean(y)
    eps = np.finfo(np.float64).eps
    proba_positive_class = np.clip(proba_positive_class, eps, 1 - eps)
    return np.log(proba_positive_class / (1 - proba_positive_class))


# ----------------------------------------------------------------------
# Dataset and data loader


class GenericDataset(Dataset):
    def __init__(self, inputs: Tuple[Any], dtypes="float", device: str = "cpu"):
        self.inputs = inputs
        self.device = device
        if is_listish(dtypes):
            self.dtypes = dtypes
        else:
            self.dtypes = [dtypes] * (len(inputs) if is_listish(inputs) else 1)

    def __len__(self):
        if is_listish(self.inputs):
            return len(self.inputs[0])
        else:
            return len(self.inputs)

    def __getitem__(self, idx):
        if is_listish(self.inputs):
            return [
                to_tensor(input[idx], self.dtypes[i], self.device)
                for i, input in enumerate(self.inputs)
            ]
        else:
            return to_tensor(self.inputs[idx], self.dtypes[0], self.device)


def create_generic_dataloader(
    inputs: Tuple[Any],
    batch_size=32,
    shuffle=False,
    device="cpu",
    dtypes="float",
):
    """
    Create a generic PyTorch DataLoader.

    Parameters
    ----------
    inputs : Tuple[Any]
        Can be a tuple of numpy arrays, a pytorch dataset, a pandas dataframe,
        a polars dataframe, or a pyarrow table.
    batch_size : int, optional
        by default 32
    shuffle : bool, optional
        by default False
    device : str, optional
        by default "cpu"

    Returns
    -------
    loader : DataLoader
        PyTorch DataLoader instance
    """
    if is_listish(inputs) | safe_isinstance(inputs, "numpy.ndarray"):
        inputs = GenericDataset(inputs, dtypes, device)

    return DataLoader(
        inputs,
        shuffle=shuffle,
        batch_size=batch_size,
    )