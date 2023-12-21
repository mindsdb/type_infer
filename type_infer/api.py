import random
import multiprocessing as mp

from scipy.stats import norm
import pandas as pd

from type_infer.base import TypeInformation
from type_infer.dtype import dtype
from type_infer.helpers import seed, log, get_nr_procs

# inference engine specific imports
from type_infer.rule_based.infer import get_column_data_type
from type_infer.rule_based.helpers import get_identifier_description_mp


def _calculate_sample_size(
    population_size,
    margin_error=.01,
    confidence_level=.995,
    sigma=1 / 2
):
    """
    Calculate the minimal sample size to use to achieve a certain
    margin of error and confidence level for a sample estimate
    of the population mean.
    Inputs
    -------
    population_size: integer
        Total size of the population that the sample is to be drawn from.
    margin_error: number
        Maximum expected difference between the true population parameter,
        such as the mean, and the sample estimate.
    confidence_level: number in the interval (0, 1)
        If we were to draw a large number of equal-size samples
        from the population, the true population parameter
        should lie within this percentage
        of the intervals (sample_parameter - e, sample_parameter + e)
        where e is the margin_error.
    sigma: number
        The standard deviation of the population.  For the case
        of estimating a parameter in the interval [0, 1], sigma=1/2
        should be sufficient.
    """
    alpha = 1 - confidence_level
    # dictionary of confidence levels and corresponding z-scores
    # computed via norm.ppf(1 - (alpha/2)), where norm is
    # a normal distribution object in scipy.stats.
    # Here, ppf is the percentile point function.
    zdict = {
        .90: 1.645,
        .91: 1.695,
        .99: 2.576,
        .97: 2.17,
        .94: 1.881,
        .93: 1.812,
        .95: 1.96,
        .98: 2.326,
        .96: 2.054,
        .92: 1.751
    }
    if confidence_level in zdict:
        z = zdict[confidence_level]
    else:
        # Inf fix
        if alpha == 0.0:
            alpha += 0.001
        z = norm.ppf(1 - (alpha / 2))
    N = population_size
    M = margin_error
    numerator = z**2 * sigma**2 * (N / (N - 1))
    denom = M**2 + ((z**2 * sigma**2) / (N - 1))
    return numerator / denom


def _sample_data(df: pd.DataFrame) -> pd.DataFrame:
    population_size = len(df)
    if population_size <= 50:
        sample_size = population_size
    else:
        sample_size = int(round(_calculate_sample_size(population_size)))

    population_size = len(df)
    input_data_sample_indexes = random.sample(range(population_size), sample_size)
    return df.iloc[input_data_sample_indexes]


def infer_types(
        data: pd.DataFrame,
        # TODO: method: InferenceEngine = Union[InferenceEngine.RuleBased, InferenceEngine.BERT],
        pct_invalid: float,
        seed_nr: int = 420,
        mp_cutoff: int = 1e4,
) -> TypeInformation:
    """
    Infers the data types of each column of the dataset by analyzing a small sample of
    each column's items.

    Inputs
    ----------
    data : pd.DataFrame
        The input dataset for which we want to infer data type information.
    pct_invalid : float
        The percentage, i.e. a float between 0.0 and 100.0, of invalid values that are
        accepted before failing the type inference for a column.
    seed_nr : int, optional
        Seed for the random number generator, by default 420
    mp_cutoff : int, optional
        How many elements in the dataframe before switching to parallel processing, by
        default 1e4.
    """
    seed(seed_nr)
    type_information = TypeInformation()
    sample_df = _sample_data(data)
    sample_size = len(sample_df)
    population_size = len(data)
    log.info(f'Analyzing a sample of {sample_size}')
    log.info(
        f'from a total population of {population_size}, this is equivalent to {round(sample_size*100/population_size, 1)}% of your data.')  # noqa

    nr_procs = get_nr_procs(df=sample_df)
    pool_size = min(nr_procs, len(sample_df.columns.values))
    if data.size > mp_cutoff and pool_size > 1:
        log.info(f'Using {pool_size} processes to deduct types.')
        pool = mp.Pool(processes=pool_size)
        # column-wise parallelization  # TODO: evaluate switching to row-wise split instead
        # TODO: this would be the call to the inference engine -> column in, type out
        answer_arr = pool.starmap(get_column_data_type, [
            (sample_df[x].dropna(), data[x], x, pct_invalid) for x in sample_df.columns.values
        ])
        pool.close()
        pool.join()
    else:
        answer_arr = []
        for x in sample_df.columns:
            answer_arr.append(get_column_data_type(sample_df[x].dropna(), data, x, pct_invalid))

    for i, col_name in enumerate(sample_df.columns):
        (data_dtype, data_dtype_dist, additional_info, warn, info) = answer_arr[i]

        for msg in warn:
            log.warning(msg)
        for msg in info:
            log.info(msg)

        if data_dtype is None:
            data_dtype = dtype.invalid

        type_information.dtypes[col_name] = data_dtype
        type_information.additional_info[col_name] = {
            'dtype_dist': data_dtype_dist
        }

    if data.size > mp_cutoff and pool_size > 1:
        pool = mp.Pool(processes=pool_size)
        answer_arr = pool.map(get_identifier_description_mp, [
            (data[x], x, type_information.dtypes[x])
            for x in sample_df.columns
        ])
        pool.close()
        pool.join()
    else:
        answer_arr = []
        for x in sample_df.columns:
            answer = get_identifier_description_mp([data[x], x, type_information.dtypes[x]])
            answer_arr.append(answer)

    for i, col_name in enumerate(sample_df.columns):
        # work with the full data
        if answer_arr[i] is not None:
            log.warning(f'Column {col_name} is an identifier of type "{answer_arr[i]}"')
            type_information.identifiers[col_name] = answer_arr[i]

        # @TODO Column removal logic was here, if the column was an identifier, move it elsewhere

    return type_information
