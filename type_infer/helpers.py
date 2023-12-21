import os
import psutil
import random
import logging
import colorlog
import multiprocessing as mp
from typing import Iterable

import numpy as np


def initialize_log():
    pid = os.getpid()

    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter())

    logging.basicConfig(handlers=[handler])
    log = logging.getLogger(f'type_infer-{pid}')
    log_level = os.environ.get('TYPE_INFER_LOG', 'DEBUG')
    log.setLevel(log_level)
    return log


log = initialize_log()


def get_nr_procs(df=None):
    if 'MINDSDB_N_WORKERS' in os.environ:
        try:
            n = int(os.environ['MINDSDB_N_WORKERS'])
        except ValueError:
            n = 1
        return n
    elif os.name == 'nt':
        return 1
    else:
        available_mem = psutil.virtual_memory().available
        if df is not None:
            max_per_proc_usage = df.size
        else:
            max_per_proc_usage = 0.2 * pow(10, 9)  # multiplier * 1GB

        proc_count = int(min(mp.cpu_count() - 1, available_mem // max_per_proc_usage))

        return max(proc_count, 1)


def seed(seed_nr: int) -> None:
    np.random.seed(seed_nr)
    random.seed(seed_nr)


def is_nan_numeric(value: object) -> bool:
    """
    Determines if **value** might be `nan` or `inf` or some other numeric value (i.e. which can be cast as `float`) that is not actually a number.
    """  # noqa
    if isinstance(value, np.ndarray) or (type(value) != str and isinstance(value, Iterable)):
        return False

    try:
        value = str(value)
        value = float(value)
    except Exception:
        return False

    try:
        if isinstance(value, float):
            a = int(value) # noqa
        isnan = False
    except Exception:
        isnan = True
    return isnan


def cast_string_to_python_type(string):
    """ Returns None, an integer, float or a string from a string"""
    if string is None or string == '':
        return None

    if string.isnumeric():
        # Did you know you can write fractions in unicode, and they are numeric but can't be cast to integers !?
        try:
            return int(string)
        except Exception:
            return None

    try:
        return clean_float(string)
    except Exception:
        return string


def clean_float(val):
    if isinstance(val, (int, float)):
        return float(val)

    if isinstance(val, float):
        return val

    val = str(val).strip(' ')
    val = val.replace(',', '.')
    val = val.rstrip('"').lstrip('"')

    if val in ('', '.', 'None', 'nan'):
        return None

    try:
        return float(val)
    except Exception:
        return None
