import os
import random
import logging
import colorlog
import numpy as np


def seed(seed_nr: int) -> None:
    np.random.seed(seed_nr)
    random.seed(seed_nr)


def is_nan_numeric(value: object) -> bool:
    """
    Determines if **value** might be `nan` or `inf` or some other numeric value (i.e. which can be cast as `float`) that is not actually a number.
    """  # noqa

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
