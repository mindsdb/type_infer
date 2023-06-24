import os
import re
import nltk
import psutil
import random
import string
import logging
import colorlog
import multiprocessing as mp

import numpy as np
import scipy.stats as st
from langid.langid import LanguageIdentifier
from langid.langid import model as langid_model

from typing import Iterable
from collections import Counter, defaultdict

from type_infer.dtype import dtype


try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    from nltk.corpus import stopwords
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords', quiet=True)


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


def get_identifier_description_mp(arg_tup):
    data, column_name, data_dtype = arg_tup
    return get_identifier_description(data, column_name, data_dtype)


def get_identifier_description(data: Iterable, column_name: str, data_dtype: dtype):
    data = list(data)
    if isinstance(data[0], list):
        nr_unique = len(set(tuple(x) for x in data))
    elif isinstance(data[0], dict):
        nr_unique = len(set(str(x) for x in data))
    else:
        nr_unique = len(set(data))

    if nr_unique == 1:
        return 'No Information'

    unique_pct = nr_unique / len(data)

    spaces = [len(str(x).split(' ')) - 1 for x in data]
    mean_spaces = np.mean(spaces)

    # Detect hash
    all_same_length = all(len(str(data[0])) == len(str(x)) for x in data)
    uuid_charset = set('0123456789abcdefABCDEF-')
    all_uuid_charset = all(set(str(x)).issubset(uuid_charset) for x in data)
    is_uuid = all_uuid_charset and all_same_length

    if all_same_length and len(data) == nr_unique and data_dtype not in (dtype.integer, dtype.float):
        str_data = [str(x) for x in data]
        randomness_per_index = []
        for i, _ in enumerate(str_data[0]):
            N = len(set(x[i] for x in str_data))
            S = st.entropy([*Counter(x[i] for x in str_data).values()])
            if S == 0:
                randomness_per_index.append(0.0)
            else:
                randomness_per_index.append(S / np.log(N))

        if np.mean(randomness_per_index) > 0.95:
            return 'Hash-like identifier'

    # Detect foreign key
    if data_dtype == dtype.integer:
        if _is_foreign_key_name(column_name):
            return 'Foreign key'

    if _is_identifier_name(column_name) or data_dtype in (dtype.categorical, dtype.binary):
        if unique_pct > 0.98:
            if is_uuid:
                return 'UUID'
            else:
                return 'Unknown identifier'

    # Everything is unique and it's too short to be rich text
    if data_dtype in (dtype.categorical, dtype.binary, dtype.short_text, dtype.rich_text) and \
            unique_pct > 0.99999 and mean_spaces < 1:
        return 'Unknown identifier'

    return None


def _is_foreign_key_name(name):
    for endings in ['id', 'ID', 'Id']:
        for add in ['-', '_', ' ']:
            if name.endswith(add + endings):
                return True
    for endings in ['ID', 'Id']:
        if name.endswith(endings):
            return True
    return False


def _is_identifier_name(name):
    for keyword in ['account', 'uuid', 'identifier', 'user']:
        if keyword in name:
            return True
    return False


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


# TODO: Should this be here?
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

    return float(val)


def get_language_dist(data):
    lang_dist = defaultdict(lambda: 0)
    lang_dist['Unknown'] = 0
    lang_probs_cache = dict()
    identifier = LanguageIdentifier.from_modelstring(langid_model, norm_probs=True)
    for text in data:
        text = str(text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        if text not in lang_probs_cache:
            try:
                lang_probs = identifier.classify(text)
            except Exception:
                lang_probs = []
            lang_probs_cache[text] = lang_probs

        lang_probs = lang_probs_cache[text]
        if len(lang_probs) > 0 and lang_probs[1] > 10 * (1 / len(identifier.nb_classes)):
            lang_dist[lang_probs[0]] += 1
        else:
            lang_dist['Unknown'] += 1

    return dict(lang_dist)


def analyze_sentences(data):
    nr_words = 0
    word_dist = defaultdict(int)
    nr_words_dist = defaultdict(int)
    stop_words = set(stopwords.words('english'))
    for text in map(str, data):
        text = text.lower()
        text_dist = defaultdict(int)
        tokens = tokenize_text(text)
        tokens_no_stop = (x for x in tokens if x not in stop_words)
        for tok in tokens_no_stop:
            text_dist[tok] += 1

        n_tokens = len(text_dist)
        nr_words_dist[n_tokens] += 1
        nr_words += n_tokens

        # merge text_dist into word_dist
        for k, v in text_dist.items():
            word_dist[k] += v

    return nr_words, dict(word_dist), dict(nr_words_dist)


# @TODO: eventually move these into .helpers.text
def tokenize_text(text):
    """ Generator instead of list comprehension for optimal memory usage & runtime """
    return (t.lower() for t in nltk.word_tokenize(decontracted(text)) if contains_alnum(t))


def decontracted(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


def contains_alnum(text):
    for c in text:
        if c.isalnum():
            return True
    return False


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
