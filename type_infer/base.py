from typing import Dict
from dataclasses import dataclass
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class TypeInformation:
    """
    For a dataset, provides information on columns types.

    ``TypeInformation`` is generated within :py:func:`infer.infer_types`,
    where a small subset of samples of each column are evaluated in a custom
    framework to understand what kind of data type the model is. The user
    may override data types, but it is recommended to do so within a JSON-AI
    config file.

    :param dtypes: For each column's name, the associated data type inferred.
    :param additional_info: Any possible sub-categories or additional descriptive
           information.
    :param identifiers: Columns within the dataset highly suspected of being identifiers
           or IDs. These do not contain useful information, and should therefore be
           ignored in subsequent training/analysis procedures unless manually indicated.
    """ # noqa

    dtypes: Dict[str, str]
    additional_info: Dict[str, object]
    identifiers: Dict[str, str]

    def __init__(self):
        self.dtypes = dict()
        self.additional_info = dict()
        self.identifiers = dict()
