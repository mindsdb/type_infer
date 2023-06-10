import unittest
import random

import pandas as pd
from type_infer.infer import get_column_data_type
from type_infer.dtype import dtype


class TestInferDtypes(unittest.TestCase):
    def test_negative_integers(self):
        data = pd.DataFrame([-random.randint(-10, 10) for _ in range(100)], columns=['test_col'])
        dtyp, dist, ainfo, warn, info = get_column_data_type(data['test_col'], data, 'test_col', 0.0)
        self.assertEqual(dtyp, dtype.integer)

    def test_negative_floats(self):
        data = pd.DataFrame([-random.randint(-10, 10) for _ in range(100)] + [0.1], columns=['test_col'])
        dtyp, dist, ainfo, warn, info = get_column_data_type(data['test_col'], data, 'test_col', 0.0)
        self.assertEqual(dtyp, dtype.float)
