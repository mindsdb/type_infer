import unittest
import random

from type_infer.infer import get_column_data_type
from type_infer.dtype import dtype


class TestInferDtypes(unittest.TestCase):
    def test_negative_floats(self):
        data = [-random.random() for _ in range(100)]

        dtyp, dist, ainfo, warn, info = get_column_data_type((data, data, 'test_col', 0.0))
        self.assertEqual(dtyp, dtype.float)
