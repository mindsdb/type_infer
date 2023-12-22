import unittest
import random

import pandas as pd
from type_infer.rule_based.core import RuleBasedEngine
from type_infer.dtype import dtype

get_column_data_type = RuleBasedEngine.get_column_data_type

class TestInferDtypes(unittest.TestCase):
    def test_negative_integers(self):
        data = pd.DataFrame([-random.randint(-10, 10) for _ in range(100)], columns=['test_col'])
        engine = RuleBasedEngine()
        dtyp, dist, ainfo, warn, info = engine.get_column_data_type(data['test_col'], data, 'test_col', 0.0)
        self.assertEqual(dtyp, dtype.integer)

    def test_negative_floats(self):
        data = pd.DataFrame([-random.randint(-10, 10) for _ in range(100)] + [0.1], columns=['test_col'])
        engine = RuleBasedEngine()
        dtyp, dist, ainfo, warn, info = engine.get_column_data_type(data['test_col'], data, 'test_col', 0.0)
        self.assertEqual(dtyp, dtype.float)
