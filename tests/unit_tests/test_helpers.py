import unittest

from type_infer import helpers
from type_infer.rule_based.helpers import get_language_dist


class TestCastStringToPythonType(unittest.TestCase):
    def test_numeric_unicode_is_none(self):
        numeric_unicode_str = '\u00BC'
        self.assertTrue(numeric_unicode_str.isnumeric())
        self.assertIsNone(helpers.cast_string_to_python_type(numeric_unicode_str))

    def test_str_is_none(self):
        self.assertIsNone(helpers.cast_string_to_python_type(''))
        self.assertIsNone(helpers.cast_string_to_python_type('None'))
        self.assertIsNone(helpers.cast_string_to_python_type('nan'))

    def test_str_is_int(self):
        self.assertEqual(helpers.cast_string_to_python_type('1'), 1)

    def test_str_is_float(self):
        self.assertEqual(helpers.cast_string_to_python_type('1.1'), 1.1)
        self.assertEqual(helpers.cast_string_to_python_type('1,1'), 1.1)
        self.assertEqual(helpers.cast_string_to_python_type('1.'), 1.0)
        self.assertEqual(helpers.cast_string_to_python_type('inf'), float('inf'))


class TestIsNanNumeric(unittest.TestCase):
    def test_nan_is_numeric(self):
        self.assertTrue(helpers.is_nan_numeric('nan'))
        self.assertTrue(helpers.is_nan_numeric(float('nan')))

    def test_inf_is_numeric(self):
        self.assertTrue(helpers.is_nan_numeric('inf'))
        self.assertTrue(helpers.is_nan_numeric(float('inf')))


class TestLangid(unittest.TestCase):
    def test_get_language_dist(self):
        dist = get_language_dist(['This is clearly English', 'Y esto es claramente español'])
        self.assertTrue('en' in dist and dist['en'] == 1)
        self.assertTrue('es' in dist and dist['es'] == 1)
