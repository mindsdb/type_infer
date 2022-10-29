import unittest

from type_infer import helpers


class TestCastStringToPythonType(unittest.TestCase):
    def test_numeric_unicode_is_none(self):
        numeric_unicode = '\u00BC'
        self.assertTrue(numeric_unicode.isnumeric())
        self.assertIsNone(helpers.cast_string_to_python_type(numeric_unicode))

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
