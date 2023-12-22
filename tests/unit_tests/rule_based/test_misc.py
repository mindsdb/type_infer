import unittest

from type_infer.rule_based.helpers import tokenize_text


class TestDates(unittest.TestCase):
    def test_get_tokens(self):
        sentences = ['hello, world!', ' !hello! world!!,..#', '#hello!world']
        for sent in sentences:
            assert list(tokenize_text(sent)) == ['hello', 'world']

        assert list(tokenize_text("don't wouldn't")) == ['do', 'not', 'would', 'not']
