import toml
import unittest
from pathlib import Path

import type_infer
from type_infer.helpers import tokenize_text


class TestDates(unittest.TestCase):
    def test_versions_are_in_sync(self):
        """
        Checks if the pyproject.toml and package.__init__.py __version__ are in sync.
        Ref: github.com/python-poetry/poetry/issues/144#issuecomment-877835259
        """
        path = Path(__file__).resolve().parents[2] / "pyproject.toml"
        pyproject = toml.loads(open(str(path)).read())
        pyproject_version = pyproject["tool"]["poetry"]["version"]

        package_init_version = type_infer.__version__

        self.assertEqual(package_init_version, pyproject_version)

    def test_get_tokens(self):
        sentences = ['hello, world!', ' !hello! world!!,..#', '#hello!world']
        for sent in sentences:
            assert tokenize_text(sent) == ['hello', 'world']

        assert tokenize_text("don't wouldn't") == ['do', 'not', 'would', 'not']
