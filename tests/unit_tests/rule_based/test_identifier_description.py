"""
Unit tests for get_identifier_description and related helpers in
type_infer/rule_based/helpers.py

Covers issue: https://github.com/mindsdb/type_infer/issues/31
"""
import unittest

from type_infer.dtype import dtype
from type_infer.rule_based.helpers import (
    get_identifier_description,
    get_identifier_description_mp,
    _is_foreign_key_name,
    _is_identifier_name,
)


class TestGetIdentifierDescriptionNoInformation(unittest.TestCase):
    """Single-unique-value data should always return 'No Information'."""

    def test_single_unique_integer(self):
        data = [42, 42, 42, 42]
        self.assertEqual(
            get_identifier_description(data, "count", dtype.integer),
            "No Information",
        )

    def test_single_unique_string(self):
        data = ["active", "active", "active"]
        self.assertEqual(
            get_identifier_description(data, "status", dtype.categorical),
            "No Information",
        )

    def test_single_unique_float(self):
        data = [3.14, 3.14, 3.14]
        self.assertEqual(
            get_identifier_description(data, "pi", dtype.float),
            "No Information",
        )

    def test_single_unique_list(self):
        data = [[1, 2], [1, 2], [1, 2]]
        self.assertEqual(
            get_identifier_description(data, "coords", dtype.array),
            "No Information",
        )

    def test_single_unique_dict(self):
        data = [{"a": 1}, {"a": 1}, {"a": 1}]
        self.assertEqual(
            get_identifier_description(data, "meta", dtype.json),
            "No Information",
        )


class TestGetIdentifierDescriptionForeignKey(unittest.TestCase):
    """Integer columns whose name matches foreign-key patterns return 'Foreign key'."""

    def _make_int_col(self, n=20):
        """Return n distinct integers to avoid 'No Information'."""
        return list(range(1, n + 1))

    def test_underscore_id_suffix(self):
        self.assertEqual(
            get_identifier_description(self._make_int_col(), "user_id", dtype.integer),
            "Foreign key",
        )

    def test_hyphen_id_suffix(self):
        self.assertEqual(
            get_identifier_description(self._make_int_col(), "order-id", dtype.integer),
            "Foreign key",
        )

    def test_space_id_suffix(self):
        self.assertEqual(
            get_identifier_description(self._make_int_col(), "product id", dtype.integer),
            "Foreign key",
        )

    def test_uppercase_ID_suffix(self):
        self.assertEqual(
            get_identifier_description(self._make_int_col(), "customer_ID", dtype.integer),
            "Foreign key",
        )

    def test_titlecase_Id_suffix(self):
        self.assertEqual(
            get_identifier_description(self._make_int_col(), "category_Id", dtype.integer),
            "Foreign key",
        )

    def test_bare_uppercase_ID(self):
        self.assertEqual(
            get_identifier_description(self._make_int_col(), "categoryID", dtype.integer),
            "Foreign key",
        )

    def test_bare_titlecase_Id(self):
        self.assertEqual(
            get_identifier_description(self._make_int_col(), "categoryId", dtype.integer),
            "Foreign key",
        )

    def test_non_integer_dtype_does_not_trigger(self):
        """Foreign key detection only fires for integer dtype."""
        result = get_identifier_description(
            [f"val{i}" for i in range(1, 21)], "user_id", dtype.categorical
        )
        # Should NOT return 'Foreign key' for string dtype
        self.assertNotEqual(result, "Foreign key")

    def test_non_fk_name_integer(self):
        """Plain integer column without an FK name pattern returns None."""
        result = get_identifier_description(
            self._make_int_col(), "age", dtype.integer
        )
        self.assertIsNone(result)


class TestGetIdentifierDescriptionUUID(unittest.TestCase):
    """High-cardinality UUID-charset columns return 'UUID'."""

    def _make_uuids(self, n=50):
        import uuid
        return [str(uuid.uuid4()) for _ in range(n)]

    def test_uuid_detection(self):
        uuids = self._make_uuids(50)
        result = get_identifier_description(uuids, "account_uuid", dtype.categorical)
        self.assertEqual(result, "UUID")

    def test_uuid_on_identifier_name(self):
        uuids = self._make_uuids(50)
        result = get_identifier_description(uuids, "user_identifier", dtype.categorical)
        self.assertEqual(result, "UUID")


class TestGetIdentifierDescriptionUnknownIdentifier(unittest.TestCase):
    """High-cardinality non-UUID columns with identifier-like names return 'Unknown identifier'."""

    def test_unknown_identifier_account_name(self):
        # 50 distinct short non-uuid strings with high cardinality
        data = [f"ACC{i:04d}" for i in range(50)]
        result = get_identifier_description(data, "account_code", dtype.categorical)
        self.assertEqual(result, "Unknown identifier")

    def test_unknown_identifier_user_name(self):
        data = [f"USR{i:04d}" for i in range(50)]
        result = get_identifier_description(data, "user_key", dtype.categorical)
        self.assertEqual(result, "Unknown identifier")


class TestGetIdentifierDescriptionHashLike(unittest.TestCase):
    """
    Equal-length, fully-unique strings with high per-position entropy
    should return 'Hash-like identifier'.
    """

    def _make_hashes(self, n=30, length=32):
        """Generate pseudo-random fixed-length hex strings."""
        import hashlib
        return [hashlib.md5(f"seed_{i}".encode()).hexdigest() for i in range(n)]

    def test_md5_hash_column(self):
        hashes = self._make_hashes(30)
        result = get_identifier_description(hashes, "checksum", dtype.categorical)
        self.assertEqual(result, "Hash-like identifier")

    def test_sha256_hash_column(self):
        import hashlib
        hashes = [hashlib.sha256(f"item_{i}".encode()).hexdigest() for i in range(30)]
        result = get_identifier_description(hashes, "hash_col", dtype.categorical)
        self.assertEqual(result, "Hash-like identifier")


class TestGetIdentifierDescriptionNoneReturn(unittest.TestCase):
    """Normal columns with varied data and no special patterns return None."""

    def test_normal_text_column(self):
        data = ["apple", "banana", "cherry", "date", "elderberry"] * 4
        result = get_identifier_description(data, "fruit", dtype.categorical)
        self.assertIsNone(result)

    def test_normal_float_column(self):
        import random
        random.seed(42)
        data = [round(random.uniform(0, 100), 2) for _ in range(30)]
        result = get_identifier_description(data, "temperature", dtype.float)
        self.assertIsNone(result)


class TestGetIdentifierDescriptionMpWrapper(unittest.TestCase):
    """get_identifier_description_mp must produce identical output to the direct call."""

    def test_mp_wrapper_matches_direct_no_information(self):
        data = [1, 1, 1, 1]
        expected = get_identifier_description(data, "col", dtype.integer)
        self.assertEqual(
            get_identifier_description_mp((data, "col", dtype.integer)),
            expected,
        )

    def test_mp_wrapper_matches_direct_foreign_key(self):
        data = list(range(1, 21))
        expected = get_identifier_description(data, "order_id", dtype.integer)
        self.assertEqual(
            get_identifier_description_mp((data, "order_id", dtype.integer)),
            expected,
        )

    def test_mp_wrapper_matches_direct_none(self):
        data = ["foo", "bar", "baz"] * 5
        expected = get_identifier_description(data, "label", dtype.categorical)
        self.assertEqual(
            get_identifier_description_mp((data, "label", dtype.categorical)),
            expected,
        )


class TestIsForeignKeyName(unittest.TestCase):
    """Unit tests for the _is_foreign_key_name helper."""

    def test_underscore_id(self):
        self.assertTrue(_is_foreign_key_name("user_id"))

    def test_underscore_ID(self):
        self.assertTrue(_is_foreign_key_name("user_ID"))

    def test_underscore_Id(self):
        self.assertTrue(_is_foreign_key_name("user_Id"))

    def test_hyphen_id(self):
        self.assertTrue(_is_foreign_key_name("user-id"))

    def test_space_id(self):
        self.assertTrue(_is_foreign_key_name("user id"))

    def test_bare_ID(self):
        self.assertTrue(_is_foreign_key_name("userID"))

    def test_bare_Id(self):
        self.assertTrue(_is_foreign_key_name("userId"))

    def test_no_match_plain(self):
        self.assertFalse(_is_foreign_key_name("valid"))

    def test_no_match_grid(self):
        self.assertFalse(_is_foreign_key_name("grid"))

    def test_no_match_idea(self):
        self.assertFalse(_is_foreign_key_name("idea"))

    def test_no_match_name(self):
        self.assertFalse(_is_foreign_key_name("name"))


class TestIsIdentifierName(unittest.TestCase):
    """Unit tests for the _is_identifier_name helper."""

    def test_contains_account(self):
        self.assertTrue(_is_identifier_name("account_code"))

    def test_contains_uuid(self):
        self.assertTrue(_is_identifier_name("uuid"))

    def test_contains_identifier(self):
        self.assertTrue(_is_identifier_name("user_identifier"))

    def test_contains_user(self):
        self.assertTrue(_is_identifier_name("username"))

    def test_no_match(self):
        self.assertFalse(_is_identifier_name("price"))

    def test_no_match_order(self):
        self.assertFalse(_is_identifier_name("order_total"))

    def test_no_match_category(self):
        self.assertFalse(_is_identifier_name("category"))


if __name__ == "__main__":
    unittest.main()
