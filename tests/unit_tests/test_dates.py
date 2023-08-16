import unittest

from type_infer.dtype import dtype
from type_infer.infer import type_check_date


class TestDates(unittest.TestCase):

    def test_0_type_check_dates(self):
        """ Checks parsing of string containing a date to dtype 'date'.
        """
        self.assertEqual(type_check_date('31/12/2010'), dtype.date)
    
    def test_1_type_check_datetime(self):
        """ Checks parsing of string containing a date to dtype 'datetime'.
        """
        self.assertEqual(type_check_date('31/12/2010 23:15:41'), dtype.datetime)
    
    def test_2_type_check_timestamp_unix_seconds(self):
        """ Checks parsing a number containing 1989-12-15T07:30:00 (as seconds
            since Unix epoch) to dtype 'timestamp'.
        """
        self.assertEqual(type_check_date(629721000.0), dtype.timestamp)
    
    def test_3_type_check_timestamp_unix_miliseconds(self):
        """ Checks parsing a number containing 1989-12-15T07:30:00 (as miliseconds
            since Unix epoch) to dtype 'timestamp'.
        """
        self.assertEqual(type_check_date(629721000000.0), dtype.timestamp)

    def test_4_type_check_timestamp_unix_microseconds(self):
        """ Checks parsing a number containing 1989-12-15T07:30:00 (as microseconds
            since Unix epoch) to dtype 'timestamp'.
        """
        self.assertEqual(type_check_date(629721000000000.0), dtype.timestamp)
    
    def test_5_type_check_timestamp_unix_nanoseconds(self):
        """ Checks parsing a number containing 1989-12-15T07:30:00 (as nanoseconds
            since Unix epoch) to dtype 'timestamp'.
        """
        self.assertEqual(type_check_date(629721000000000000.0), dtype.timestamp)
    
    def test_6_type_check_timestamp_julian_days(self):
        """ Checks parsing a number containing 1989-12-15T07:30:00 (as days since
            Julian calendar epoch) to dtype 'timestamp'.
        """
        self.assertEqual(type_check_date(2447875.81250), dtype.timestamp)