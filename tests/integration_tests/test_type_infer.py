import unittest
import pandas as pd

from type_infer.infer import infer_types


class TestTypeInference(unittest.TestCase):
    def test_0_airline_sentiment(self):
        df = pd.read_csv("tests/data/airline_sentiment_sample.csv")
        inferred_types = infer_types(df, pct_invalid=0)

        expected_types = {
            'airline_sentiment': 'categorical',
            'retweet_count': 'categorical',
            'airline': 'categorical',
            'name': 'categorical',

            'airline_sentiment_gold': 'binary',

            'tweet_id': 'integer',
            'negativereason_confidence': 'float',
            'airline_sentiment_confidence': 'float',

            'tweet_created': 'datetime',

            'negativereason': 'short_text',
            'tweet_location': 'short_text',
            'user_timezone': 'short_text',
            'text': 'rich_text',

            'tweet_coord': 'categorical',  # TODO: should be detected as coordinates
            'negativereason_gold': 'invalid',
        }
        expected_ids = {
            'tweet_id': 'Foreign key',
            'name': 'Unknown identifier'
        }
        self.assertEqual(expected_types, inferred_types.dtypes)
        self.assertEqual(expected_ids, inferred_types.identifiers)
