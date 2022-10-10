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

    def test_1_stack_overflow_survey(self):
        df = pd.read_csv("tests/data/stack_overflow_survey_sample.csv")
        inferred_types = infer_types(df, pct_invalid=0)

        expected_types = {
            'Respondent': 'integer',
            'Professional': 'binary',
            'ProgramHobby': 'categorical',
            'Country': 'short_text',
            'University': 'categorical',
            'EmploymentStatus': 'binary',
            'FormalEducation': 'categorical',
            'CompanySize': 'tags',
            'CompanyType': 'tags',
            'CareerSatisfaction': 'integer',
            'JobSatisfaction': 'integer',
            'HoursPerWeek': 'integer',
            'AssessJobRemote': 'categorical',
            'LearnedHiring': 'tags',
            'ImportantHiringOpenSource': 'categorical',
            'TabsSpaces': 'categorical',
            'ExCoderReturn': 'invalid',
        }
        expected_ids = {
            'Professional': 'No Information'
        }

        for col in expected_types:
            self.assertTrue(expected_types[col], inferred_types.dtypes[col])

        for col in expected_ids:
            self.assertTrue(expected_ids[col], inferred_types.identifiers[col])
