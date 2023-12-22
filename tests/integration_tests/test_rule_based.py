from uuid import uuid4
import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from type_infer.dtype import dtype
from type_infer.api import infer_types


class TestRuleBasedTypeInference(unittest.TestCase):
    def test_0_airline_sentiment(self):
        df = pd.read_csv("tests/data/airline_sentiment_sample.csv")
        config = {'engine': 'rule_based', 'pct_invalid': 0, 'seed': 420, 'mp_cutoff': 1e4}
        inferred_types = infer_types(df, config=config)

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

            'negativereason': 'categorical',
            'tweet_location': 'short_text',
            'user_timezone': 'short_text',
            'text': 'rich_text',

            'tweet_coord': 'categorical',  # TODO: should be detected as coordinates
            'negativereason_gold': 'invalid',
        }
        expected_ids = {
            'tweet_id': 'Foreign key',
            'name': 'Unknown identifier',
        }
        self.assertEqual(expected_types, inferred_types.dtypes)
        self.assertEqual(expected_ids, inferred_types.identifiers)

    def test_1_stack_overflow_survey(self):
        df = pd.read_csv("tests/data/stack_overflow_survey_sample.csv")
        config = {'engine': 'rule_based', 'pct_invalid': 0, 'seed': 420, 'mp_cutoff': 1e4}


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

        inferred_types = infer_types(df, config=config)

        for col in expected_types:
            self.assertTrue(expected_types[col], inferred_types.dtypes[col])

        for col in expected_ids:
            self.assertTrue(expected_ids[col], inferred_types.identifiers[col])

    def test_2_simple(self):
        n_points = 50
        n_corrupted = 2
        df = pd.DataFrame({
            'date': [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(n_points)],
            'datetime': [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%dT%H:%M') for i in range(n_points)],
            'integer': [*range(n_points)],
            'float': np.linspace(0, n_points, n_points),
            'uuid': [str(uuid4()) for i in range(n_points)],
        })

        # manual tinkering
        df['float'].iloc[-n_corrupted:] = 'random string'

        pct_invalid = 100 * (n_corrupted) / n_points
        config = {'engine': 'rule_based', 'pct_invalid': pct_invalid, 'seed': 420, 'mp_cutoff': 1e4}

        inferred_types = infer_types(df, config=config)
        expected_types = {
            'date': dtype.date,
            'datetime': dtype.datetime,
            'integer': dtype.integer,
            'float': dtype.float,
            'uuid': dtype.categorical,
        }
        self.assertEqual(expected_types, inferred_types.dtypes)  # check type inference is correct
        self.assertTrue(inferred_types.additional_info['date']['dtype_dist']['date'] == n_points)  # no dropped rows (pct_invalid is 0)   # noqa
        self.assertTrue(inferred_types.additional_info['float']['dtype_dist']['float'] == n_points - 2)  # due to str injection  # noqa
        self.assertTrue('uuid' in inferred_types.identifiers)
        self.assertTrue(inferred_types.identifiers['uuid'] == 'UUID')
