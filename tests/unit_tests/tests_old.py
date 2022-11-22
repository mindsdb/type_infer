
'''
    # These are mindsdb_native tests, we should adapt them to type_infer
    # Not critical since we removed a bunch of these capacities, other are well tested already
    def test_deduce_foreign_key(self):
        """Tests that basic cases of type deduction work correctly"""
        predictor = Predictor(name='test_deduce_foreign_key')
        predictor.breakpoint = 'DataAnalyzer'

        n_points = 100

        df = pd.DataFrame({
            'numeric_id': list(range(n_points)),
            'uuid': [str(uuid4()) for i in range(n_points)],
            'to_predict': [i % 5 for i in range(n_points)]
        })

        try:
            predictor.learn(from_data=df, to_predict='to_predict')
        except BreakpointException:
            pass
        else:
            raise AssertionError

        stats_v2 = predictor.transaction.lmd['stats_v2']

        assert isinstance(stats_v2['numeric_id']['identifier'], str)
        assert isinstance(stats_v2['uuid']['identifier'], str)

        assert 'numeric_id' in predictor.transaction.lmd['columns_to_ignore']
        assert 'uuid' in predictor.transaction.lmd['columns_to_ignore']

    def test_empty_values(self):
        predictor = Predictor(name='test_empty_values')
        predictor.breakpoint = 'TypeDeductor'

        n_points = 100
        df = pd.DataFrame({
            'numeric_float_1': np.linspace(0, n_points, n_points),
            'numeric_float_2': np.linspace(0, n_points, n_points),
            'numeric_float_3': np.linspace(0, n_points, n_points),
        })
        df['numeric_float_1'].iloc[::2] = None

        try:
            predictor.learn(
                from_data=df,
                to_predict='numeric_float_3',
                advanced_args={'force_column_usage': list(df.columns)}
            )
        except BreakpointException:
            pass
        else:
            raise AssertionError

        stats_v2 = predictor.transaction.lmd['stats_v2']
        assert stats_v2['numeric_float_1']['typing']['data_type'] == DATA_TYPES.NUMERIC
        assert stats_v2['numeric_float_1']['typing']['data_subtype'] == DATA_SUBTYPES.FLOAT
        assert stats_v2['numeric_float_1']['typing']['data_type_dist'][DATA_TYPES.NUMERIC] == 50
        assert stats_v2['numeric_float_1']['typing']['data_subtype_dist'][DATA_SUBTYPES.FLOAT] == 50

    def test_type_mix(self):
        predictor = Predictor(name='test_type_mix')
        predictor.breakpoint = 'TypeDeductor'

        n_points = 100
        df = pd.DataFrame({
            'numeric_float_1': np.linspace(0, n_points, n_points),
            'numeric_float_2': np.linspace(0, n_points, n_points),
            'numeric_float_3': np.linspace(0, n_points, n_points),
        })
        df['numeric_float_1'].iloc[:2] = 'random string'

        try:
            predictor.learn(
                from_data=df,
                to_predict='numeric_float_3',
                advanced_args={'force_column_usage': list(df.columns)}
            )
        except BreakpointException:
            pass
        else:
            raise AssertionError


        stats_v2 = predictor.transaction.lmd['stats_v2']
        assert stats_v2['numeric_float_1']['typing']['data_type'] == DATA_TYPES.NUMERIC
        assert stats_v2['numeric_float_1']['typing']['data_subtype'] == DATA_SUBTYPES.FLOAT
        assert stats_v2['numeric_float_1']['typing']['data_type_dist'][DATA_TYPES.NUMERIC] == 98
        assert stats_v2['numeric_float_1']['typing']['data_subtype_dist'][DATA_SUBTYPES.FLOAT] == 98

    def test_sample(self):
        sample_settings = {
            'sample_for_analysis': True,
            'sample_function': sample_data
        }
        sample_settings['sample_function'] = mock.MagicMock(wraps=sample_data)
        setattr(sample_settings['sample_function'], '__name__', 'sample_data')

        predictor = Predictor(name='test_sample_1')
        predictor.breakpoint = 'TypeDeductor'

        n_points = 100
        df = pd.DataFrame({
            'numeric_int_1': [x % 10 for x in list(range(n_points))],
            'numeric_int_2': [x % 10 for x in list(range(n_points))]
        })

        try:
            predictor.learn(
                from_data=df,
                to_predict='numeric_int_2',
                advanced_args={'force_column_usage': list(df.columns)},
                sample_settings=sample_settings
            )
        except BreakpointException:
            pass
        else:
            raise AssertionError

        assert sample_settings['sample_function'].called

        stats_v2 = predictor.transaction.lmd['stats_v2']
        assert stats_v2['numeric_int_1']['typing']['data_type'] == DATA_TYPES.NUMERIC
        assert stats_v2['numeric_int_1']['typing']['data_subtype'] == DATA_SUBTYPES.INT
        assert stats_v2['numeric_int_1']['typing']['data_type_dist'][DATA_TYPES.NUMERIC] <= n_points
        assert stats_v2['numeric_int_1']['typing']['data_subtype_dist'][DATA_SUBTYPES.INT] <= n_points

        sample_settings = {
            'sample_for_analysis': False,
            'sample_function': sample_data
        }
        sample_settings['sample_function'] = mock.MagicMock(wraps=sample_data)
        setattr(sample_settings['sample_function'], '__name__', 'sample_data')

        predictor = Predictor(name='test_sample_2')
        predictor.breakpoint = 'TypeDeductor'

        try:
            predictor.learn(
                from_data=df,
                to_predict='numeric_int_2',
                advanced_args={'force_column_usage': list(df.columns)},
                sample_settings=sample_settings
            )
        except BreakpointException:
            pass
        else:
            raise AssertionError

        assert not sample_settings['sample_function'].called

    def test_small_dataset_no_sampling(self):
        sample_settings = {
            'sample_for_analysis': False,
            'sample_function': mock.MagicMock(wraps=sample_data)
        }
        setattr(sample_settings['sample_function'], '__name__', 'sample_data')

        predictor = Predictor(name='test_small_dataset_no_sampling')
        predictor.breakpoint = 'TypeDeductor'

        n_points = 50
        df = pd.DataFrame({
            'numeric_int_1': [*range(n_points)],
            'numeric_int_2': [*range(n_points)],
        })

        try:
            predictor.learn(
                from_data=df,
                to_predict='numeric_int_2',
                advanced_args={'force_column_usage': list(df.columns)},
                sample_settings=sample_settings
            )
        except BreakpointException:
            pass
        else:
            raise AssertionError

        assert not sample_settings['sample_function'].called

        stats_v2 = predictor.transaction.lmd['stats_v2']

        assert stats_v2['numeric_int_1']['typing']['data_type'] == DATA_TYPES.NUMERIC
        assert stats_v2['numeric_int_1']['typing']['data_subtype'] == DATA_SUBTYPES.INT

        # This ensures that no sampling was applied
        assert stats_v2['numeric_int_1']['typing']['data_type_dist'][DATA_TYPES.NUMERIC] == n_points
        assert stats_v2['numeric_int_1']['typing']['data_subtype_dist'][DATA_SUBTYPES.INT] == n_points

    def test_date_formats(self):
        n_points = 50
        df = pd.DataFrame({
            'date': [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(n_points)],
            'datetime': [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%dT%H:%M') for i in range(n_points)],
        })

        predictor = Predictor(name='test_date_formats')
        predictor.breakpoint = 'TypeDeductor'

        try:
            predictor.learn(
                from_data=df,
                to_predict='datetime',
                advanced_args={'force_column_usage': list(df.columns)}
            )
        except BreakpointException:
            pass
        else:
            raise AssertionError

        assert predictor.transaction.lmd['stats_v2']['date']['typing']['data_type'] == DATA_TYPES.DATE
        assert predictor.transaction.lmd['stats_v2']['date']['typing']['data_subtype'] == DATA_SUBTYPES.DATE

        assert predictor.transaction.lmd['stats_v2']['datetime']['typing']['data_type'] == DATA_TYPES.DATE
        assert predictor.transaction.lmd['stats_v2']['datetime']['typing']['data_subtype'] == DATA_SUBTYPES.TIMESTAMP
'''
