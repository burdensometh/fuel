from collections import OrderedDict

import numpy
from numpy.testing import assert_raises, assert_equal

import h5py

from fuel.datasets import IndexableDataset
from fuel.datasets.hdf5 import H5PYDataset
from fuel.streams import DataStream
from fuel.schemes import (ConstantScheme, SequentialExampleScheme,
                          SequentialScheme, ShuffledExampleScheme,
                          ShuffledScheme, ConcatenatedScheme,
                          TimeSeriesCrossValidation, cross_validation)


def make_cross_validation(**kwargs):
    starts = [0, 2, 5, 7, 10, 3, 8, 12]
    ends = [2, 5, 7, 10, 12, 4, 9, 14]
    entities = ['A', 'A', 'B', 'B', 'C', 'C', 'A', 'D']
    defaults = dict(feature_starts=starts, label_ends=ends,
                    entities=entities,
                    validation_folds=[[2, 3], [1]])
    defaults.update(kwargs)
    return TimeSeriesCrossValidation(**defaults)


def iterator_requester(scheme):
    def get_request_iterator(*args, **kwargs):
        scheme_obj = scheme(*args, **kwargs)
        return scheme_obj.get_request_iterator()
    return get_request_iterator


def test_constant_scheme():
    get_request_iterator = iterator_requester(ConstantScheme)
    assert list(get_request_iterator(3, num_examples=7)) == [3, 3, 1]
    assert list(get_request_iterator(3, num_examples=9)) == [3, 3, 3]
    assert list(get_request_iterator(3, num_examples=2)) == [2]
    assert list(get_request_iterator(2, times=3)) == [2, 2, 2]
    assert list(get_request_iterator(3, times=1)) == [3]
    it = get_request_iterator(3)
    assert [next(it) == 3 for _ in range(10)]
    assert_raises(ValueError, get_request_iterator, 10, 2, 2)
    assert not ConstantScheme(3, 3).requests_examples


def test_sequential_scheme():
    get_request_iterator = iterator_requester(SequentialScheme)
    assert list(get_request_iterator(5, 3)) == [[0, 1, 2], [3, 4]]
    assert list(get_request_iterator(4, 2)) == [[0, 1], [2, 3]]
    assert list(get_request_iterator(
        [4, 3, 2, 1, 0], 3)) == [[4, 3, 2], [1, 0]]
    assert list(get_request_iterator(
        [3, 2, 1, 0], 2)) == [[3, 2], [1, 0]]
    assert not SequentialScheme(3, 3).requests_examples


def test_shuffled_scheme_sorted_indices():
    get_request_iterator = iterator_requester(ShuffledScheme)
    indices = list(range(7))
    rng = numpy.random.RandomState(3)
    test_rng = numpy.random.RandomState(3)
    test_rng.shuffle(indices)
    assert list(get_request_iterator(7, 3, rng=rng, sorted_indices=True)) == \
        [sorted(indices[:3]), sorted(indices[3:6]), sorted(indices[6:])]
    assert list(get_request_iterator(7, 3, rng=rng, sorted_indices=True)) != \
        [sorted(indices[:3]), sorted(indices[3:6]), sorted(indices[6:])]

    indices = list(range(6))[::-1]
    expected = indices[:]
    rng = numpy.random.RandomState(3)
    test_rng = numpy.random.RandomState(3)
    test_rng.shuffle(expected)
    assert (list(get_request_iterator(indices, 3, rng=rng,
                                      sorted_indices=True)) ==
            [sorted(expected[:3]), sorted(expected[3:6])])


def test_shuffled_scheme_unsorted_indices():
    get_request_iterator = iterator_requester(ShuffledScheme)
    indices = list(range(7))
    rng = numpy.random.RandomState(3)
    test_rng = numpy.random.RandomState(3)
    test_rng.shuffle(indices)
    assert list(get_request_iterator(7, 3, rng=rng, sorted_indices=False)) == \
        [indices[:3], indices[3:6], indices[6:]]
    assert list(get_request_iterator(7, 3, rng=rng, sorted_indices=False)) != \
        [indices[:3], indices[3:6], indices[6:]]

    indices = list(range(6))[::-1]
    expected = indices[:]
    rng = numpy.random.RandomState(3)
    test_rng = numpy.random.RandomState(3)
    test_rng.shuffle(expected)
    assert (list(get_request_iterator(indices, 3, rng=rng,
                                      sorted_indices=False)) ==
            [expected[:3], expected[3:6]])


def test_shuffled_scheme_requests_batches():
    assert not ShuffledScheme(3, 3).requests_examples


def test_shuffled_example_scheme():
    get_request_iterator = iterator_requester(ShuffledExampleScheme)
    indices = list(range(7))
    rng = numpy.random.RandomState(3)
    test_rng = numpy.random.RandomState(3)
    test_rng.shuffle(indices)
    assert list(get_request_iterator(7, rng=rng)) == indices


def test_shuffled_example_scheme_no_rng():
    scheme = ShuffledExampleScheme(7)
    assert scheme.rng is not None


def test_shuffled_example_scheme_requests_examples():
    assert ShuffledExampleScheme(3).requests_examples


def test_sequential_example_scheme():
    get_request_iterator = iterator_requester(SequentialExampleScheme)
    assert list(get_request_iterator(7)) == list(range(7))
    assert list(get_request_iterator(range(7)[::-1])) == list(range(7)[::-1])


def test_sequential_example_scheme_requests_examples():
    assert SequentialExampleScheme(3).requests_examples


def test_concatenated_scheme():
    sch = ConcatenatedScheme(schemes=[ConstantScheme(batch_size=10, times=5),
                                      ConstantScheme(batch_size=20, times=3),
                                      ConstantScheme(batch_size=30, times=1)])
    assert (list(sch.get_request_iterator()) ==
            ([10] * 5) + ([20] * 3) + [30])


def test_concatenated_scheme_raises_value_error_on_different_request_types():
    assert_raises(ValueError, ConcatenatedScheme,
                  [ConstantScheme(batch_size=10, times=5),
                   SequentialExampleScheme(examples=3)])


def test_concatenated_scheme_infers_request_type():
    assert not ConcatenatedScheme(
        schemes=[ConstantScheme(batch_size=10, times=5),
                 ConstantScheme(batch_size=10, times=5)]).requests_examples
    assert ConcatenatedScheme(
        schemes=[SequentialExampleScheme(examples=10),
                 SequentialExampleScheme(examples=10)]).requests_examples


def test_cross_validation():
    # test raise when strict=True
    cross = cross_validation(SequentialExampleScheme, 10, 3)
    assert_raises(ValueError, next, cross)

    # test IndexScheme when strict=False
    cross = cross_validation(SequentialExampleScheme, 10, 3, False)

    (train, valid, valid_size) = next(cross)
    assert list(train.get_request_iterator()) == list(range(3, 10))
    assert list(valid.get_request_iterator()) == list(range(0, 3))

    # test that indices are not depleted
    assert list(train.get_request_iterator()) == list(range(3, 10))
    assert list(valid.get_request_iterator()) == list(range(0, 3))
    assert valid_size == 3

    (train, valid, valid_size) = next(cross)
    assert (list(train.get_request_iterator()) ==
            list(range(0, 3)) + list(range(6, 10)))
    assert list(valid.get_request_iterator()) == list(range(3, 6))

    # test that indices are not depleted
    assert (list(train.get_request_iterator()) ==
            list(range(0, 3)) + list(range(6, 10)))
    assert list(valid.get_request_iterator()) == list(range(3, 6))
    assert valid_size == 3

    (train, valid, valid_size) = next(cross)
    assert list(train.get_request_iterator()) == list(range(0, 6))
    assert list(valid.get_request_iterator()) == list(range(6, 10))

    # test that indices are not depleted
    assert list(train.get_request_iterator()) == list(range(0, 6))
    assert list(valid.get_request_iterator()) == list(range(6, 10))
    assert valid_size == 4

    assert_raises(StopIteration, next, cross)

    # test BatchScheme
    cross = cross_validation(SequentialScheme, 8, 2, batch_size=2)

    (train, valid) = next(cross)
    assert list(train.get_request_iterator()) == [[4, 5], [6, 7]]
    assert list(valid.get_request_iterator()) == [[0, 1], [2, 3]]

    # test that indices are not depleted
    assert list(train.get_request_iterator()) == [[4, 5], [6, 7]]
    assert list(valid.get_request_iterator()) == [[0, 1], [2, 3]]

    (train, valid) = next(cross)
    assert list(train.get_request_iterator()) == [[0, 1], [2, 3]]
    assert list(valid.get_request_iterator()) == [[4, 5], [6, 7]]

    # test that indices are not depleted
    assert list(train.get_request_iterator()) == [[0, 1], [2, 3]]
    assert list(valid.get_request_iterator()) == [[4, 5], [6, 7]]

    assert_raises(StopIteration, next, cross)


class TestTimeSeriesCrossValidation(object):
    def test_fold_indices_purging_and_entity_exclusion(self):
        cv = make_cross_validation()
        fold0, fold1 = cv.get_fold_indices()
        # Fold 0 validates B examples 2,3: example 4 overlaps their
        # coverage ([7,10) vs [10,12) merely touches and is kept),
        # example 6 overlaps [7,10) with [8,9). Entities A and C are
        # both present in fold 1, but fold 0 only excludes B's entity,
        # so A/C examples are retained unless they overlap.
        assert_equal(fold0[0], numpy.array([0, 1, 4, 5, 7]))
        assert_equal(fold0[1], numpy.array([2, 3]))
        # Fold 1 validates example 1 (entity A, interval [2,5)). Entity
        # A examples 0 and 6 are excluded, example 5 overlaps in time.
        assert_equal(fold1[0], numpy.array([2, 3, 4, 7]))
        assert_equal(fold1[1], numpy.array([1]))
        assert cv.num_folds == 2
        assert cv.num_examples == 8

    def test_half_open_intervals_that_merely_touch_do_not_overlap(self):
        # [10, 12) validated, [12, 14) is a different entity and only
        # touches the boundary, so it remains in the training set.
        cv = TimeSeriesCrossValidation(
            feature_starts=[10, 12], label_ends=[12, 14],
            entities=['X', 'Y'], validation_folds=[[0]])
        train_indices, valid_indices = cv.get_fold_indices()[0]
        assert_equal(train_indices, numpy.array([1]))
        assert_equal(valid_indices, numpy.array([0]))

    def test_touching_validation_intervals_are_not_merged(self):
        # Two validation intervals [1, 3) and [3, 5) merely touch. They
        # must not be merged into [1, 5), which would wrongly exclude a
        # different-entity example lying in no validation interval.
        cv = TimeSeriesCrossValidation(
            feature_starts=[1, 3, 6], label_ends=[3, 5, 7],
            entities=['X', 'X', 'Y'], validation_folds=[[0, 1]])
        train_indices, _ = cv.get_fold_indices()[0]
        assert_equal(train_indices, numpy.array([2]))

    def test_empty_intervals_never_overlap(self):
        # A zero-length label window [t, t) covers no time and must
        # neither be excluded by overlapping validation intervals nor by
        # the embargo window, as long as its entity is clean.
        cv = TimeSeriesCrossValidation(
            feature_starts=[5, 8], label_ends=[5, 10],
            entities=['Y', 'X'], validation_folds=[[1]], embargo=10)
        train_indices, _ = cv.get_fold_indices()[0]
        assert_equal(train_indices, numpy.array([0]))
        # Merging drops empty intervals entirely.
        assert TimeSeriesCrossValidation._merge_intervals(
            [(5, 5), (8, 10)]) == [(8, 10)]

    def test_entity_exclusion_is_independent_of_time(self):
        # Example 0 is far away in time but shares entity A with the
        # validation example, hence it must be excluded.
        cv = TimeSeriesCrossValidation(
            feature_starts=[0, 100], label_ends=[1, 101],
            entities=['A', 'B'], validation_folds=[[1]])
        train_indices, _ = cv.get_fold_indices()[0]
        assert_equal(train_indices, numpy.array([0]))

        # With both examples belonging to entity A the fold has no
        # training example left, which must be reported explicitly.
        try:
            TimeSeriesCrossValidation(
                feature_starts=[0, 100], label_ends=[1, 101],
                entities=['A', 'A'], validation_folds=[[1]])
        except ValueError as error:
            assert 'empty training set' in str(error)
        else:
            raise AssertionError('expected ValueError')

    def test_embargo_excludes_window_after_latest_validation_end(self):
        cv = make_cross_validation(embargo=2)
        # Fold 0 latest validation end is 10. The embargo window is
        # [10, 12): example 4's interval [10, 12) is fully inside it.
        train0, valid0 = cv.get_fold_indices()[0]
        assert_equal(train0, numpy.array([0, 1, 5, 7]))
        assert_equal(valid0, numpy.array([2, 3]))
        assert cv.fold_exclusion_stats[0]['embargo'] == 1
        # Fold 1 latest validation end is 5, embargo window [5, 7):
        # example 2's interval [5, 7) is inside it.
        train1, _ = cv.get_fold_indices()[1]
        assert_equal(train1, numpy.array([3, 4, 7]))
        assert cv.fold_exclusion_stats[1]['embargo'] == 1

    def test_embargo_boundary_is_half_open(self):
        # Embargo window is [2, 3): an example starting exactly at 3 is
        # allowed, an example [2, 3) itself is excluded.
        cv = TimeSeriesCrossValidation(
            feature_starts=[0, 2, 3], label_ends=[2, 3, 4],
            entities=['A', 'B', 'C'], validation_folds=[[0]], embargo=1)
        train_indices, _ = cv.get_fold_indices()[0]
        assert_equal(train_indices, numpy.array([2]))

    def test_datetime64_intervals_and_timedelta64_embargo(self):
        starts = numpy.array(['2020-01-01', '2020-01-03', '2020-01-04'],
                             dtype='datetime64[D]')
        ends = numpy.array(['2020-01-02', '2020-01-04', '2020-01-05'],
                           dtype='datetime64[D]')
        cv = TimeSeriesCrossValidation(
            starts, ends, ['A', 'B', 'C'], validation_folds=[[1]],
            embargo=numpy.timedelta64(1, 'D'))
        # Validation covers [Jan 3, Jan 4); the embargo window
        # [Jan 4, Jan 5) excludes example 2, whose interval is exactly
        # [Jan 4, Jan 5), while example 0 stays in the training set.
        train_indices, valid_indices = cv.get_fold_indices()[0]
        assert_equal(train_indices, numpy.array([0]))
        assert_equal(valid_indices, numpy.array([1]))
        assert cv.fold_exclusion_stats[0]['embargo'] == 1

    def test_empty_training_fold_is_reported(self):
        assert_raises(
            ValueError, TimeSeriesCrossValidation,
            feature_starts=[0], label_ends=[1], entities=['A'],
            validation_folds=[[0]])
        try:
            TimeSeriesCrossValidation(
                feature_starts=[0, 1], label_ends=[1, 2],
                entities=['A', 'A'], validation_folds=[[1]])
        except ValueError as error:
            message = str(error)
            assert 'empty training set' in message
            assert 'fold 0' in message
        else:
            raise AssertionError('expected ValueError')

    def test_input_validation(self):
        kwargs = dict(feature_starts=[0, 1], label_ends=[1, 2],
                      entities=['A', 'B'], validation_folds=[[0]])
        # mismatched lengths
        assert_raises(
            ValueError, TimeSeriesCrossValidation,
            feature_starts=[0], label_ends=[1, 2], entities=['A', 'B'],
            validation_folds=[[0]])
        # start after end
        bad = dict(kwargs)
        bad['feature_starts'] = [2, 1]
        assert_raises(ValueError, TimeSeriesCrossValidation, **bad)
        # out-of-range index
        bad = dict(kwargs)
        bad['validation_folds'] = [[3]]
        assert_raises(ValueError, TimeSeriesCrossValidation, **bad)
        # duplicate index within a fold
        bad = dict(kwargs)
        bad['validation_folds'] = [[0, 0]]
        assert_raises(ValueError, TimeSeriesCrossValidation, **bad)
        # index shared across folds
        bad = dict(kwargs)
        bad['validation_folds'] = [[0], [0]]
        assert_raises(ValueError, TimeSeriesCrossValidation, **bad)
        # empty fold
        bad = dict(kwargs)
        bad['validation_folds'] = [[]]
        assert_raises(ValueError, TimeSeriesCrossValidation, **bad)
        # no folds at all
        bad = dict(kwargs)
        bad['validation_folds'] = []
        assert_raises(ValueError, TimeSeriesCrossValidation, **bad)
        # negative embargo
        bad = dict(kwargs)
        bad['embargo'] = -1
        assert_raises(ValueError, TimeSeriesCrossValidation, **bad)
        # 2D inputs
        assert_raises(
            ValueError, TimeSeriesCrossValidation,
            feature_starts=[[0, 1]], label_ends=[1, 2],
            entities=['A', 'B'], validation_folds=[[0]])

    def test_returned_schemes_drive_a_data_stream(self):
        dataset = IndexableDataset(OrderedDict([
            ('features', numpy.arange(16, dtype='float32').reshape(8, 2)),
            ('targets', numpy.arange(8, dtype='int64').reshape(8, 1))]))
        cv = make_cross_validation(scheme_class=SequentialScheme,
                                   scheme_kwargs={'batch_size': 2})
        train_scheme, valid_scheme = next(iter(cv))
        train_stream = DataStream(dataset, iteration_scheme=train_scheme)
        valid_stream = DataStream(dataset, iteration_scheme=valid_scheme)
        assert train_stream.sources == ('features', 'targets')
        train_features = numpy.concatenate(
            [batch[0] for batch in
             train_stream.get_epoch_iterator()], axis=0)
        assert_equal(train_features[:, 0] // 2,
                     numpy.array([0, 1, 4, 5, 7]))
        valid_targets = numpy.concatenate(
            [batch[1] for batch in
             valid_stream.get_epoch_iterator()], axis=0)
        assert_equal(valid_targets.ravel(), numpy.array([2, 3]))
        # Iterating again must produce the same indices.
        train_features_again = numpy.concatenate(
            [batch[0] for batch in
             train_stream.get_epoch_iterator()], axis=0)
        assert_equal(train_features, train_features_again)

    def test_default_scheme_requests_examples(self):
        cv = make_cross_validation()
        train_scheme, valid_scheme = next(iter(cv))
        assert train_scheme.requests_examples
        assert valid_scheme.requests_examples
        assert list(train_scheme.get_request_iterator()) == [0, 1, 4, 5, 7]
        assert list(valid_scheme.get_request_iterator()) == [2, 3]

    def test_save_hdf5_splits_round_trip(self):
        cv = make_cross_validation()
        features = numpy.arange(24, dtype='uint8').reshape(8, 3)
        targets = numpy.arange(8, dtype='uint8').reshape(8, 1)
        h5file = h5py.File(
            'time_series_splits.hdf5', mode='w', driver='core',
            backing_store=False)
        h5file['features'] = features
        h5file['features'].dims[0].label = 'batch'
        h5file['features'].dims[1].label = 'feature'
        h5file['targets'] = targets
        h5file['targets'].dims[0].label = 'batch'
        h5file['targets'].dims[1].label = 'index'

        split_names = cv.save_hdf5_splits(
            h5file, sources=('features', 'targets'))
        assert split_names == ('fold0_train', 'fold0_valid',
                               'fold1_train', 'fold1_valid')
        # All sources of a split point at the same index dataset.
        split_attrs = h5file.attrs['split']
        fold0_rows = [row for row in split_attrs
                      if row['split'].decode('utf8') == 'fold0_train']
        assert len(fold0_rows) == 2
        assert h5file[fold0_rows[0]['indices']][:].tolist() == \
            h5file[fold0_rows[1]['indices']][:].tolist() == \
            [0, 1, 4, 5, 7]

        # Reopening through H5PYDataset must give back the same indices
        # and the two sources must return the same real examples.
        train_set = H5PYDataset(h5file, which_sets=('fold0_train',))
        assert train_set.num_examples == 5
        handle = train_set.open()
        data = train_set.get_data(handle, slice(0, 5))
        train_set.close(handle)
        expected_train = numpy.array([0, 1, 4, 5, 7])
        assert_equal(data[0], features[expected_train])
        assert_equal(data[1], targets[expected_train])

        valid_set = H5PYDataset(h5file, which_sets=('fold1_valid',))
        handle = valid_set.open()
        data = valid_set.get_data(handle, slice(0, 1))
        valid_set.close(handle)
        assert_equal(data[0], features[[1]])
        assert_equal(data[1], targets[[1]])
        h5file.close()

    def test_save_hdf5_splits_to_path_and_reopen(self, tmp_path):
        cv = make_cross_validation()
        path = str(tmp_path / 'folded.hdf5')
        features = numpy.arange(16, dtype='uint8').reshape(8, 2)
        targets = numpy.arange(8, dtype='uint8').reshape(8, 1)
        with h5py.File(path, mode='w') as h5file:
            h5file['features'] = features
            h5file['targets'] = targets
        cv.save_hdf5_splits(path, sources=('features', 'targets'))

        # Reopen the persisted file and check every fold round-trips.
        expected = cv.get_fold_indices()
        for fold_number, (train_indices, valid_indices) in enumerate(
                expected):
            train_set = H5PYDataset(
                path, which_sets=('fold{}_train'.format(fold_number),))
            handle = train_set.open()
            train_data = train_set.get_data(
                handle, slice(0, len(train_indices)))
            train_set.close(handle)
            assert_equal(train_data[0], features[train_indices])
            assert_equal(train_data[1], targets[train_indices])

            valid_set = H5PYDataset(
                path, which_sets=('fold{}_valid'.format(fold_number),))
            handle = valid_set.open()
            valid_data = valid_set.get_data(
                handle, slice(0, len(valid_indices)))
            valid_set.close(handle)
            assert_equal(valid_data[0], features[valid_indices])
            assert_equal(valid_data[1], targets[valid_indices])

    def test_save_hdf5_splits_overwrite_guard(self):
        cv = make_cross_validation()
        h5file = h5py.File(
            'time_series_overwrite.hdf5', mode='w', driver='core',
            backing_store=False)
        h5file['features'] = numpy.arange(8, dtype='uint8').reshape(8, 1)
        h5file['targets'] = numpy.arange(8, dtype='uint8').reshape(8, 1)
        cv.save_hdf5_splits(h5file, sources=('features', 'targets'))
        assert_raises(
            ValueError, cv.save_hdf5_splits,
            h5file, ('features', 'targets'))
        cv.save_hdf5_splits(
            h5file, ('features', 'targets'), overwrite=True)
        assert set(H5PYDataset.get_all_splits(h5file)) == {
            'fold0_train', 'fold0_valid', 'fold1_train', 'fold1_valid'}
        h5file.close()

    def test_save_hdf5_splits_rejects_missing_or_mismatched_source(self):
        cv = make_cross_validation()
        h5file = h5py.File(
            'time_series_bad_source.hdf5', mode='w', driver='core',
            backing_store=False)
        h5file['features'] = numpy.arange(
            12, dtype='uint8').reshape(4, 3)
        assert_raises(
            ValueError, cv.save_hdf5_splits, h5file, ('targets',))
        assert_raises(
            ValueError, cv.save_hdf5_splits, h5file, ('features',))
        assert_raises(
            ValueError, cv.save_hdf5_splits, h5file, ())
        h5file.close()
