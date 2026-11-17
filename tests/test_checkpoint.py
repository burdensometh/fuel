"""Tests for explicit batch acknowledgement and checkpointing.

These tests exercise the deterministic chain
``H5PYDataset -> Sequential/ShuffledScheme -> Batch/Padding`` and verify
that a restored stream produces exactly the in-memory continuation of an
uninterrupted run (sample order, labels, masks and the trailing partial
batch), that unacknowledged batches are re-sent, and that incompatible
inputs or configurations are refused instead of silently resuming.
"""
import json
import os
import tempfile

import h5py
import numpy
from numpy.testing import assert_equal, assert_raises

from fuel import config
from fuel.checkpoint import (load_checkpoint, save_checkpoint)
from fuel.datasets import IndexableDataset
from fuel.datasets.hdf5 import H5PYDataset
from fuel.exceptions import (CheckpointNotSupportedError,
                             IncompatibleCheckpointError)
from fuel.schemes import SequentialScheme, ShuffledScheme
from fuel.streams import DataStream
from fuel.transformers import Batch, Cache, Mapping, Padding, Unpack


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_floatx():
    if not hasattr(config, 'floatX'):
        config.floatX = 'float64'


def _make_indexable(num_examples=23, seq_len=True):
    if seq_len:
        features = [numpy.arange(k % 5 + 1, dtype='float32')
                    for k in range(num_examples)]
    else:
        features = numpy.arange(num_examples, dtype='float32').reshape(-1, 1)
    targets = numpy.arange(num_examples, dtype='int32').reshape(-1, 1)
    return IndexableDataset({'features': features, 'targets': targets})


def _build_h5py_vlen(filename):
    lengths = [3, 5, 1, 7, 2, 4, 6]
    features = [numpy.arange(n, dtype='uint8') for n in lengths]
    targets = numpy.arange(len(lengths), dtype='uint8').reshape(-1, 1)
    h5file = h5py.File(filename, mode='w', driver='core',
                       backing_store=True)
    dtype = h5py.special_dtype(vlen=numpy.dtype('uint8'))
    dataset = h5file.create_dataset('features', (len(lengths),), dtype=dtype)
    dataset[...] = features
    dataset.dims[0].label = 'batch'
    shapes = h5file.create_dataset('features_shapes', (len(lengths), 1),
                                   dtype='uint8')
    shapes[...] = numpy.array([[n] for n in lengths])
    dataset.dims.create_scale(shapes, 'shapes')
    dataset.dims[0].attach_scale(shapes)
    shape_labels = h5file.create_dataset(
        'features_shape_labels', (1,), dtype='S7')
    shape_labels[...] = [b'length']
    dataset.dims.create_scale(shape_labels, 'shape_labels')
    dataset.dims[0].attach_scale(shape_labels)
    target_dataset = h5file.create_dataset(
        'targets', (len(lengths), 1), dtype='uint8')
    target_dataset[...] = targets
    target_dataset.dims[0].label = 'batch'
    target_dataset.dims[1].label = 'index'
    split_dict = {'train': {'features': (0, len(lengths)),
                            'targets': (0, len(lengths))}}
    h5file.attrs['split'] = H5PYDataset.create_split_array(split_dict)
    return h5file


def _assert_batches_equal(actual, expected):
    assert_equal(len(actual), len(expected))
    for actual_batch, expected_batch in zip(actual, expected):
        assert_equal(len(actual_batch), len(expected_batch))
        for actual_source, expected_source in zip(actual_batch,
                                                  expected_batch):
            assert_equal(actual_source, expected_source)


def _json_round_trip(checkpoint):
    return json.loads(json.dumps(checkpoint))


def _run_epochs(stream, num_epochs, acknowledge=True):
    """Collect epochs, acknowledging every batch when requested."""
    epochs = []
    for _ in range(num_epochs):
        epoch = []
        for batch in stream.get_epoch_iterator():
            epoch.append(batch)
            if acknowledge:
                stream.acknowledge()
        epochs.append(epoch)
    return epochs


# ---------------------------------------------------------------------------
# Checkpoint format and basic semantics
# ---------------------------------------------------------------------------

class TestCheckpointFormat(object):
    def setUp(self):
        _require_floatx()

    def test_checkpoint_is_json_serializable(self):
        dataset = _make_indexable(num_examples=10, seq_len=False)
        stream = DataStream(
            dataset,
            iteration_scheme=SequentialScheme(10, 2))
        stream.get_epoch_iterator()
        checkpoint = stream.get_checkpoint()
        # Must not raise.
        assert isinstance(_json_round_trip(checkpoint), dict)

    def test_checkpoint_records_epoch_and_acknowledged_count(self):
        dataset = _make_indexable(num_examples=10, seq_len=False)
        stream = DataStream(
            dataset,
            iteration_scheme=SequentialScheme(10, 2))
        iterator = stream.get_epoch_iterator()
        next(iterator)
        next(iterator)
        stream.acknowledge()
        checkpoint = stream.get_checkpoint()
        assert checkpoint['acknowledged'] == 2
        assert checkpoint['epoch'] == 0
        assert checkpoint['fuel_checkpoint_version'] == 1

    def test_save_and_load_round_trip(self):
        dataset = _make_indexable(num_examples=10, seq_len=False)
        stream = DataStream(
            dataset,
            iteration_scheme=SequentialScheme(10, 2))
        stream.get_epoch_iterator()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as f:
            filename = f.name
        try:
            save_checkpoint(stream.get_checkpoint(), filename)
            loaded = load_checkpoint(filename)
            assert loaded['acknowledged'] == 0
        finally:
            os.remove(filename)

    def test_acknowledge_only_counts_delivered_batches(self):
        dataset = _make_indexable(num_examples=10, seq_len=False)
        stream = DataStream(
            dataset,
            iteration_scheme=SequentialScheme(10, 2))
        iterator = stream.get_epoch_iterator()
        next(iterator)
        # No acknowledge: nothing is acknowledged yet.
        assert stream.get_checkpoint()['acknowledged'] == 0
        stream.acknowledge()
        assert stream.get_checkpoint()['acknowledged'] == 1
        next(iterator)
        # The newly delivered, unacknowledged batch is not counted.
        assert stream.get_checkpoint()['acknowledged'] == 1

    def test_read_ahead_is_not_reported_as_consumed(self):
        # A stream whose iterator was created but never read must report
        # zero consumed batches, even though the request iterator advanced.
        dataset = _make_indexable(num_examples=10, seq_len=False)
        stream = DataStream(
            dataset,
            iteration_scheme=SequentialScheme(10, 2))
        stream.get_epoch_iterator()
        assert stream.get_checkpoint()['acknowledged'] == 0


# ---------------------------------------------------------------------------
# Resuming reproduces an uninterrupted run
# ---------------------------------------------------------------------------

class TestSequentialResume(object):
    def setUp(self):
        _require_floatx()

    def test_resume_sequential_batches(self):
        dataset = _make_indexable(num_examples=10, seq_len=False)
        reference = _run_epochs(
            DataStream(dataset,
                       iteration_scheme=SequentialScheme(10, 4)), 2)[0]

        stream = DataStream(dataset,
                            iteration_scheme=SequentialScheme(10, 4))
        iterator = stream.get_epoch_iterator()
        for _ in range(2):
            next(iterator)
            stream.acknowledge()
        checkpoint = _json_round_trip(stream.get_checkpoint())

        restored = DataStream(dataset,
                              iteration_scheme=SequentialScheme(10, 4))
        restored.set_checkpoint(checkpoint)
        resumed = list(restored.get_epoch_iterator())
        _assert_batches_equal(resumed, reference[2:])

    def test_unacknowledged_batch_is_sent_again(self):
        dataset = _make_indexable(num_examples=10, seq_len=False)
        reference = _run_epochs(
            DataStream(dataset,
                       iteration_scheme=SequentialScheme(10, 2)), 1)[0]

        stream = DataStream(dataset,
                            iteration_scheme=SequentialScheme(10, 2))
        iterator = stream.get_epoch_iterator()
        next(iterator)
        stream.acknowledge()
        next(iterator)  # delivered but not acknowledged
        next(iterator)  # delivered but not acknowledged
        checkpoint = _json_round_trip(stream.get_checkpoint())

        restored = DataStream(dataset,
                              iteration_scheme=SequentialScheme(10, 2))
        restored.set_checkpoint(checkpoint)
        resumed = list(restored.get_epoch_iterator())
        # The two unacknowledged batches are sent again.
        _assert_batches_equal(resumed, reference[1:])
        assert_equal(resumed[0][0], reference[1][0])

    def test_trailing_partial_batch_matches(self):
        dataset = _make_indexable(num_examples=10, seq_len=False)
        reference = _run_epochs(
            DataStream(dataset,
                       iteration_scheme=SequentialScheme(10, 4)), 1)[0]
        # 10 examples in batches of 4 -> trailing batch of size 2.
        assert_equal(len(reference[-1][0]), 2)

        stream = DataStream(dataset,
                            iteration_scheme=SequentialScheme(10, 4))
        iterator = stream.get_epoch_iterator()
        next(iterator)
        stream.acknowledge()
        restored = DataStream(dataset,
                              iteration_scheme=SequentialScheme(10, 4))
        restored.set_checkpoint(_json_round_trip(stream.get_checkpoint()))
        resumed = list(restored.get_epoch_iterator())
        _assert_batches_equal(resumed, reference[1:])
        assert_equal(len(resumed[-1][0]), 2)


class TestShuffledResume(object):
    def setUp(self):
        _require_floatx()

    def _stream(self, dataset, seed=7, batch_size=4):
        return DataStream(
            dataset,
            iteration_scheme=ShuffledScheme(
                dataset.num_examples, batch_size,
                rng=numpy.random.RandomState(seed)))

    def test_resume_preserves_sample_order(self):
        dataset = _make_indexable(num_examples=23, seq_len=False)
        reference = _run_epochs(self._stream(dataset), 1)[0]

        stream = self._stream(dataset)
        iterator = stream.get_epoch_iterator()
        for _ in range(3):
            next(iterator)
            stream.acknowledge()
        checkpoint = _json_round_trip(stream.get_checkpoint())

        restored = self._stream(dataset)
        restored.set_checkpoint(checkpoint)
        resumed = list(restored.get_epoch_iterator())
        _assert_batches_equal(resumed, reference[3:])

    def test_resume_restores_rng_for_following_epochs(self):
        dataset = _make_indexable(num_examples=17, seq_len=False)
        reference = _run_epochs(self._stream(dataset), 3)

        stream = self._stream(dataset)
        # Finish epoch 0, consume 2 batches of epoch 1.
        for batch in stream.get_epoch_iterator():
            stream.acknowledge()
        iterator = stream.get_epoch_iterator()
        for _ in range(2):
            next(iterator)
            stream.acknowledge()
        checkpoint = _json_round_trip(stream.get_checkpoint())
        assert checkpoint['epoch'] == 1

        restored = self._stream(dataset)
        restored.set_checkpoint(checkpoint)
        _assert_batches_equal(list(restored.get_epoch_iterator()),
                              reference[1][2:])
        # The next epoch after resume must still match the reference.
        _assert_batches_equal(list(restored.get_epoch_iterator()),
                              reference[2])

    def test_resume_at_zero_preserves_whole_epoch_order(self):
        dataset = _make_indexable(num_examples=13, seq_len=False)
        reference = _run_epochs(self._stream(dataset), 1)[0]
        stream = self._stream(dataset)
        stream.get_epoch_iterator()  # created but not read
        restored = self._stream(dataset)
        restored.set_checkpoint(
            _json_round_trip(stream.get_checkpoint()))
        _assert_batches_equal(list(restored.get_epoch_iterator()), reference)


# ---------------------------------------------------------------------------
# Batch + Padding chain (labels and masks)
# ---------------------------------------------------------------------------

class TestBatchPaddingResume(object):
    def setUp(self):
        _require_floatx()

    def _padding_stream(self, dataset, seed=3):
        data_stream = DataStream(
            dataset,
            iteration_scheme=ShuffledScheme(
                dataset.num_examples, 3,
                rng=numpy.random.RandomState(seed)))
        return Padding(data_stream)

    def test_resume_reproduces_labels_and_masks(self):
        dataset = _make_indexable(num_examples=19)
        reference = _run_epochs(self._padding_stream(dataset), 2)

        stream = self._padding_stream(dataset)
        stream.get_epoch_iterator()  # epoch 0, read nothing
        # consume epoch 1 partially
        iterator = stream.get_epoch_iterator()
        for _ in range(2):
            next(iterator)
            stream.acknowledge()
        checkpoint = _json_round_trip(stream.get_checkpoint())
        assert checkpoint['epoch'] == 1

        restored = self._padding_stream(dataset)
        restored.set_checkpoint(checkpoint)
        resumed = list(restored.get_epoch_iterator())
        _assert_batches_equal(resumed, reference[1][2:])
        # Masks are present and identical (source named `*_mask`).
        assert 'features_mask' in restored.sources
        assert 'targets_mask' in restored.sources

    def test_trailing_batch_with_padding_matches(self):
        dataset = _make_indexable(num_examples=10)
        reference = _run_epochs(self._padding_stream(dataset), 1)[0]
        # 10 examples, batch size 3 -> trailing batch of size 1.
        stream = self._padding_stream(dataset)
        iterator = stream.get_epoch_iterator()
        total = 0
        while True:
            try:
                batch = next(iterator)
            except StopIteration:
                break
            total += 1
            stream.acknowledge()
        assert total == len(reference)
        assert_equal(len(reference[-1][0]), 1)


# ---------------------------------------------------------------------------
# H5PYDataset chain
# ---------------------------------------------------------------------------

class TestH5PYResume(object):
    def setUp(self):
        _require_floatx()
        handle, self.filename = tempfile.mkstemp(suffix='.hdf5')
        os.close(handle)
        self.h5file = _build_h5py_vlen(self.filename)

    def tearDown(self):
        self.h5file.close()
        os.remove(self.filename)

    def _stream(self, **dataset_kwargs):
        dataset = H5PYDataset(self.h5file, which_sets=('train',),
                              **dataset_kwargs)
        return Padding(DataStream(
            dataset,
            iteration_scheme=ShuffledScheme(
                dataset.num_examples, 2,
                rng=numpy.random.RandomState(5))))

    def test_resume_reproduces_vlen_data_and_masks(self):
        reference = _run_epochs(self._stream(), 2)

        stream = self._stream()
        stream.get_epoch_iterator()
        iterator = stream.get_epoch_iterator()
        next(iterator)
        stream.acknowledge()
        checkpoint = _json_round_trip(stream.get_checkpoint())

        restored = self._stream()
        restored.set_checkpoint(checkpoint)
        _assert_batches_equal(list(restored.get_epoch_iterator()),
                              reference[1][1:])

    def test_changed_subset_is_refused(self):
        stream = self._stream()
        stream.get_epoch_iterator()
        next(stream.get_epoch_iterator())
        stream.acknowledge()
        checkpoint = _json_round_trip(stream.get_checkpoint())

        other = self._stream(subset=slice(0, 5))
        assert_raises(IncompatibleCheckpointError, other.set_checkpoint,
                      checkpoint)

    def test_changed_file_content_is_refused(self):
        stream = self._stream()
        iterator = stream.get_epoch_iterator()
        next(iterator)
        stream.acknowledge()
        checkpoint = _json_round_trip(stream.get_checkpoint())
        stream.close()

        # Rewrite the file with different values.
        self.h5file.close()
        self.h5file = _build_h5py_vlen(self.filename)
        features = self.h5file['features']
        features[0] = numpy.full(len(features[0]), 99, dtype='uint8')
        self.h5file.close()
        self.h5file = h5py.File(self.filename, mode='r')

        other = self._stream()
        assert_raises(IncompatibleCheckpointError, other.set_checkpoint,
                      checkpoint)


# ---------------------------------------------------------------------------
# Fingerprint binding: refuse to resume on changed input/configuration
# ---------------------------------------------------------------------------

class TestIncompatibleCheckpoint(object):
    def setUp(self):
        _require_floatx()
        self.dataset = _make_indexable(num_examples=20, seq_len=False)

    def _checkpoint(self, batch_size=4, seed=2):
        stream = DataStream(
            self.dataset,
            iteration_scheme=ShuffledScheme(
                20, batch_size, rng=numpy.random.RandomState(seed)))
        iterator = stream.get_epoch_iterator()
        next(iterator)
        stream.acknowledge()
        return _json_round_trip(stream.get_checkpoint())

    def test_changed_batch_size_is_refused(self):
        checkpoint = self._checkpoint(batch_size=4)
        other = DataStream(
            self.dataset,
            iteration_scheme=ShuffledScheme(
                20, 5, rng=numpy.random.RandomState(2)))
        assert_raises(IncompatibleCheckpointError, other.set_checkpoint,
                      checkpoint)

    def test_changed_scheme_is_refused(self):
        checkpoint = self._checkpoint()
        other = DataStream(
            self.dataset,
            iteration_scheme=SequentialScheme(20, 4))
        assert_raises(IncompatibleCheckpointError, other.set_checkpoint,
                      checkpoint)

    def test_changed_dataset_is_refused(self):
        checkpoint = self._checkpoint()
        other_dataset = _make_indexable(num_examples=21, seq_len=False)
        other = DataStream(
            other_dataset,
            iteration_scheme=ShuffledScheme(
                21, 4, rng=numpy.random.RandomState(2)))
        assert_raises(IncompatibleCheckpointError, other.set_checkpoint,
                      checkpoint)

    def test_changed_environment_is_refused(self):
        checkpoint = self._checkpoint()
        other = DataStream(
            self.dataset,
            iteration_scheme=ShuffledScheme(
                20, 4, rng=numpy.random.RandomState(2)))
        # Pretend the checkpoint was taken under a different default seed.
        checkpoint['environment']['default_seed'] += 1
        assert_raises(IncompatibleCheckpointError, other.set_checkpoint,
                      checkpoint)

    def test_changed_transformer_chain_is_refused(self):
        checkpoint = self._checkpoint()
        padded = Padding(DataStream(
            self.dataset,
            iteration_scheme=ShuffledScheme(
                20, 4, rng=numpy.random.RandomState(2))))
        assert_raises(IncompatibleCheckpointError, padded.set_checkpoint,
                      checkpoint)

    def test_unsupported_checkpoint_version_is_refused(self):
        checkpoint = self._checkpoint()
        checkpoint['fuel_checkpoint_version'] = 999
        other = DataStream(
            self.dataset,
            iteration_scheme=ShuffledScheme(
                20, 4, rng=numpy.random.RandomState(2)))
        assert_raises(IncompatibleCheckpointError, other.set_checkpoint,
                      checkpoint)


# ---------------------------------------------------------------------------
# Unsupported components must fail before anything is written
# ---------------------------------------------------------------------------

class TestUnsupportedComponents(object):
    def setUp(self):
        _require_floatx()

    def test_cache_is_rejected(self):
        stream = Batch(DataStream(_make_indexable(num_examples=20)),
                       __import__('fuel.schemes',
                                  fromlist=['ConstantScheme']
                                  ).ConstantScheme(5))
        cached = Cache(stream, __import__(
            'fuel.schemes', fromlist=['ConstantScheme']).ConstantScheme(2))
        assert_raises(CheckpointNotSupportedError, cached.get_checkpoint)

    def test_mapping_is_rejected(self):
        stream = DataStream(
            _make_indexable(num_examples=20, seq_len=False),
            iteration_scheme=SequentialScheme(20, 2))
        mapped = Mapping(stream, lambda data: data)
        assert_raises(CheckpointNotSupportedError, mapped.get_checkpoint)

    def test_unpack_is_rejected(self):
        stream = Batch(DataStream(_make_indexable(num_examples=20)),
                       __import__('fuel.schemes',
                                  fromlist=['ConstantScheme']
                                  ).ConstantScheme(5))
        unpacked = Unpack(stream)
        assert_raises(CheckpointNotSupportedError, unpacked.get_checkpoint)

    def test_error_is_raised_before_any_output_is_produced(self):
        stream = DataStream(
            _make_indexable(num_examples=20, seq_len=False),
            iteration_scheme=SequentialScheme(20, 2))
        mapped = Mapping(stream, lambda data: data)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as f:
            filename = f.name
        try:
            assert_raises(CheckpointNotSupportedError,
                          mapped.get_checkpoint)
            # Nothing was written because get_checkpoint raised before
            # save_checkpoint could be called.
            assert os.path.getsize(filename) == 0
        finally:
            os.remove(filename)


# ---------------------------------------------------------------------------
# Acknowledgement is required across the whole chain
# ---------------------------------------------------------------------------

class TestAcknowledgement(object):
    def setUp(self):
        _require_floatx()

    def test_acknowledge_through_outer_transformer(self):
        dataset = _make_indexable(num_examples=12, seq_len=False)
        data_stream = DataStream(
            dataset, iteration_scheme=SequentialScheme(12, 3))
        padded = Padding(data_stream)
        iterator = padded.get_epoch_iterator()
        next(iterator)
        padded.acknowledge()
        assert padded.get_checkpoint()['acknowledged'] == 1

    def test_iterator_acknowledge_shortcut(self):
        dataset = _make_indexable(num_examples=12, seq_len=False)
        stream = DataStream(dataset,
                            iteration_scheme=SequentialScheme(12, 3))
        iterator = stream.get_epoch_iterator()
        next(iterator)
        iterator.acknowledge()
        assert stream.get_checkpoint()['acknowledged'] == 1

    def test_resume_after_acknowledge_through_transformer(self):
        dataset = _make_indexable(num_examples=16)
        reference = _run_epochs(
            Padding(DataStream(
                dataset, iteration_scheme=SequentialScheme(16, 4))), 1)[0]

        stream = Padding(DataStream(
            dataset, iteration_scheme=SequentialScheme(16, 4)))
        iterator = stream.get_epoch_iterator()
        next(iterator)
        stream.acknowledge()
        restored = Padding(DataStream(
            dataset, iteration_scheme=SequentialScheme(16, 4)))
        restored.set_checkpoint(_json_round_trip(stream.get_checkpoint()))
        _assert_batches_equal(list(restored.get_epoch_iterator()),
                              reference[1:])
