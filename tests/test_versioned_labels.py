import os

import h5py
import numpy
from numpy.testing import assert_equal, assert_raises
from six.moves import cPickle

from fuel.datasets.hdf5 import H5PYDataset
from fuel.datasets.versioned_labels import (
    LabelRevisionStore, VersionedLabelsH5PYDataset)
from fuel.exceptions import LabelRevisionError
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme


def make_feature_file(path, backing_store=True):
    """Creates a small feature file with stable string IDs and labels."""
    num_examples = 30
    ids = numpy.array(
        ['id_{:03d}'.format(i).encode('utf8') for i in range(num_examples)],
        dtype='S10')
    features = numpy.arange(num_examples * 4,
                            dtype='uint16').reshape(num_examples, 4)
    targets = (numpy.arange(num_examples, dtype='uint8') * 2).reshape(
        num_examples, 1)
    h5file = h5py.File(path, mode='w', driver='core',
                       backing_store=backing_store)
    h5file['ids'] = ids
    h5file['features'] = features
    h5file['features'].dims[0].label = 'batch'
    h5file['features'].dims[1].label = 'feature'
    h5file['targets'] = targets
    h5file['targets'].dims[0].label = 'batch'
    h5file['targets'].dims[1].label = 'index'
    split_dict = {'train': {'ids': (0, 20), 'features': (0, 20),
                            'targets': (0, 20)},
                  'test': {'ids': (20, 30), 'features': (20, 30),
                           'targets': (20, 30)}}
    h5file.attrs['split'] = H5PYDataset.create_split_array(split_dict)
    return h5file


class TestLabelRevisionStore(object):
    def setUp(self):
        self.h5file = make_feature_file(
            'versioned_features.hdf5', backing_store=False)
        self.original_targets = self.h5file['targets'][...]
        self.ids = self.h5file['ids'][...]
        self.store_path = 'labels.sidecar.hdf5'
        self.store = LabelRevisionStore.create(
            self.store_path, self.h5file, 'ids', 'targets')

    def tearDown(self):
        self.h5file.close()
        for path in (self.store_path, self.store_path + '.lock'):
            if os.path.exists(path):
                os.remove(path)

    def test_create_snapshots_base_version(self):
        assert self.store.latest_version == 0
        assert self.store.versions == ()
        assert_equal(self.store.base_ids(), self.ids)
        assert_equal(self.store.base_labels(), self.original_targets)
        assert self.store.id_source == 'ids'
        assert self.store.label_source == 'targets'

    def test_create_refuses_existing_file(self):
        assert_raises(IOError, LabelRevisionStore.create,
                      self.store_path, self.h5file, 'ids', 'targets')

    def test_create_from_path(self):
        disk_file = make_feature_file(
            'disk_features.hdf5', backing_store=True)
        disk_file.close()
        try:
            store = LabelRevisionStore.create(
                'labels_from_path.hdf5', 'disk_features.hdf5',
                'ids', 'targets')
            assert_equal(store.base_labels(), self.original_targets)
        finally:
            os.remove('labels_from_path.hdf5')
            os.remove('disk_features.hdf5')

    def test_missing_store_file_raises(self):
        assert_raises(IOError, LabelRevisionStore, 'does_not_exist.hdf5')

    def test_publish_first_revision(self):
        changed_ids = self.ids[[1, 3]]
        new_labels = numpy.array([[101], [103]], dtype='uint8')
        version = self.store.publish(changed_ids, new_labels)
        assert version == 1
        assert self.store.latest_version == 1
        assert self.store.versions == (1,)
        # Only the changed IDs are physically stored.
        with h5py.File(self.store_path, mode='r') as handle:
            assert_equal(handle['versions/1/ids'][...], changed_ids)
        # Resolution gives a complete array, untouched IDs carry over.
        resolved = self.store.resolve(1)
        expected = self.original_targets.copy()
        expected[[1, 3]] = new_labels
        assert_equal(resolved, expected)

    def test_publish_auto_increments_versions(self):
        self.store.publish(self.ids[[1]], numpy.array([[7]], dtype='uint8'))
        version = self.store.publish(
            self.ids[[2]], numpy.array([[8]], dtype='uint8'))
        assert version == 2
        assert self.store.versions == (1, 2)
        # Revision 2 inherits revision 1's change.
        resolved = self.store.resolve(2)
        assert resolved[1, 0] == 7
        assert resolved[2, 0] == 8
        assert resolved[0, 0] == self.original_targets[0, 0]

    def test_explicit_version_must_be_incremental(self):
        assert_raises(LabelRevisionError, self.store.publish,
                      self.ids[[1]], numpy.array([[7]], dtype='uint8'),
                      version=5)
        assert self.store.latest_version == 0
        assert self.store.versions == ()
        self.store.publish(
            self.ids[[1]], numpy.array([[7]], dtype='uint8'), version=1)
        assert_raises(LabelRevisionError, self.store.publish,
                      self.ids[[2]], numpy.array([[8]], dtype='uint8'),
                      version=1)

    def test_unknown_id_rejected_entirely(self):
        bad_ids = numpy.array(
            [self.ids[1], b'id_999'], dtype='S10')
        labels = numpy.array([[1], [2]], dtype='uint8')
        assert_raises(LabelRevisionError, self.store.publish, bad_ids, labels)
        # Nothing was written.
        assert self.store.latest_version == 0
        assert self.store.versions == ()
        with h5py.File(self.store_path, mode='r') as handle:
            assert 'pending' not in handle

    def test_contradictory_same_version_rejected(self):
        bad_ids = numpy.array([self.ids[1], self.ids[1]], dtype='S10')
        labels = numpy.array([[1], [2]], dtype='uint8')
        assert_raises(LabelRevisionError, self.store.publish, bad_ids, labels)
        assert self.store.latest_version == 0
        # Repeated identical labels are allowed (deduped).
        good_ids = numpy.array([self.ids[1], self.ids[1]], dtype='S10')
        good_labels = numpy.array([[5], [5]], dtype='uint8')
        self.store.publish(good_ids, good_labels)
        assert self.store.resolve(1)[1, 0] == 5
        with h5py.File(self.store_path, mode='r') as handle:
            assert len(handle['versions/1/ids']) == 1

    def test_bad_shape_and_dtype_rejected(self):
        assert_raises(
            LabelRevisionError, self.store.publish,
            self.ids[[1, 2]], numpy.array([1, 2], dtype='uint8').reshape(2,))
        assert_raises(
            LabelRevisionError, self.store.publish,
            self.ids[[1]], numpy.array([[1.5]], dtype='float64'))
        assert self.store.latest_version == 0

    def test_noop_revision_rejected(self):
        same_labels = self.original_targets[[1, 2]]
        assert_raises(
            LabelRevisionError, self.store.publish, self.ids[[1, 2]],
            same_labels)
        assert self.store.latest_version == 0

    def test_resolve_unknown_version(self):
        assert_raises(LabelRevisionError, self.store.resolve, 1)
        self.store.publish(self.ids[[1]], numpy.array([[7]], dtype='uint8'))
        assert_raises(LabelRevisionError, self.store.resolve, 2)
        assert_raises(LabelRevisionError, self.store.resolve, -1)

    def test_resolve_version_zero_is_original(self):
        assert_equal(self.store.resolve(0), self.original_targets)

    def test_reopen_and_read_old_versions(self):
        self.store.publish(self.ids[[1]], numpy.array([[7]], dtype='uint8'))
        self.store.publish(self.ids[[1]], numpy.array([[9]], dtype='uint8'))
        reopened = LabelRevisionStore(self.store_path)
        assert reopened.latest_version == 2
        assert reopened.resolve(0)[1, 0] == self.original_targets[1, 0]
        assert reopened.resolve(1)[1, 0] == 7
        assert reopened.resolve(2)[1, 0] == 9
        assert reopened.resolve()[1, 0] == 9

    def test_full_correction_set_only_changes_stored(self):
        # A "complete" correction set where most labels are unchanged.
        full_ids = self.ids
        full_labels = self.original_targets.copy()
        full_labels[5, 0] = 77
        self.store.publish(full_ids, full_labels)
        with h5py.File(self.store_path, mode='r') as handle:
            assert len(handle['versions/1/ids']) == 1
        resolved = self.store.resolve(1)
        assert_equal(resolved, full_labels)

    def test_unicode_and_bytes_ids_equivalent(self):
        unicode_ids = numpy.array(['id_001'], dtype=object)
        self.store.publish(unicode_ids, numpy.array([[7]], dtype='uint8'))
        assert self.store.resolve(1)[1, 0] == 7

    def test_integer_ids(self):
        h5file = h5py.File('int_features.hdf5', mode='w', driver='core',
                           backing_store=False)
        h5file['ids'] = numpy.arange(10, dtype='int64')
        h5file['targets'] = numpy.arange(10, dtype='uint8').reshape(10, 1)
        try:
            store = LabelRevisionStore.create(
                'int_sidecar.hdf5', h5file, 'ids', 'targets')
            store.publish(numpy.array([3, 7], dtype='int64'),
                          numpy.array([[33], [77]], dtype='uint8'))
            resolved = store.resolve(1)
            assert resolved[3, 0] == 33
            assert resolved[7, 0] == 77
            assert resolved[0, 0] == 0
        finally:
            h5file.close()
            for path in ('int_sidecar.hdf5', 'int_sidecar.hdf5.lock'):
                if os.path.exists(path):
                    os.remove(path)

    def test_original_feature_file_untouched(self):
        self.store.publish(self.ids[[1]], numpy.array([[7]], dtype='uint8'))
        assert_equal(self.h5file['targets'][...], self.original_targets)

    def test_create_rejects_duplicate_ids(self):
        h5file = h5py.File('dup_features.hdf5', mode='w', driver='core',
                           backing_store=False)
        h5file['ids'] = numpy.array([b'a', b'a'], dtype='S2')
        h5file['targets'] = numpy.array([[0], [1]], dtype='uint8')
        try:
            assert_raises(
                LabelRevisionError, LabelRevisionStore.create,
                'dup_sidecar.hdf5', h5file, 'ids', 'targets')
            assert not os.path.exists('dup_sidecar.hdf5')
        finally:
            h5file.close()


class TestVersionedLabelsH5PYDataset(object):
    def setUp(self):
        self.h5file = make_feature_file(
            'versioned_features.hdf5', backing_store=False)
        self.original_targets = self.h5file['targets'][...]
        self.ids = self.h5file['ids'][...]
        self.store = LabelRevisionStore.create(
            'labels.sidecar.hdf5', self.h5file, 'ids', 'targets')
        # Revision 1: correct labels of examples 1 and 3.
        self.rev1_ids = self.ids[[1, 3]]
        self.rev1_labels = numpy.array([[101], [103]], dtype='uint8')
        self.store.publish(self.rev1_ids, self.rev1_labels)

    def tearDown(self):
        self.h5file.close()
        for path in ('labels.sidecar.hdf5',
                     'labels.sidecar.hdf5.lock'):
            if os.path.exists(path):
                os.remove(path)

    def _make_dataset(self, **kwargs):
        kwargs.setdefault(
            'sources', ('features', 'targets'))
        return VersionedLabelsH5PYDataset(
            self.h5file, which_sets=('train',),
            label_store=self.store, **kwargs)

    def _expected_labels_at(self, version):
        expected = self.original_targets.copy()
        if version >= 1:
            expected[[1, 3]] = self.rev1_labels
        return expected

    def test_open_pins_latest_version(self):
        dataset = self._make_dataset()
        assert dataset.label_version is None
        handle = dataset.open()
        assert handle.version == 1
        assert dataset.label_version == 1
        dataset.close(handle)

    def test_labels_aligned_with_features(self):
        dataset = self._make_dataset()
        handle = dataset.open()
        features, targets = dataset.get_data(handle, slice(0, 10))
        assert_equal(features, self.h5file['features'][0:10])
        assert_equal(targets, self._expected_labels_at(1)[0:10])
        dataset.close(handle)

    def test_list_request_stays_aligned(self):
        dataset = self._make_dataset(sort_indices=True)
        handle = dataset.open()
        request = [7, 1, 3, 2]
        features, targets = dataset.get_data(handle, request)
        assert_equal(features, self.h5file['features'][...][request])
        assert_equal(
            targets, self._expected_labels_at(1)[request])
        dataset.close(handle)

    def test_revision_published_during_epoch_not_visible_until_next(self):
        dataset = self._make_dataset()
        state = dataset.open()
        assert state.version == 1
        # A new revision appears while the epoch is running.
        rev2_ids = self.ids[[2]]
        rev2_labels = numpy.array([[202]], dtype='uint8')
        self.store.publish(rev2_ids, rev2_labels)
        # Reset keeps version 1.
        state = dataset.reset(state)
        assert state.version == 1
        assert dataset.label_version == 1
        _, targets = dataset.get_data(state, slice(0, 5))
        assert_equal(targets, self._expected_labels_at(1)[0:5])
        # The next epoch picks up version 2.
        state = dataset.next_epoch(state)
        assert state.version == 2
        assert dataset.label_version == 2
        expected = self._expected_labels_at(2)
        expected[2, 0] = 202
        _, targets = dataset.get_data(state, slice(0, 5))
        assert_equal(targets, expected[0:5])
        dataset.close(state)

    def test_reset_rereads_same_version(self):
        dataset = self._make_dataset()
        state = dataset.open()
        labels_before = state.labels.copy()
        state = dataset.reset(state)
        assert state.version == 1
        assert_equal(state.labels, labels_before)

    def test_explicit_version_pinned_across_epochs(self):
        self.store.publish(
            self.ids[[2]], numpy.array([[202]], dtype='uint8'))
        dataset = self._make_dataset(label_version=1)
        state = dataset.open()
        assert state.version == 1
        state = dataset.next_epoch(state)
        assert state.version == 1
        state = dataset.reset(state)
        assert state.version == 1

    def test_pinned_old_version_after_reopen(self):
        self.store.publish(
            self.ids[[2]], numpy.array([[202]], dtype='uint8'))
        dataset = VersionedLabelsH5PYDataset(
            self.h5file, which_sets=('train',),
            label_store='labels.sidecar.hdf5', label_version=1,
            sources=('features', 'targets'))
        handle = dataset.open()
        _, targets = dataset.get_data(handle, slice(0, 5))
        assert_equal(targets, self._expected_labels_at(1)[0:5])
        dataset.close(handle)

    def test_data_stream_epoch_semantics(self):
        dataset = self._make_dataset()
        stream = DataStream(
            dataset, iteration_scheme=SequentialScheme(20, 5))
        epoch = stream.get_epoch_iterator()
        batches = list(epoch)
        assert len(batches) == 4
        # First batch labels match revision 1.
        assert_equal(
            batches[0][1], self._expected_labels_at(1)[0:5])
        # Publish revision 2 after the first epoch completed.
        self.store.publish(
            self.ids[[2]], numpy.array([[202]], dtype='uint8'))
        # A new epoch iterator triggers next_epoch().
        epoch = stream.get_epoch_iterator()
        first_batch = next(epoch)
        assert first_batch[1][2, 0] == 202
        stream.close()

    def test_load_in_memory(self):
        dataset = self._make_dataset(load_in_memory=True)
        handle = dataset.open()
        features, targets = dataset.get_data(handle, slice(0, 8))
        assert_equal(features, self.h5file['features'][0:8])
        assert_equal(targets, self._expected_labels_at(1)[0:8])
        request = [5, 1, 3]
        features, targets = dataset.get_data(handle, request)
        assert_equal(features, self.h5file['features'][...][request])
        assert_equal(targets, self._expected_labels_at(1)[request])
        dataset.close(handle)

    def test_test_split_uses_correct_row_alignment(self):
        dataset = VersionedLabelsH5PYDataset(
            self.h5file, which_sets=('test',),
            label_store=self.store,
            sources=('features', 'targets'))
        handle = dataset.open()
        features, targets = dataset.get_data(handle, slice(0, 10))
        assert_equal(features, self.h5file['features'][20:30])
        assert_equal(targets, self._expected_labels_at(1)[20:30])
        dataset.close(handle)

    def test_invalid_label_source_raises(self):
        assert_raises(
            ValueError, VersionedLabelsH5PYDataset,
            self.h5file, which_sets=('train',),
            label_store=self.store, label_source='nonexistent')

    def test_misaligned_ids_rejected(self):
        # Change the feature file's IDs behind the store's back.
        del self.h5file['ids']
        self.h5file['ids'] = numpy.array(
            ['other_{:03d}'.format(i).encode('utf8')
             for i in range(30)], dtype='S10')
        assert_raises(
            LabelRevisionError, VersionedLabelsH5PYDataset,
            self.h5file, which_sets=('train',),
            label_store=self.store,
            sources=('features', 'targets'))

    def test_pickling(self):
        # Pickling requires a path-reopenable (on-disk) feature file.
        disk_file = make_feature_file('disk_features.hdf5',
                                      backing_store=True)
        disk_file.close()
        disk_store = LabelRevisionStore.create(
            'disk_labels.hdf5', 'disk_features.hdf5', 'ids', 'targets')
        disk_store.publish(
            self.ids[[1, 3]], self.rev1_labels)
        try:
            dataset = VersionedLabelsH5PYDataset(
                'disk_features.hdf5', which_sets=('train',),
                label_store=disk_store, label_version=1,
                sources=('features', 'targets'))
            reloaded = cPickle.loads(cPickle.dumps(dataset))
            assert reloaded.explicit_label_version == 1
            handle = reloaded.open()
            assert handle.version == 1
            _, targets = reloaded.get_data(handle, slice(0, 5))
            assert_equal(targets, self._expected_labels_at(1)[0:5])
            reloaded.close(handle)
        finally:
            for path in ('disk_features.hdf5', 'disk_labels.hdf5',
                         'disk_labels.hdf5.lock'):
                if os.path.exists(path):
                    os.remove(path)

    def test_data_stream_pickling(self):
        disk_file = make_feature_file('disk_features.hdf5',
                                      backing_store=True)
        disk_file.close()
        disk_store = LabelRevisionStore.create(
            'disk_labels.hdf5', 'disk_features.hdf5', 'ids', 'targets')
        try:
            stream = DataStream(
                VersionedLabelsH5PYDataset(
                    'disk_features.hdf5', which_sets=('train',),
                    label_store=disk_store,
                    sources=('features', 'targets')),
                iteration_scheme=SequentialScheme(20, 10))
            cPickle.loads(cPickle.dumps(stream))
            stream.close()
        finally:
            for path in ('disk_features.hdf5', 'disk_labels.hdf5',
                         'disk_labels.hdf5.lock'):
                if os.path.exists(path):
                    os.remove(path)

    def test_only_requested_sources(self):
        dataset = self._make_dataset(sources=('targets',))
        assert dataset.sources == ('targets',)
        handle = dataset.open()
        result = dataset.get_data(handle, slice(0, 4))
        assert len(result) == 1
        assert_equal(result[0], self._expected_labels_at(1)[0:4])
        dataset.close(handle)

    def test_get_data_without_explicit_open(self):
        dataset = self._make_dataset()
        try:
            _, targets = dataset.get_data(request=slice(0, 3))
            assert_equal(targets, self._expected_labels_at(1)[0:3])
        finally:
            dataset.close(None)

    def test_full_disk_persistence_and_old_version_reread(self):
        feature_path = 'persist_features.hdf5'
        sidecar_path = 'persist_labels.hdf5'
        disk_file = make_feature_file(feature_path, backing_store=True)
        disk_file.close()
        store = LabelRevisionStore.create(
            sidecar_path, feature_path, 'ids', 'targets')
        store.publish(self.ids[[1, 3]], self.rev1_labels)
        store.publish(self.ids[[2]], numpy.array([[202]], dtype='uint8'))
        try:
            # Close everything and reopen from scratch, pinning version 1.
            del store
            reopened_store = LabelRevisionStore(sidecar_path)
            dataset = VersionedLabelsH5PYDataset(
                feature_path, which_sets=('train',),
                label_store=reopened_store, label_version=1,
                sources=('features', 'targets'))
            handle = dataset.open()
            _, targets = dataset.get_data(handle, slice(0, 5))
            assert_equal(targets, self._expected_labels_at(1)[0:5])
            # Version 2 is not visible while version 1 is pinned.
            assert targets[2, 0] == self.original_targets[2, 0]
            dataset.close(handle)
            # The original labels are still present and unchanged in the
            # feature HDF5 file.
            with h5py.File(feature_path, mode='r') as feature_handle:
                assert_equal(feature_handle['targets'][...],
                             self.original_targets)
            # The latest version can still be read explicitly.
            latest = VersionedLabelsH5PYDataset(
                feature_path, which_sets=('train',),
                label_store=reopened_store,
                sources=('features', 'targets'))
            handle = latest.open()
            assert handle.version == 2
            _, targets = latest.get_data(handle, slice(0, 5))
            assert targets[2, 0] == 202
            latest.close(handle)
        finally:
            for path in (feature_path, sidecar_path,
                         sidecar_path + '.lock'):
                if os.path.exists(path):
                    os.remove(path)
