import os
import tables

import h5py
import numpy
from numpy.testing import assert_equal, assert_raises
from six.moves import range, cPickle

from fuel.datasets.hdf5 import (PytablesDataset, H5PYDataset,
                                ShardedH5PYDataset)
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme


class TestPytablesDataset(object):
    def setUp(self):
        num_rows = 500
        filters = tables.Filters(complib='blosc', complevel=5)
        h5file = tables.open_file(
            'tmp.h5', mode='w', title='Test', filters=filters)
        group = h5file.create_group("/", 'Data')
        atom = tables.UInt8Atom()
        y = h5file.create_carray(group, 'y', atom=atom, title='Data targets',
                                 shape=(num_rows, 1), filters=filters)
        for i in range(num_rows):
            y[i] = i
        h5file.flush()
        h5file.close()
        self.dataset = PytablesDataset('tmp.h5', ('y',), 20, 500)
        self.dataset_default = PytablesDataset('tmp.h5', ('y',))

    def tearDown(self):
        self.dataset.close_file()
        self.dataset_default.close_file()
        os.remove('tmp.h5')

    def test_dataset_default(self):
        assert self.dataset_default.start == 0
        assert self.dataset_default.stop == 500

    def test_get_data_slice_request(self):
        assert_equal(self.dataset.get_data(request=slice(0, 10))[0],
                     numpy.arange(20, 30).reshape(10, 1))

    def test_get_data_list_request(self):
        assert_equal(self.dataset.get_data(request=list(range(10)))[0],
                     numpy.arange(20, 30).reshape(10, 1))

    def test_get_data_value_error(self):
        assert_raises(ValueError, self.dataset.get_data, None, True)

    def test_pickling(self):
        dataset = cPickle.loads(cPickle.dumps(self.dataset))
        assert_equal(len(dataset.nodes), 1)


class TestH5PYDataset(object):
    def setUp(self):
        self.features = numpy.arange(3600, dtype='uint16').reshape((100, 36))
        self.targets = numpy.arange(30, dtype='uint8').reshape((30, 1))
        h5file = h5py.File(
            'file.hdf5', mode='w', driver='core', backing_store=False)
        h5file['features'] = self.features
        h5file['features'].dims[0].label = 'batch'
        h5file['features'].dims[1].label = 'feature'
        h5file['targets'] = self.targets
        h5file['targets'].dims[0].label = 'batch'
        h5file['targets'].dims[1].label = 'index'
        split_dict = {'train': {'features': (0, 20, None), 'targets': (0, 20)},
                      'test': {'features': (20, 30), 'targets': (20, 30)},
                      'unlabeled': {'features': (30, 100, None, '.')}}
        h5file.attrs['split'] = H5PYDataset.create_split_array(split_dict)
        self.h5file = h5file

        vlen_h5file = h5py.File(
            'test_vl.hdf5', mode='w', driver='core', backing_store=False)
        self.vlen_features = [
            numpy.arange(12, dtype='uint8').reshape((3, 2, 2)),
            numpy.arange(48, dtype='uint8').reshape((3, 4, 4)),
            numpy.arange(60, dtype='uint8').reshape((3, 5, 4)),
            numpy.arange(18, dtype='uint8').reshape((3, 2, 3))]
        self.vlen_targets = numpy.arange(4, dtype='uint8').reshape((4, 1))
        dtype = h5py.special_dtype(vlen=numpy.dtype('uint8'))
        features = vlen_h5file.create_dataset('features', (4,), dtype=dtype)
        features[...] = [d.flatten() for d in self.vlen_features]
        features.dims[0].label = 'batch'
        features_shapes = vlen_h5file.create_dataset(
            'features_shapes', (4, 3), dtype='uint8')
        features_shapes[...] = numpy.array(
            [d.shape for d in self.vlen_features])
        features.dims.create_scale(features_shapes, 'shapes')
        features.dims[0].attach_scale(features_shapes)
        features_shape_labels = vlen_h5file.create_dataset(
            'features_shape_labels', (3,), dtype='S7')
        features_shape_labels[...] = [
            'channel'.encode('utf8'), 'height'.encode('utf8'),
            'width'.encode('utf8')]
        features.dims.create_scale(features_shape_labels, 'shape_labels')
        features.dims[0].attach_scale(features_shape_labels)
        targets = vlen_h5file.create_dataset('targets', (4, 1), dtype='uint8')
        targets[...] = self.vlen_targets
        targets.dims[0].label = 'batch'
        targets.dims[1].label = 'index'
        split_dict = {'train': {'features': (0, 4), 'targets': (0, 4)}}
        vlen_h5file.attrs['split'] = H5PYDataset.create_split_array(split_dict)
        self.vlen_h5file = vlen_h5file

    def tearDown(self):
        self.h5file.close()
        self.vlen_h5file.close()

    def test_raises_value_error_when_which_sets_is_string(self):
        assert_raises(ValueError, H5PYDataset, self.h5file, 'train')

    def test_split_parsing(self):
        train_set = H5PYDataset(self.h5file, which_sets=('train',))
        assert train_set.provides_sources == ('features', 'targets')
        test_set = H5PYDataset(self.h5file, which_sets=('test',))
        assert test_set.provides_sources == ('features', 'targets')
        unlabeled_set = H5PYDataset(self.h5file, which_sets=('unlabeled',))
        assert unlabeled_set.provides_sources == ('features',)

    def test_get_all_splits(self):
        splits = ('train', 'test', 'unlabeled')
        available_splits = H5PYDataset.get_all_splits(self.h5file)
        assert (all(split in available_splits for split in splits) and
                all(split in splits for split in available_splits))

    def test_get_all_sources(self):
        sources = ('features', 'targets')
        all_sources = H5PYDataset.get_all_sources(self.h5file)
        assert (all(source in all_sources for source in sources) and
                all(source in sources for source in all_sources))

    def test_axis_labels(self):
        dataset = H5PYDataset(self.h5file, which_sets=('train',))
        assert dataset.axis_labels == {'features': ('batch', 'feature'),
                                       'targets': ('batch', 'index')}

    def test_pickling(self):
        try:
            features = numpy.arange(360, dtype='uint16').reshape((10, 36))
            h5file = h5py.File('file.hdf5', mode='w')
            h5file['features'] = features
            split_dict = {'train': {'features': (0, 10, None, '.')}}
            h5file.attrs['split'] = H5PYDataset.create_split_array(split_dict)
            dataset = cPickle.loads(
                cPickle.dumps(H5PYDataset(h5file, which_sets=('train',))))
            # Make sure _out_of_memory_{open,close} accesses
            # external_file_handle rather than _external_file_handle
            dataset._out_of_memory_open()
            dataset._out_of_memory_close()
            assert dataset.data_sources is None
        finally:
            os.remove('file.hdf5')

    def test_data_stream_pickling(self):
        stream = DataStream(H5PYDataset(self.h5file, which_sets=('train',)),
                            iteration_scheme=SequentialScheme(100, 10))
        cPickle.loads(cPickle.dumps(stream))
        stream.close()

    def test_multiple_instances(self):
        dataset_1 = H5PYDataset(self.h5file, which_sets=('train',))
        dataset_2 = H5PYDataset(self.h5file, which_sets=('train',))
        handle_1 = dataset_1.open()
        handle_2 = dataset_2.open()
        dataset_1.get_data(state=handle_1, request=slice(0, 10))
        dataset_2.get_data(state=handle_2, request=slice(0, 10))
        dataset_1.close(handle_1)
        dataset_2.close(handle_2)

    def test_single_split(self):
        train_set = H5PYDataset(self.h5file, which_sets=('train',))
        test_set = H5PYDataset(self.h5file, which_sets=('test',))
        train_handle = train_set.open()
        test_handle = test_set.open()
        assert_equal(train_set.get_data(train_handle, slice(0, 8)),
                     (self.features[:8], self.targets[:8]))
        assert_equal(test_set.get_data(test_handle, slice(0, 2)),
                     (self.features[20:22], self.targets[20:22]))
        train_set.close(train_handle)
        test_set.close(test_handle)

    def test_multiple_split_in_memory(self):
        dataset = H5PYDataset(self.h5file, which_sets=('train', 'test'),
                              load_in_memory=True)
        handle = dataset.open()
        assert_equal(dataset.get_data(handle, slice(0, 30)),
                     (self.features[:30], self.targets[:30]))
        dataset.close(handle)

    def test_multiple_split_out_of_memory_list_request(self):
        dataset = H5PYDataset(self.h5file, which_sets=('train', 'test'),
                              load_in_memory=False)
        handle = dataset.open()
        assert_equal(dataset.get_data(handle, list(range(30))),
                     (self.features[:30], self.targets[:30]))
        dataset.close(handle)

    def test_out_of_memory(self):
        dataset = H5PYDataset(
            self.h5file, which_sets=('test',), load_in_memory=False)
        handle = dataset.open()
        assert_equal(dataset.get_data(handle, slice(3, 5)),
                     (self.features[23:25], self.targets[23:25]))
        dataset.close(handle)

    def test_in_memory(self):
        dataset = H5PYDataset(
            self.h5file, which_sets=('train',), load_in_memory=True)
        handle = dataset.open()
        request = slice(0, 10)
        assert_equal(dataset.get_data(handle, request),
                     (self.features[request], self.targets[request]))
        dataset.close(handle)

    def test_out_of_memory_sorted_indices(self):
        dataset = H5PYDataset(
            self.h5file, which_sets=('train',), load_in_memory=False,
            sort_indices=True)
        handle = dataset.open()
        request = [7, 4, 6, 2, 5]
        assert_equal(dataset.get_data(handle, request),
                     (self.features[request], self.targets[request]))
        dataset.close(handle)

    def test_out_of_memory_unsorted_indices(self):
        dataset = H5PYDataset(
            self.h5file, which_sets=('train',), load_in_memory=False,
            sort_indices=False)
        handle = dataset.open()
        assert_raises(TypeError, dataset.get_data, handle, [7, 4, 6, 2, 5])
        dataset.close(handle)

    def test_in_memory_example_scheme(self):
        dataset = H5PYDataset(
            self.h5file, which_sets=('train',), load_in_memory=True)
        iter_ = dataset.get_example_stream().get_epoch_iterator()
        assert_equal(next(iter_), (self.features[0], self.targets[0]))
        assert_equal(next(iter_), (self.features[1], self.targets[1]))

    def test_out_of_memory_example_scheme(self):
        dataset = H5PYDataset(
            self.h5file, which_sets=('train',), load_in_memory=False)
        iter_ = dataset.get_example_stream().get_epoch_iterator()
        assert_equal(next(iter_), (self.features[0], self.targets[0]))
        assert_equal(next(iter_), (self.features[1], self.targets[1]))

    def test_value_error_on_unequal_sources(self):
        def get_subsets():
            return H5PYDataset(self.h5file, which_sets=('train',)).subsets
        split_dict = {'train': {'features': (0, 20), 'targets': (0, 15)},
                      'test': {'features': (20, 30), 'targets': (20, 30)},
                      'unlabeled': {'features': (30, 100, None, '.')}}
        self.h5file.attrs['split'] = H5PYDataset.create_split_array(split_dict)
        assert_raises(ValueError, get_subsets)

    def test_io_error_on_unopened_file_handle(self):
        def get_file_handle():
            dataset = H5PYDataset(self.h5file, which_sets=('train',))
            dataset._external_file_handle = None
            return dataset._file_handle
        assert_raises(IOError, get_file_handle)

    def test_value_error_in_memory_get_data(self):
        dataset = H5PYDataset(self.h5file, which_sets=('train',))
        assert_raises(ValueError, dataset._in_memory_get_data, None, None)
        assert_raises(ValueError, dataset._in_memory_get_data, True, None)

    def test_value_error_out_of_memory_get_data(self):
        dataset = H5PYDataset(self.h5file, which_sets=('train',))
        assert_raises(ValueError, dataset._out_of_memory_get_data, None,
                      dict())

    def test_index_split_out_of_memory(self):
        features = numpy.arange(50, dtype='uint8').reshape((10, 5))
        h5file = h5py.File(
            'index_split.hdf5', mode='w', driver='core', backing_store=False)
        h5file['features'] = features
        h5file['features'].dims[0].label = 'batch'
        h5file['features'].dims[1].label = 'feature'
        h5file['train_features_subset'] = numpy.arange(0, 10, 2)
        h5file['test_features_subset'] = numpy.arange(1, 10, 2)
        train_ref = h5file['train_features_subset'].ref
        test_ref = h5file['test_features_subset'].ref
        split_dict = {'train': {'features': (-1, -1, train_ref, '.')},
                      'test': {'features': (-1, -1, test_ref, '')}}
        h5file.attrs['split'] = H5PYDataset.create_split_array(split_dict)
        dataset = H5PYDataset(
            h5file, which_sets=('train',), load_in_memory=False)
        handle = dataset.open()
        request = slice(0, 5)
        assert_equal(
            dataset.get_data(handle, request)[0], features[0:10:2])
        assert_equal(dataset.num_examples, 5)
        dataset.close(handle)

    def test_index_split_in_memory(self):
        features = numpy.arange(50, dtype='uint8').reshape((10, 5))
        h5file = h5py.File(
            'index_split.hdf5', mode='w', driver='core', backing_store=False)
        h5file['features'] = features
        h5file['features'].dims[0].label = 'batch'
        h5file['features'].dims[1].label = 'feature'
        h5file['train_features_subset'] = numpy.arange(0, 10, 2)
        h5file['test_features_subset'] = numpy.arange(1, 10, 2)
        train_ref = h5file['train_features_subset'].ref
        test_ref = h5file['test_features_subset'].ref
        split_dict = {'train': {'features': (-1, -1, train_ref, '.')},
                      'test': {'features': (-1, -1, test_ref, '')}}
        h5file.attrs['split'] = H5PYDataset.create_split_array(split_dict)
        dataset = H5PYDataset(
            h5file, which_sets=('train',), load_in_memory=True)
        handle = dataset.open()
        request = slice(0, 5)
        assert_equal(
            dataset.get_data(handle, request)[0], features[0:10:2])
        assert_equal(dataset.num_examples, 5)
        dataset.close(handle)

    def test_index_subset_sorted(self):
        dataset = H5PYDataset(
            self.h5file, which_sets=('train',), subset=[0, 2, 4])
        handle = dataset.open()
        request = slice(0, 3)
        assert_equal(dataset.get_data(handle, request),
                     (self.features[[0, 2, 4]], self.targets[[0, 2, 4]]))
        dataset.close(handle)

    def test_index_subset_unsorted(self):
        # A subset should have the same ordering no matter how you specify it.
        dataset = H5PYDataset(
            self.h5file, which_sets=('train',), subset=[0, 4, 2])
        handle = dataset.open()
        request = slice(0, 3)
        assert_equal(dataset.get_data(handle, request),
                     (self.features[[0, 2, 4]], self.targets[[0, 2, 4]]))
        dataset.close(handle)

    def test_vlen_axis_labels(self):
        dataset = H5PYDataset(self.vlen_h5file, which_sets=('train',))
        assert_equal(dataset.axis_labels['features'],
                     ('batch', 'channel', 'height', 'width'))
        assert_equal(dataset.axis_labels['targets'], ('batch', 'index'))

    def test_vlen_sources_raises_error_on_dim_gt_1(self):
        targets = self.vlen_h5file['targets']
        targets_shapes = self.vlen_h5file.create_dataset(
            'targets_shapes', (4, 1), dtype='uint8')
        targets.dims.create_scale(targets_shapes, 'shapes')
        targets.dims[0].attach_scale(targets_shapes)
        assert_raises(ValueError, H5PYDataset, self.vlen_h5file, 'train')

    def test_vlen_reshape_in_memory(self):
        dataset = H5PYDataset(
            self.vlen_h5file, which_sets=('train',), subset=slice(1, 3),
            load_in_memory=True)
        expected_features = numpy.empty((2,), dtype=numpy.object)
        for i, f in enumerate(self.vlen_features[1:3]):
            expected_features[i] = f
        expected_targets = self.vlen_targets[1:3]
        handle = dataset.open()
        rval = dataset.get_data(handle, slice(0, 2))
        for val, truth in zip(rval[0], expected_features):
            assert_equal(val, truth)
        assert_equal(rval[1], expected_targets)
        dataset.close(handle)

    def test_vlen_reshape_out_of_memory(self):
        dataset = H5PYDataset(
            self.vlen_h5file, which_sets=('train',), subset=slice(1, 3),
            load_in_memory=False)
        expected_features = numpy.empty((2,), dtype=numpy.object)
        for i, f in enumerate(self.vlen_features[1:3]):
            expected_features[i] = f
        expected_targets = self.vlen_targets[1:3]
        handle = dataset.open()
        rval = dataset.get_data(handle, slice(0, 2))
        for val, truth in zip(rval[0], expected_features):
            assert_equal(val, truth)
        assert_equal(rval[1], expected_targets)
        dataset.close(handle)

    def test_vlen_reshape_out_of_memory_unordered(self):
        dataset = H5PYDataset(
            self.vlen_h5file, which_sets=('train',), load_in_memory=False)
        expected_features = numpy.empty((4,), dtype=numpy.object)
        for i, j in enumerate([0, 3, 1, 2]):
            expected_features[i] = self.vlen_features[j]
        expected_targets = self.vlen_targets[[0, 3, 1, 2]]
        handle = dataset.open()
        rval = dataset.get_data(handle, [0, 3, 1, 2])
        for val, truth in zip(rval[0], expected_features):
            assert_equal(val, truth)
        assert_equal(rval[1], expected_targets)
        dataset.close(handle)

    def test_vlen_reshape_out_of_memory_unordered_no_check(self):
        dataset = H5PYDataset(
            self.vlen_h5file, which_sets=('train',), load_in_memory=False,
            sort_indices=False)
        expected_features = numpy.empty((4,), dtype=numpy.object)
        for i, j in enumerate([0, 1, 2, 3]):
            expected_features[i] = self.vlen_features[j]
        expected_targets = self.vlen_targets[[0, 1, 2, 3]]
        handle = dataset.open()
        rval = dataset.get_data(handle, [0, 1, 2, 3])
        for val, truth in zip(rval[0], expected_features):
            assert_equal(val, truth)
        assert_equal(rval[1], expected_targets)
        dataset.close(handle)

    def test_vlen_in_memory_example_scheme(self):
        dataset = H5PYDataset(
            self.vlen_h5file, which_sets=('train',), load_in_memory=True,
            sort_indices=False)
        iter_ = dataset.get_example_stream().get_epoch_iterator()
        assert_equal(next(iter_),
                     (self.vlen_features[0], self.vlen_targets[0]))
        assert_equal(next(iter_),
                     (self.vlen_features[1], self.vlen_targets[1]))

    def test_vlen_out_of_memory_example_scheme(self):
        dataset = H5PYDataset(
            self.vlen_h5file, which_sets=('train',), load_in_memory=False,
            sort_indices=False)
        iter_ = dataset.get_example_stream().get_epoch_iterator()
        assert_equal(next(iter_),
                     (self.vlen_features[0], self.vlen_targets[0]))
        assert_equal(next(iter_),
                     (self.vlen_features[1], self.vlen_targets[1]))

    def test_dataset_get_data_without_open(self):
        dataset = H5PYDataset(self.h5file, which_sets=('train',),
                              load_in_memory=False)
        try:
            dataset.get_data(request=(slice(0, 2)))
        except IOError:
            assert False
        dataset.close(None)


class TestShardedH5PYDataset(object):
    def setUp(self):
        self.directory = 'sharded_hdf5'
        if not os.path.isdir(self.directory):
            os.mkdir(self.directory)
        # Shard 0 contributes 12 examples, shard 1 contributes 20 and shard 2
        # contributes none, so that empty shards are exercised too.
        self.features = {}
        self.targets = {}
        self.paths = []
        for index, num_rows in enumerate((20, 30, 5)):
            path = os.path.join(self.directory, 'shard_{}.hdf5'.format(index))
            features = numpy.arange(num_rows * 4, dtype='uint16').reshape(
                num_rows, 4) + 1000 * index
            targets = ((numpy.arange(num_rows) + 100 * index) % 256).astype(
                'uint8').reshape(num_rows, 1)
            h5file = h5py.File(path, mode='w')
            h5file['features'] = features
            h5file['features'].dims[0].label = 'batch'
            h5file['features'].dims[1].label = 'feature'
            h5file['targets'] = targets
            h5file['targets'].dims[0].label = 'batch'
            h5file['targets'].dims[1].label = 'index'
            splits = {
                'train': {'features': (0, 12), 'targets': (0, 12)},
                'test': {'features': (12, num_rows),
                         'targets': (12, num_rows)}}
            if index == 2:
                # A valid but empty split, to exercise skipped shards.
                splits = {'train': {'features': (0, 0), 'targets': (0, 0)},
                          'test': {'features': (0, num_rows),
                                   'targets': (0, num_rows)}}
            h5file.attrs['split'] = H5PYDataset.create_split_array(splits)
            h5file.close()
            self.features[index] = features
            self.targets[index] = targets
            self.paths.append(path)

    def tearDown(self):
        for path in self.paths:
            os.remove(path)
        os.rmdir(self.directory)

    def _make_dataset(self, **kwargs):
        kwargs.setdefault('shards', [
            (self.paths[0], 'train'), (self.paths[1], 'train')])
        return ShardedH5PYDataset(**kwargs)

    def _expected(self):
        features = numpy.concatenate(
            [self.features[0][:12], self.features[1][:12]])
        targets = numpy.concatenate(
            [self.targets[0][:12], self.targets[1][:12]])
        return features, targets

    def test_num_examples_and_shard_layout(self):
        dataset = self._make_dataset()
        assert dataset.num_examples == 24
        assert dataset.shard_num_examples == (12, 12)
        assert dataset.shard_files == (self.paths[0], self.paths[1])
        assert dataset.shard_splits == (('train',), ('train',))

    def test_integer_request(self):
        dataset = self._make_dataset()
        features, targets = self._expected()
        handle = dataset.open()
        for index in (0, 11, 12, 23):
            data = dataset.get_data(handle, index)
            assert_equal(data[0], features[index])
            assert_equal(data[1], targets[index])
        dataset.close(handle)

    def test_slice_request_spanning_shards(self):
        dataset = self._make_dataset()
        features, targets = self._expected()
        handle = dataset.open()
        request = slice(10, 15)
        data = dataset.get_data(handle, request)
        assert_equal(data[0], features[request])
        assert_equal(data[1], targets[request])
        dataset.close(handle)

    def test_slice_request_with_step(self):
        dataset = self._make_dataset()
        features, targets = self._expected()
        handle = dataset.open()
        for request in (slice(0, 24, 3), slice(1, 23, 5), slice(11, 13)):
            data = dataset.get_data(handle, request)
            assert_equal(data[0], features[request])
            assert_equal(data[1], targets[request])
        dataset.close(handle)

    def test_unsorted_and_duplicated_list_request(self):
        dataset = self._make_dataset()
        features, targets = self._expected()
        handle = dataset.open()
        request = [23, 0, 12, 0, 11, 23, 5]
        data = dataset.get_data(handle, request)
        assert_equal(data[0], features[request])
        assert_equal(data[1], targets[request])
        dataset.close(handle)

    def test_different_split_per_shard(self):
        dataset = ShardedH5PYDataset(
            [(self.paths[0], 'train'), (self.paths[1], 'test')])
        features = numpy.concatenate(
            [self.features[0][:12], self.features[1][12:]])
        targets = numpy.concatenate(
            [self.targets[0][:12], self.targets[1][12:]])
        assert dataset.num_examples == 30
        assert dataset.shard_num_examples == (12, 18)
        handle = dataset.open()
        request = [29, 12, 0, 12]
        data = dataset.get_data(handle, request)
        assert_equal(data[0], features[request])
        assert_equal(data[1], targets[request])
        dataset.close(handle)

    def test_default_which_sets(self):
        dataset = ShardedH5PYDataset(self.paths[:2], which_sets='train')
        assert dataset.num_examples == 24

    def test_dict_of_shards(self):
        dataset = ShardedH5PYDataset(
            {self.paths[0]: 'train', self.paths[1]: 'test'})
        assert dataset.num_examples == 30
        assert dataset.shard_splits == (('train',), ('test',))

    def test_empty_shard_is_skipped(self):
        # Shard 2's train split is (0, 12) on a 5-row dataset, i.e. empty.
        dataset = ShardedH5PYDataset(
            [(self.paths[0], 'train'), (self.paths[2], 'train'),
             (self.paths[1], 'train')])
        assert dataset.shard_num_examples == (12, 0, 12)
        assert dataset.num_examples == 24
        features, targets = self._expected()
        handle = dataset.open()
        request = [0, 12, 23]
        data = dataset.get_data(handle, request)
        assert_equal(data[0], features[request])
        assert_equal(data[1], targets[request])
        dataset.close(handle)

    def test_subset(self):
        dataset = self._make_dataset(subset=[20, 2, 13])
        features, targets = self._expected()
        assert dataset.num_examples == 3
        handle = dataset.open()
        # List-based subsets are sorted, like `H5PYDataset`.
        data = dataset.get_data(handle, slice(0, 3))
        assert_equal(data[0], features[[2, 13, 20]])
        assert_equal(data[1], targets[[2, 13, 20]])
        dataset.close(handle)

    def test_sources_argument(self):
        dataset = self._make_dataset(sources=('targets', 'features'))
        assert dataset.sources == ('targets', 'features')
        handle = dataset.open()
        data = dataset.get_data(handle, 0)
        assert data[0].shape == (1,)
        assert data[1].shape == (4,)
        dataset.close(handle)

    def test_rejects_incompatible_dtypes(self):
        path = os.path.join(self.directory, 'bad_dtype.hdf5')
        h5file = h5py.File(path, mode='w')
        h5file['features'] = numpy.zeros((4, 4), dtype='float32')
        h5file['features'].dims[0].label = 'batch'
        h5file['features'].dims[1].label = 'feature'
        h5file['targets'] = numpy.zeros((4, 1), dtype='uint8')
        h5file['targets'].dims[0].label = 'batch'
        h5file['targets'].dims[1].label = 'index'
        h5file.attrs['split'] = H5PYDataset.create_split_array(
            {'train': {'features': (0, 4), 'targets': (0, 4)}})
        h5file.close()
        try:
            assert_raises(
                ValueError, ShardedH5PYDataset,
                [(self.paths[0], 'train'), (path, 'train')])
        finally:
            os.remove(path)

    def test_rejects_incompatible_sources(self):
        path = os.path.join(self.directory, 'bad_sources.hdf5')
        h5file = h5py.File(path, mode='w')
        h5file['features'] = numpy.zeros((4, 4), dtype='uint16')
        h5file['features'].dims[0].label = 'batch'
        h5file['features'].dims[1].label = 'feature'
        h5file['labels'] = numpy.zeros((4, 1), dtype='uint8')
        h5file['labels'].dims[0].label = 'batch'
        h5file['labels'].dims[1].label = 'index'
        h5file.attrs['split'] = H5PYDataset.create_split_array(
            {'train': {'features': (0, 4), 'labels': (0, 4)}})
        h5file.close()
        try:
            assert_raises(
                ValueError, ShardedH5PYDataset,
                [(self.paths[0], 'train'), (path, 'train')])
        finally:
            os.remove(path)

    def test_rejects_incompatible_axis_labels(self):
        path = os.path.join(self.directory, 'bad_labels.hdf5')
        h5file = h5py.File(path, mode='w')
        h5file['features'] = numpy.zeros((4, 4), dtype='uint16')
        h5file['features'].dims[0].label = 'batch'
        h5file['features'].dims[1].label = 'pixel'
        h5file['targets'] = numpy.zeros((4, 1), dtype='uint8')
        h5file['targets'].dims[0].label = 'batch'
        h5file['targets'].dims[1].label = 'index'
        h5file.attrs['split'] = H5PYDataset.create_split_array(
            {'train': {'features': (0, 4), 'targets': (0, 4)}})
        h5file.close()
        try:
            assert_raises(
                ValueError, ShardedH5PYDataset,
                [(self.paths[0], 'train'), (path, 'train')])
        finally:
            os.remove(path)

    def test_raises_when_all_shards_are_empty(self):
        assert_raises(
            ValueError, ShardedH5PYDataset,
            [(self.paths[2], 'train')])

    def test_raises_on_missing_split(self):
        assert_raises(
            ValueError, ShardedH5PYDataset, [(self.paths[0], 'nope')])

    def test_raises_on_out_of_range_index(self):
        dataset = self._make_dataset()
        handle = dataset.open()
        assert_raises(ValueError, dataset.get_data, handle, 24)
        assert_raises(ValueError, dataset.get_data, handle, -1)
        dataset.close(handle)

    def test_out_of_memory_only_reads_requested_shard(self):
        dataset = self._make_dataset(max_open_files=1)
        handle = dataset.open()
        dataset.get_data(handle, 0)
        assert len(dataset._handle_pool._handles) == 1
        # The handle is not held while no request is in flight.
        assert dataset._handle_pool._ref_counts == {}
        dataset.close(handle)

    def test_max_open_files_bound(self):
        paths = [self.paths[0], self.paths[1]]
        dataset = ShardedH5PYDataset(
            [(path, 'train') for path in paths], max_open_files=1)
        handle = dataset.open()
        # Reading indices from both shards in a single request still works
        # and the pool never holds more than one file.
        dataset.get_data(handle, [0, 20])
        assert len(dataset._handle_pool._handles) <= 1
        dataset.close(handle)

    def test_closing_one_stream_does_not_break_another(self):
        dataset = self._make_dataset()
        features, targets = self._expected()
        stream_1 = DataStream(dataset, iteration_scheme=SequentialScheme(
            dataset.num_examples, 4))
        stream_2 = DataStream(dataset, iteration_scheme=SequentialScheme(
            dataset.num_examples, 4))
        iterator_1 = stream_1.get_epoch_iterator()
        iterator_2 = stream_2.get_epoch_iterator()
        next(iterator_1)
        next(iterator_2)
        stream_1.close()
        data = next(iterator_2)
        assert_equal(data[0], features[4:8])
        assert_equal(data[1], targets[4:8])
        stream_2.close()

    def test_pickling(self):
        dataset = self._make_dataset()
        unpickled = cPickle.loads(cPickle.dumps(dataset))
        assert unpickled.num_examples == dataset.num_examples
        features, targets = self._expected()
        handle = unpickled.open()
        request = [1, 13, 1]
        data = unpickled.get_data(handle, request)
        assert_equal(data[0], features[request])
        assert_equal(data[1], targets[request])
        unpickled.close(handle)

    def test_pickling_stream(self):
        dataset = self._make_dataset()
        stream = DataStream(dataset, iteration_scheme=SequentialScheme(
            dataset.num_examples, 4))
        unpickled = cPickle.loads(cPickle.dumps(stream))
        iterator = unpickled.get_epoch_iterator()
        next(iterator)
        stream.close()
        unpickled.close()

    def test_load_in_memory(self):
        dataset = self._make_dataset(load_in_memory=True)
        features, targets = self._expected()
        assert dataset.data_sources[0].shape == (24, 4)
        handle = dataset.open()
        request = [22, 1, 22, 0]
        data = dataset.get_data(handle, request)
        assert_equal(data[0], features[request])
        assert_equal(data[1], targets[request])
        dataset.close(handle)

    def test_load_in_memory_with_subset(self):
        dataset = self._make_dataset(
            load_in_memory=True, subset=slice(10, 15))
        features, targets = self._expected()
        handle = dataset.open()
        data = dataset.get_data(handle, [4, 0, 4])
        assert_equal(data[0], features[[14, 10, 14]])
        assert_equal(data[1], targets[[14, 10, 14]])
        dataset.close(handle)

    def test_example_stream(self):
        dataset = self._make_dataset()
        features, targets = self._expected()
        iterator = dataset.get_example_stream().get_epoch_iterator()
        for index in range(3):
            data = next(iterator)
            assert_equal(data[0], features[index])
            assert_equal(data[1], targets[index])

    def test_variable_length_shapes_match_example(self):
        images = {
            0: [numpy.arange(12, dtype='uint8').reshape(3, 2, 2),
                numpy.arange(24, dtype='uint8').reshape(3, 4, 2)],
            1: [numpy.arange(30, dtype='uint8').reshape(3, 5, 2),
                numpy.arange(18, dtype='uint8').reshape(3, 3, 2),
                numpy.arange(6, dtype='uint8').reshape(3, 1, 2)]}
        paths = []
        for index, shard_images in images.items():
            path = os.path.join(
                self.directory, 'vlen_{}.hdf5'.format(index))
            h5file = h5py.File(path, mode='w')
            dtype = h5py.special_dtype(vlen=numpy.dtype('uint8'))
            features = h5file.create_dataset(
                'features', (len(shard_images),), dtype=dtype)
            features[...] = [image.flatten() for image in shard_images]
            features.dims[0].label = 'batch'
            shapes = h5file.create_dataset(
                'features_shapes', (len(shard_images), 3), dtype='int32')
            shapes[...] = numpy.array(
                [image.shape for image in shard_images])
            features.dims.create_scale(shapes, 'shapes')
            features.dims[0].attach_scale(shapes)
            labels = h5file.create_dataset(
                'features_shape_labels', (3,), dtype='S7')
            labels[...] = [b'channel', b'height', b'width']
            features.dims.create_scale(labels, 'shape_labels')
            features.dims[0].attach_scale(labels)
            targets = numpy.arange(
                len(shard_images), dtype='uint8').reshape(
                    len(shard_images), 1) + 10 * index
            h5file['targets'] = targets
            h5file['targets'].dims[0].label = 'batch'
            h5file['targets'].dims[1].label = 'index'
            h5file.attrs['split'] = H5PYDataset.create_split_array(
                {'train': {'features': (0, len(shard_images)),
                           'targets': (0, len(shard_images))}})
            h5file.close()
            paths.append(path)
        try:
            all_images = images[0] + images[1]
            all_targets = numpy.concatenate([
                numpy.arange(2, dtype='uint8') + 0,
                numpy.arange(3, dtype='uint8') + 10]).reshape(5, 1)
            dataset = ShardedH5PYDataset(
                [(paths[0], 'train'), (paths[1], 'train')],
                load_in_memory=False)
            handle = dataset.open()
            request = [4, 1, 4, 0, 2]
            data = dataset.get_data(handle, request)
            for value, index in zip(data[0], request):
                assert_equal(value, all_images[index])
            assert_equal(data[1], all_targets[request])
            dataset.close(handle)
        finally:
            for path in paths:
                os.remove(path)
