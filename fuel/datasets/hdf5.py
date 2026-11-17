import numbers
import threading
from bisect import bisect_right
from itertools import product
from collections import OrderedDict, defaultdict

import h5py
import numpy
import six
import tables
from six.moves import zip, range

from fuel.datasets import Dataset
from fuel.schemes import SequentialExampleScheme
from fuel.utils import do_not_pickle_attributes, Subset
from fuel.utils.cache import cache_file


@do_not_pickle_attributes('nodes', 'h5file')
class PytablesDataset(Dataset):
    """A pytables dataset.

    An HDF5 Dataset which was created with pytables. The dataset should
    have the following structure: `/<data_node>/paths/to/sources`. In
    order to have train/validation/test split you may want to open
    several datasets with different data nodes or source paths. It is
    also possible to use start and stop arguments to split your dataset.

    Parameters
    ----------
    sources : tuple of strings
        Sources which the dataset returns.
    start : int
        Start index. Optional, by default is 0.
    stop : int
        Stop index. Optional, if is not provided, will be set to the
        number of rows of the first source.
    data_node : str
        Parent data node in HDF5 file, all path are relative to this node.
    sources_in_file : tuple of strings
        Names of nodes in HDF5 file which contain sources. Should the same
        length as `sources`.
        Optional, if not set will be equal to `sources`.

    """
    def __init__(self, path, sources, start=0, stop=None, data_node='Data',
                 sources_in_file=None):
        if sources_in_file is None:
            sources_in_file = sources
        self.sources_in_file = sources_in_file
        self.provides_sources = sources
        self.path = path
        self.data_node = data_node
        self.start = start
        self.stop = stop
        self.nodes = None
        self.open_file(path)
        super(PytablesDataset, self).__init__(self.provides_sources)

    def open_file(self, path):
        self.h5file = tables.open_file(path, mode="r")
        node = self.h5file.get_node('/', self.data_node)

        self.nodes = [getattr(node, source) for source in self.sources_in_file]
        if self.stop is None:
            self.stop = self.nodes[0].nrows
        self.num_examples = self.stop - self.start

    def load(self):
        self.open_file(self.path)

    def close_file(self):
        self.h5file.close()
        del self._h5file
        del self._nodes

    def get_data(self, state=None, request=None):
        """ Returns data from HDF5 dataset.

        .. note:: The best performance if `request` is a slice.

        """
        if isinstance(request, slice):
            request = slice(request.start + self.start,
                            request.stop + self.start, request.step)
            data = [node[request] for node in self.nodes]
        elif isinstance(request, list):
            request = [index + self.start for index in request]
            data = [node[request, ...] for node in self.nodes]
        else:
            raise ValueError
        return data


@do_not_pickle_attributes('data_sources', 'external_file_handle',
                          'source_shapes', 'in_memory_subset', 'subsets')
class H5PYDataset(Dataset):
    """An h5py-fueled HDF5 dataset.

    This dataset class assumes a particular file layout:

    * Data sources reside in the root group, and their names define the
      source names.
    * Data sources are not explicitly split. Instead, splits are defined
      in the `split` attribute of the root group. It's expected to be a
      1D numpy array of compound ``dtype`` with seven fields, organized as
      follows:

      1. ``split`` : string identifier for the split name
      2. ``source`` : string identifier for the source name
      3. ``start`` : start index (inclusive) of the split in the source
         array, used if ``indices`` is a null reference.
      4. ``stop`` : stop index (exclusive) of the split in the source
         array, used if ``indices`` is a null reference.
      5. ``indices`` : h5py.Reference, reference to a dataset containing
         subset indices for this split/source pair. If it's a null
         reference, ``start`` and ``stop`` are used.
      6. ``available`` : boolean, ``False`` is this split is not available
         for this source
      7. ``comment`` : comment string

    Parameters
    ----------
    file_or_path : :class:`h5py.File` or str
        HDF5 file handle, or path to the HDF5 file.
    which_sets : iterable of str
        Which split(s) to use. If one than more split is requested,
        the provided sources will be the intersection of provided
        sources for these splits. **Note: for all splits that are
        specified as a list of indices, those indices will get sorted
        no matter what.**
    subset : {slice, list of int}, optional
        Which subset of data to use *within the context of the split*.
        Can be either a slice or a list of indices. Defaults to `None`,
        in which case the whole split is used.
    load_in_memory : bool, optional
        Whether to load the data in main memory. Defaults to `False`.
    driver : str, optional
        Low-level driver to use. Defaults to `None`. See h5py
        documentation for a complete list of available options.
    sort_indices : bool, optional
        HDF5 doesn't support fancy indexing with an unsorted list of
        indices. In order to allow that, the dataset can sort the list
        of indices, access the data in sorted order and shuffle back
        the data in the unsorted order. Setting this flag to `True`
        (the default) will activate this behaviour. For greater
        performance, set this flag to `False`. Note that in that case,
        it is the user's responsibility to make sure that indices are
        ordered.

    Attributes
    ----------
    sources : tuple of strings
        The sources this dataset will provide when queried for data.
    provides_sources : tuple of strings
        The sources this dataset *is able to* provide for the requested
        split.
    example_iteration_scheme : :class:`.IterationScheme` or ``None``
        The iteration scheme the class uses in order to produce a stream of
        examples.
    vlen_sources : tuple of strings
        All sources provided by this dataset which have variable length.
    default_axis_labels : dict mapping string to tuple of strings
        Maps all sources provided by this dataset to their axis labels.

    """
    interface_version = '0.3'
    _ref_counts = defaultdict(int)
    _file_handles = {}

    def __init__(self, file_or_path, which_sets, subset=None,
                 load_in_memory=False, driver=None, sort_indices=True,
                 **kwargs):
        if isinstance(file_or_path, h5py.File):
            self.path = file_or_path.filename
            self.external_file_handle = file_or_path
        else:
            self.path = file_or_path
            self.path = cache_file(self.path)
            self.external_file_handle = None
        which_sets_invalid_value = (
            isinstance(which_sets, six.string_types) or
            not all(isinstance(s, six.string_types) for s in which_sets))
        if which_sets_invalid_value:
            raise ValueError('`which_sets` should be an iterable of strings')
        self.which_sets = which_sets
        self.user_given_subset = subset if subset else slice(None)
        self.load_in_memory = load_in_memory
        self.driver = driver
        self.sort_indices = sort_indices

        self._parse_dataset_info()

        kwargs.setdefault('axis_labels', self.default_axis_labels)
        super(H5PYDataset, self).__init__(**kwargs)

        # It is really important to do it here, because self.num_examples
        # call will cause a crash if done before calling
        # super(...).__init__
        self.example_iteration_scheme = SequentialExampleScheme(
            self.num_examples)

    def _parse_dataset_info(self):
        """Parses information related to the HDF5 interface.

        In addition to verifying that the `self.which_sets` split is
        available, this method sets the following attributes:

        * `provides_sources`
        * `vlen_sources`
        * `default_axis_labels`

        """
        self._out_of_memory_open()
        handle = self._file_handle
        available_splits = self.get_all_splits(handle)
        which_sets = self.which_sets
        provides_sources = None
        for split in which_sets:
            if split not in available_splits:
                raise ValueError(
                    "'{}' split is not provided by this ".format(split) +
                    "dataset. Available splits are " +
                    "{}.".format(available_splits))
            split_provides_sources = set(
                self.get_provided_sources(handle, split))
            if provides_sources:
                provides_sources &= split_provides_sources
            else:
                provides_sources = split_provides_sources
        self.provides_sources = tuple(sorted(provides_sources))
        self.vlen_sources = self.get_vlen_sources(handle)
        self.default_axis_labels = self.get_axis_labels(handle)
        self._out_of_memory_close()

    @staticmethod
    def create_split_array(split_dict):
        """Create a valid array for the `split` attribute of the root node.

        Parameters
        ----------
        split_dict : dict
            Maps split names to dict. Those dict map source names to
            tuples. Those tuples contain two, three or four elements:
            the start index, the stop index, (optionally) subset
            indices and (optionally) a comment.  If a particular
            split/source combination isn't present in the split dict,
            it's considered as unavailable and the `available` element
            will be set to `False` it its split array entry.

        """
        # Determine maximum split, source and string lengths
        split_len = max(len(split) for split in split_dict)
        sources = set()
        comment_len = 1
        for split in split_dict.values():
            sources |= set(split.keys())
            for val in split.values():
                if len(val) == 4:
                    comment_len = max([comment_len, len(val[-1])])
        sources = sorted(list(sources))
        source_len = max(len(source) for source in sources)

        # Instantiate empty split array
        split_array = numpy.empty(
            len(split_dict) * len(sources),
            dtype=numpy.dtype([
                ('split', 'a', split_len),
                ('source', 'a', source_len),
                ('start', numpy.int64, 1),
                ('stop', numpy.int64, 1),
                ('indices', h5py.special_dtype(ref=h5py.Reference)),
                ('available', numpy.bool, 1),
                ('comment', 'a', comment_len)]))

        # Fill split array
        for i, (split, source) in enumerate(product(split_dict, sources)):
            if source in split_dict[split]:
                start, stop = split_dict[split][source][:2]
                available = True
                indices = h5py.Reference()
                # Workaround for bug when pickling an empty string
                comment = '.'
                if len(split_dict[split][source]) > 2:
                    indices = split_dict[split][source][2]
                if len(split_dict[split][source]) > 3:
                    comment = split_dict[split][source][3]
                    if not comment:
                        comment = '.'
            else:
                (start, stop, indices, available, comment) = (
                    0, 0, h5py.Reference(), False, '.')
            # Workaround for H5PY being unable to store unicode type
            split_array[i]['split'] = split.encode('utf8')
            split_array[i]['source'] = source.encode('utf8')
            split_array[i]['start'] = start
            split_array[i]['stop'] = stop
            split_array[i]['indices'] = indices
            split_array[i]['available'] = available
            split_array[i]['comment'] = comment.encode('utf8')

        return split_array

    @staticmethod
    def get_all_splits(h5file):
        """Returns the names of all splits of an HDF5 dataset.

        Parameters
        ----------
        h5file : HDF5 file handle
            An HDF5 dataset respecting the H5PYDataset interface.

        Returns
        -------
        available_splits : tuple of str
            Names of all splits in ``h5file``.

        """
        available_splits = tuple(
            set(row['split'].decode('utf8') for row in h5file.attrs['split']))
        return available_splits

    @staticmethod
    def get_all_sources(h5file):
        """Returns the names of all sources of an HDF5 dataset.

        Parameters
        ----------
        h5file : HDF5 file handle
            An HDF5 dataset respecting the H5PYDataset interface.

        Returns
        -------
        all_sources : tuple of str
            Names of all sources in ``h5file``.

        """
        all_sources = tuple(
            set(row['source'].decode('utf8') for row in h5file.attrs['split']))
        return all_sources

    @staticmethod
    def get_provided_sources(h5file, split):
        """Returns the sources provided by a specific split.

        Parameters
        ----------
        h5file : HDF5 file handle
            An HDF5 dataset respecting the H5PYDataset interface.
        split : str
            Name of the split.

        Returns
        -------
        provided_sources : tuple of str
            Names of sources provided by ``split`` in ``h5file``.

        """
        provided_sources = tuple(
            row['source'].decode('utf8') for row in h5file.attrs['split']
            if row['split'].decode('utf8') == split and row['available'])
        return provided_sources

    @staticmethod
    def get_vlen_sources(h5file):
        """Returns the names of variable-length sources in an HDF5 dataset.

        Parameters
        ----------
        h5file : HDF5 file handle
            An HDF5 dataset respecting the H5PYDataset interface.
        split : str
            Name of the split.

        Returns
        -------
        vlen_sources : tuple of str
            Names of all variable-length sources in ``h5file``.

        """
        vlen_sources = []
        for source_name in H5PYDataset.get_all_sources(h5file):
            source = h5file[source_name]
            if len(source.dims) > 0 and 'shapes' in source.dims[0]:
                if len(source.dims) > 1:
                    raise ValueError('Variable-length sources must have only '
                                     'one dimension.')
                vlen_sources.append(source_name)
        return vlen_sources

    @staticmethod
    def get_axis_labels(h5file):
        """Returns axis labels for all sources in an HDF5 dataset.

        Parameters
        ----------
        h5file : HDF5 file handle
            An HDF5 dataset respecting the H5PYDataset interface.

        Returns
        -------
        axis_labels : dict
            Maps source names to a tuple of str representing the axis
            labels.

        """
        axis_labels = {}
        vlen_sources = H5PYDataset.get_vlen_sources(h5file)
        for source_name in H5PYDataset.get_all_sources(h5file):
            if source_name in vlen_sources:
                axis_labels[source_name] = (
                    (h5file[source_name].dims[0].label,) +
                    tuple(label.decode('utf8') for label in
                          h5file[source_name].dims[0]['shape_labels']))
            else:
                axis_labels[source_name] = tuple(
                    dim.label for dim in h5file[source_name].dims)
        return axis_labels

    @staticmethod
    def get_subsets(h5file, splits, sources):
        """Returns the subsets for a given splits/sources combination.

        Parameters
        ----------
        h5file : HDF5 file handle
            An HDF5 dataset respecting the H5PYDataset interface.
        splits : :class:`tuple` of :class:`str`
            Split names.
        sources : :class:`tuple` of :class:`str`
            Which sources should be considered.

        Returns
        -------
        :class:`list` of :class:`fuel.utils.Subset`
            The subsets, one per source in ``sources``, associated with
            the splits/sources combination.

        """
        subsets = [Subset.empty_subset(len(h5file[source_name]))
                   for source_name in sources]
        for split in splits:
            for i, source in enumerate(sources):
                row, = [r for r in h5file.attrs['split'] if
                        (r['split'].decode('utf8') == split and
                         r['source'].decode('utf8') == source)]
                if row['indices']:
                    subsets[i] += Subset(
                        h5file[row['indices']], len(h5file[source]))
                else:
                    subsets[i] += Subset(
                        slice(row['start'], row['stop']), len(h5file[source]))

        return subsets

    def load(self):
        # If the dataset is unpickled, it makes no sense to have an external
        # file handle. However, since `load` is also called during the lifetime
        # of a dataset (e.g. if load_in_memory = True), we don't want to
        # accidentally overwrite the reference to a potential external file
        # handle, hence this check.
        if not hasattr(self, '_external_file_handle'):
            self.external_file_handle = None

        self._out_of_memory_open()
        handle = self._file_handle

        # Infer subsets based on `which_sets`
        subsets = self.get_subsets(handle, self.which_sets, self.sources)
        # Sanity check to make sure that all sources have equal length
        if any(subset.num_examples != subsets[0].num_examples for subset in
                subsets):
            raise ValueError("sources have different lengths")
        # Produce the final subsets by taking the `subset` constructor argument
        # into account.
        self.subsets = [Subset.subset_of(subset, self.user_given_subset)
                        for subset in subsets]

        # Load data sources and source shapes (if requested)
        if self.load_in_memory:
            data_sources = []
            source_shapes = []
            for source_name, subset in zip(self.sources, self.subsets):
                data_sources.append(
                    subset.index_within_subset(
                        handle[source_name], slice(None)))
                if source_name in self.vlen_sources:
                    shapes = subset.index_within_subset(
                        handle[source_name].dims[0]['shapes'],
                        slice(None))
                else:
                    shapes = None
                source_shapes.append(shapes)
            self.data_sources = tuple(data_sources)
            self.source_shapes = tuple(source_shapes)
            # This exists only for request sanity checking purposes.
            self.in_memory_subset = Subset(
                slice(None), len(self.data_sources[0]))
        else:
            self.data_sources = None
            self.source_shapes = None
            self.in_memory_subset = None

        self._out_of_memory_close()

    @property
    def num_examples(self):
        return self.subsets[0].num_examples

    def open(self):
        return None if self.load_in_memory else self._out_of_memory_open()

    def _out_of_memory_open(self):
        if not self.external_file_handle:
            if self.path not in self._file_handles:
                handle = h5py.File(
                    name=self.path, mode="r", driver=self.driver)
                self._file_handles[self.path] = handle
            self._ref_counts[self.path] += 1

    def close(self, state):
        if not self.load_in_memory:
            self._out_of_memory_close()

    def _out_of_memory_close(self):
        if not self.external_file_handle:
            self._ref_counts[self.path] -= 1
            if not self._ref_counts[self.path]:
                del self._ref_counts[self.path]
                self._file_handles[self.path].close()
                del self._file_handles[self.path]

    @property
    def _file_handle(self):
        if self.external_file_handle:
            return self.external_file_handle
        elif self.path in self._file_handles:
            return self._file_handles[self.path]
        else:
            raise IOError('no open handle for file {}'.format(self.path))

    def get_data(self, state=None, request=None):
        if self.load_in_memory:
            data, shapes = self._in_memory_get_data(state, request)
        else:
            data, shapes = self._out_of_memory_get_data(state, request)
        for i in range(len(data)):
            if shapes[i] is not None:
                if isinstance(request, numbers.Integral):
                    data[i] = data[i].reshape(shapes[i])
                else:
                    for j in range(len(data[i])):
                        data[i][j] = data[i][j].reshape(shapes[i][j])
        return tuple(data)

    def _in_memory_get_data(self, state=None, request=None):
        if state is not None or request is None:
            raise ValueError
        data = [self.in_memory_subset.index_within_subset(data_source, request)
                for data_source in self.data_sources]
        shapes = [self.in_memory_subset.index_within_subset(shape, request)
                  if shape is not None else None
                  for shape in self.source_shapes]
        return data, shapes

    def _out_of_memory_get_data(self, state=None, request=None):
        if not isinstance(request, (numbers.Integral, slice, list)):
            raise ValueError()
        data = []
        shapes = []
        # TODO: This is not an ideal solution, really unpickling should be
        # restoring the state of the dataset fully, i.e. load() must be
        # modified to account for the out-of-memory case.
        # See https://git.io/vKkSm
        try:
            handle = self._file_handle
        except IOError:
            self._out_of_memory_open()
            handle = self._file_handle
        for source_name, subset in zip(self.sources, self.subsets):
            # Process the data request within the context of the data source
            # subset
            data.append(
                subset.index_within_subset(
                    handle[source_name], request,
                    sort_indices=self.sort_indices))
            # If this source has variable length, get the shapes as well
            if source_name in self.vlen_sources:
                shapes.append(
                    subset.index_within_subset(
                        handle[source_name].dims[0]['shapes'], request,
                        sort_indices=self.sort_indices))
            else:
                shapes.append(None)
        return data, shapes


def _dtype_signature(dtype):
    """Returns a hashable description of an HDF5 dtype.

    :class:`numpy.dtype` equality is not enough on its own: every
    variable-length dtype is reported as ``object``, so two datasets whose
    elements have different base types (or ``bytes`` versus ``str``) would
    otherwise look identical. This helper also covers compound and
    sub-array dtypes.

    """
    vlen = h5py.check_dtype(vlen=dtype)
    if vlen is not None:
        if isinstance(vlen, numpy.dtype):
            return ('vlen', 'numpy', vlen.str)
        return ('vlen', 'python', getattr(vlen, '__name__', str(vlen)))
    if dtype.subdtype is not None:
        base, shape = dtype.subdtype
        return ('subarray', tuple(shape), _dtype_signature(base))
    if dtype.names is not None:
        return ('compound', tuple(
            (name, _dtype_signature(dtype.fields[name][0]))
            for name in dtype.names))
    return ('fixed', dtype.str)


class _HandlePool(object):
    """A bounded cache of read-only HDF5 handles.

    Handles are opened on demand and cached, so consecutive requests hitting
    the same shard do not reopen it. When ``max_open_files`` is reached, the
    least recently used handle that no request is currently using is closed
    to make room. A handle in use by a request is never closed by someone
    else's request.

    """
    def __init__(self, max_open_files=None):
        self.max_open_files = max_open_files
        self._handles = OrderedDict()
        self._ref_counts = defaultdict(int)
        self._open_states = 0
        self._lock = threading.RLock()

    def acquire(self, path, driver=None):
        """Opens ``path`` if needed, marks it in use and returns its handle."""
        with self._lock:
            if path in self._handles:
                self._handles.move_to_end(path)
            else:
                self._evict_if_needed()
                self._handles[path] = h5py.File(
                    name=path, mode='r', driver=driver)
            self._ref_counts[path] += 1
            return self._handles[path]

    def release(self, path):
        """Marks ``path`` as no longer used by the calling request."""
        with self._lock:
            if not self._ref_counts[path]:
                return
            self._ref_counts[path] -= 1
            if not self._ref_counts[path]:
                del self._ref_counts[path]

    def _evict_if_needed(self):
        if (self.max_open_files is None or
                len(self._handles) < self.max_open_files):
            return
        for candidate in list(self._handles):
            if not self._ref_counts[candidate]:
                self._handles.pop(candidate).close()
                return
        # Every cached handle is in use: exceeding the bound for the moment
        # is the only option that preserves correctness.

    def open_state(self):
        with self._lock:
            self._open_states += 1

    def close_state(self):
        with self._lock:
            self._open_states = max(0, self._open_states - 1)
            if not self._open_states:
                self._clear_idle()

    def _clear_idle(self):
        for path in list(self._handles):
            if not self._ref_counts[path]:
                self._handles.pop(path).close()

    def close_idle_if_unused(self):
        """Closes cached handles if no stream is currently open."""
        with self._lock:
            if not self._open_states:
                self._clear_idle()

    def close(self):
        with self._lock:
            for handle in self._handles.values():
                handle.close()
            self._handles.clear()
            self._ref_counts.clear()

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_handles'] = OrderedDict()
        state['_ref_counts'] = defaultdict(int)
        state['_open_states'] = 0
        state['_lock'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.RLock()


class _ShardState(object):
    """The state of a :class:`ShardedH5PYDataset` stream.

    Handles live in the dataset-wide :class:`_HandlePool` and are only leased
    for the duration of a request, so this state holds no file reference by
    itself. Closing one stream therefore cannot close a file another stream
    is still reading.

    """

    def __init__(self, max_open_files=None):
        self.max_open_files = max_open_files

    def __getstate__(self):
        return {'max_open_files': self.max_open_files}

    def __setstate__(self, state):
        self.max_open_files = state['max_open_files']


class ShardedH5PYDataset(Dataset):
    """A dataset made of several HDF5 files, sampled as a single dataset.

    Every shard is an HDF5 file respecting the :class:`H5PYDataset`
    interface. Shards are combined in the order in which they are provided,
    each contributing the examples of the split(s) selected for it. The
    combined dataset is then sampled like a single :class:`H5PYDataset`:
    integer, slice and list requests are supported, including lists with
    duplicated or unsorted indices.

    Requests are indices into the concatenation of the shards. If the first
    shard contributes 10 examples and the second one 5, index ``12`` is the
    third example of the second shard. Output always follows the order of the
    request, and every source of an example (features, labels and
    variable-length shapes alike) is read from that same example.

    Shards are validated against one another when the dataset is created.
    They must provide the same sources, with compatible dtypes and axis
    semantics (axis labels, plus shape labels and dimensionality for
    variable-length sources and element shapes for fixed-size ones). An
    incompatible shard raises :class:`ValueError`. A shard contributing no
    example for its selected split(s) is skipped: it adds nothing to the
    index and its metadata is not compared with the other shards.

    Data is read from disk on demand, and a request only touches the shards
    it needs. ``max_open_files`` bounds the number of simultaneously open
    files.

    Parameters
    ----------
    shards : iterable
        Shards, in order. Each element is either a ``(file_or_path,
        which_sets)`` pair or a lone ``file_or_path``, in which case the
        ``which_sets`` argument is used. ``which_sets`` selects the split(s)
        used *within that shard* and accepts the same values as
        :class:`H5PYDataset`'s ``which_sets``. A mapping of files to
        ``which_sets`` is also accepted.
    subset : {slice, list of int}, optional
        Which subset of the combined dataset to use. Defaults to `None`, in
        which case every shard contributes all its selected examples.
    load_in_memory : bool, optional
        Whether to load the selected data in main memory. Defaults to
        `False`.
    driver : str, optional
        Low-level h5py driver used to open the shard files. Defaults to
        `None`.
    sort_indices : bool, optional
        Kept for API compatibility with :class:`H5PYDataset`. Duplicated and
        unsorted indices are always supported; the flag only indicates that
        reads may be reordered internally. Defaults to `True`.
    max_open_files : int, optional
        Maximum number of shard files kept open at once. Defaults to `None`,
        which means no bound.
    which_sets : str or tuple of str, optional
        Split(s) used for shards given without their own ``which_sets``.

    Attributes
    ----------
    shards : tuple of (str, tuple of str)
        The ``(path, which_sets)`` pairs, in order.
    shard_files : tuple of str
        Paths of the shards, in order.
    shard_splits : tuple of tuple of str
        Splits selected for each shard.
    shard_num_examples : tuple of int
        Number of examples contributed by each shard.

    """
    interface_version = '0.3'

    def __init__(self, shards, subset=None, load_in_memory=False, driver=None,
                 sort_indices=True, max_open_files=None, which_sets=None,
                 **kwargs):
        # Be forgiving about the common ``(shards, which_sets)`` call form:
        # a string or tuple of strings in second position can only be a
        # split selection, never a subset.
        if (which_sets is None and
                (isinstance(subset, six.string_types) or
                 (isinstance(subset, tuple) and subset and
                  all(isinstance(split, six.string_types)
                      for split in subset)))):
            which_sets, subset = subset, None
        if max_open_files is not None:
            if (not isinstance(max_open_files, numbers.Integral) or
                    max_open_files < 1):
                raise ValueError(
                    '`max_open_files` should be a positive integer or None')
        self.driver = driver
        self.sort_indices = sort_indices
        self.load_in_memory = load_in_memory
        self.max_open_files = max_open_files
        self.user_given_subset = subset if subset else slice(None)

        shards = self._normalize_shards(shards, which_sets)
        if not shards:
            raise ValueError('`shards` should contain at least one shard')

        self._external_handles = {}
        self._handle_pool = _HandlePool(max_open_files)
        self._data_sources = None
        self._source_shapes = None

        self._parse_shards(shards)

        kwargs.setdefault('axis_labels', self.default_axis_labels)
        super(ShardedH5PYDataset, self).__init__(**kwargs)
        self.example_iteration_scheme = SequentialExampleScheme(
            self.num_examples)

        if self.load_in_memory:
            self.load()

    # -- Construction --------------------------------------------------------

    @staticmethod
    def _normalize_which_sets(which_sets):
        if isinstance(which_sets, six.string_types):
            return (which_sets,)
        try:
            which_sets = tuple(which_sets)
        except TypeError:
            raise ValueError('`which_sets` should be an iterable of strings')
        if not which_sets or not all(
                isinstance(split, six.string_types) for split in which_sets):
            raise ValueError('`which_sets` should be an iterable of strings')
        return which_sets

    def _normalize_shards(self, shards, default_which_sets):
        if isinstance(shards, six.string_types):
            raise ValueError(
                '`shards` should be an iterable of shards, not a string')
        if isinstance(shards, dict):
            shards = list(shards.items())
        else:
            shards = list(shards)
        normalized = []
        for entry in shards:
            if (isinstance(entry, (tuple, list)) and len(entry) == 2 and
                    isinstance(entry[0], (six.string_types, h5py.File))):
                file_or_path, entry_which_sets = entry
            else:
                file_or_path, entry_which_sets = entry, default_which_sets
            if not isinstance(file_or_path, (six.string_types, h5py.File)):
                raise ValueError(
                    'shards should be file paths, h5py.File handles or '
                    '(file, which_sets) pairs')
            if entry_which_sets is None:
                raise ValueError(
                    'no split given for shard {!r}; pass a `(file, '
                    'which_sets)` pair or the `which_sets` argument'.format(
                        file_or_path))
            normalized.append((
                file_or_path,
                self._normalize_which_sets(entry_which_sets)))
        return normalized

    @staticmethod
    def _open_metadata_handle(file_or_path, path, driver):
        if isinstance(file_or_path, h5py.File):
            return file_or_path
        return h5py.File(name=path, mode='r', driver=driver)

    @staticmethod
    def _close_metadata_handle(file_or_path, handle):
        if not isinstance(file_or_path, h5py.File):
            handle.close()

    @staticmethod
    def _shard_metadata(handle, which_sets):
        available_splits = H5PYDataset.get_all_splits(handle)
        for split in which_sets:
            if split not in available_splits:
                raise ValueError(
                    "'{}' split is not provided by this shard. Available "
                    "splits are {}.".format(split, available_splits))
        provided_sources = None
        for split in which_sets:
            split_sources = set(
                H5PYDataset.get_provided_sources(handle, split))
            if provided_sources is None:
                provided_sources = split_sources
            else:
                provided_sources &= split_sources
        sources = tuple(sorted(provided_sources))
        vlen_in_file = set(H5PYDataset.get_vlen_sources(handle))
        all_axis_labels = H5PYDataset.get_axis_labels(handle)
        dtypes = {}
        numpy_dtypes = {}
        element_shapes = {}
        vlen_sources = []
        vlen_shape_shapes = {}
        axis_labels = {}
        for source in sources:
            dataset = handle[source]
            dtypes[source] = _dtype_signature(dataset.dtype)
            numpy_dtypes[source] = dataset.dtype
            element_shapes[source] = tuple(dataset.shape[1:])
            axis_labels[source] = all_axis_labels[source]
            if source in vlen_in_file:
                vlen_sources.append(source)
                vlen_shape_shapes[source] = tuple(
                    dataset.dims[0]['shapes'].shape[1:])
        return {
            'sources': sources,
            'dtypes': dtypes,
            'numpy_dtypes': numpy_dtypes,
            'element_shapes': element_shapes,
            'vlen_sources': tuple(sorted(vlen_sources)),
            'vlen_shape_shapes': vlen_shape_shapes,
            'axis_labels': axis_labels,
        }

    @staticmethod
    def _get_shard_subsets(handle, which_sets, sources):
        """Returns one :class:`Subset` per source for a shard.

        This mirrors :meth:`H5PYDataset.get_subsets`, but tolerates empty
        split ranges (which `Subset` rejects) so that shards without any
        example for the selected split can be skipped cleanly.

        """
        subsets = [Subset.empty_subset(len(handle[source]))
                   for source in sources]
        for split in which_sets:
            for i, source in enumerate(sources):
                row, = [r for r in handle.attrs['split'] if
                        (r['split'].decode('utf8') == split and
                         r['source'].decode('utf8') == source)]
                if row['indices']:
                    subsets[i] += Subset(
                        handle[row['indices']], len(handle[source]))
                else:
                    start, stop = int(row['start']), int(row['stop'])
                    if start >= stop:
                        continue
                    subsets[i] += Subset(
                        slice(start, stop), len(handle[source]))
        return subsets

    def _parse_shards(self, shards):
        shard_files = []
        shard_splits = []
        shard_num_examples = []
        shard_subsets = []
        reference = None
        reference_path = None

        for index, (file_or_path, splits) in enumerate(shards):
            if isinstance(file_or_path, h5py.File):
                path = file_or_path.filename
                self._external_handles[index] = file_or_path
            else:
                path = cache_file(file_or_path)

            handle = self._open_metadata_handle(file_or_path, path,
                                                self.driver)
            try:
                metadata = self._shard_metadata(handle, splits)
                subsets = self._get_shard_subsets(
                    handle, splits, metadata['sources'])
            finally:
                self._close_metadata_handle(file_or_path, handle)

            num_examples = subsets[0].num_examples if subsets else 0
            if any(subset.num_examples != num_examples
                   for subset in subsets):
                raise ValueError(
                    'sources of shard {} have different lengths'.format(path))
            if num_examples:
                if reference is None:
                    reference = metadata
                    reference_path = path
                else:
                    self._check_compatibility(
                        reference, reference_path, metadata, path)

            shard_files.append(path)
            shard_splits.append(splits)
            shard_num_examples.append(num_examples)
            shard_subsets.append(subsets)

        if reference is None:
            raise ValueError(
                'the provided shards contain no example for the selected '
                'split(s)')

        self.shard_files = tuple(shard_files)
        self.shard_splits = tuple(shard_splits)
        self.shard_num_examples = tuple(shard_num_examples)
        self.shards = tuple(zip(self.shard_files, self.shard_splits))

        self.provides_sources = reference['sources']
        self.vlen_sources = reference['vlen_sources']
        self.default_axis_labels = reference['axis_labels']
        self._numpy_dtypes = reference['numpy_dtypes']
        self._element_shapes = reference['element_shapes']
        self._vlen_shape_shapes = reference['vlen_shape_shapes']
        self._vlen_set = set(self.vlen_sources)

        # `_shard_subsets[source][shard_index]` maps a local example index to
        # a row of that shard's dataset. Empty shards have no entry, since
        # they are never read.
        self._shard_subsets = {}
        for source in self.provides_sources:
            source_index = reference['sources'].index(source)
            self._shard_subsets[source] = [
                None if not shard_num_examples[index] else
                shard_subsets[index][source_index]
                for index in range(len(shard_files))]

        # Cumulative shard boundaries in the combined index space. Empty
        # shards contribute a zero-length range.
        self._shard_bounds = []
        self._shard_starts = []
        start = 0
        for num_examples in shard_num_examples:
            self._shard_bounds.append((start, start + num_examples))
            self._shard_starts.append(start)
            start += num_examples
        self._num_examples_total = start

        # The `subset` argument selects within the combined index space.
        # `Subset` sorts and de-duplicates list-based subsets, matching
        # `H5PYDataset`'s documented behaviour.
        self._user_subset = Subset(
            self.user_given_subset, self._num_examples_total)

    def _check_compatibility(self, reference, reference_path, candidate,
                             path):
        if reference['sources'] != candidate['sources']:
            raise ValueError(
                'incompatible shards: shard {} provides sources {} while '
                'shard {} provides sources {}'.format(
                    reference_path, reference['sources'], path,
                    candidate['sources']))
        if reference['vlen_sources'] != candidate['vlen_sources']:
            raise ValueError(
                'incompatible shards: shard {} has variable-length sources '
                '{} while shard {} has {}'.format(
                    reference_path, reference['vlen_sources'], path,
                    candidate['vlen_sources']))
        for source in reference['sources']:
            if reference['dtypes'][source] != candidate['dtypes'][source]:
                raise ValueError(
                    'incompatible shards: source {} has dtype {} in shard {} '
                    'but dtype {} in shard {}'.format(
                        source, reference['dtypes'][source], reference_path,
                        candidate['dtypes'][source], path))
            if (reference['axis_labels'][source] !=
                    candidate['axis_labels'][source]):
                raise ValueError(
                    'incompatible shards: source {} has axis labels {} in '
                    'shard {} but {} in shard {}'.format(
                        source, reference['axis_labels'][source],
                        reference_path, candidate['axis_labels'][source],
                        path))
            if source in reference['vlen_sources']:
                if (reference['vlen_shape_shapes'][source] !=
                        candidate['vlen_shape_shapes'][source]):
                    raise ValueError(
                        'incompatible shards: variable-length source {} has '
                        '{} dimensions in shard {} but {} in shard {}'.format(
                            source,
                            len(reference['vlen_shape_shapes'][source]),
                            reference_path,
                            len(candidate['vlen_shape_shapes'][source]),
                            path))
            elif (reference['element_shapes'][source] !=
                    candidate['element_shapes'][source]):
                raise ValueError(
                    'incompatible shards: source {} has element shape {} in '
                    'shard {} but {} in shard {}'.format(
                        source, reference['element_shapes'][source],
                        reference_path,
                        candidate['element_shapes'][source], path))

    # -- Pickling ------------------------------------------------------------

    def __getstate__(self):
        state = self.__dict__.copy()
        # Live file handles and locks cannot be pickled. In-memory data is
        # dropped so that it is reloaded lazily after unpickling, as with
        # `H5PYDataset`.
        state['_external_handles'] = {}
        state['_handle_pool'] = None
        state['_data_sources'] = None
        state['_source_shapes'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._external_handles = {}
        self._handle_pool = _HandlePool(self.max_open_files)

    # -- Index translation ---------------------------------------------------

    @property
    def num_examples(self):
        return self._user_subset.num_examples

    def _translate_request(self, request):
        """Translates a request into the combined index space.

        Returns a ``(kind, value)`` pair, where ``kind`` is ``'int'``,
        ``'list'`` or ``'slice'`` and ``value`` is respectively an index, a
        list of indices in request order, or a ``(start, stop, step)``
        tuple.

        """
        subset = self._user_subset
        if isinstance(request, numbers.Integral):
            global_index, = subset[[int(request)]]
            return 'int', int(global_index)
        translated = subset[request]
        if isinstance(translated, slice):
            return 'slice', translated.indices(self._num_examples_total)
        return 'list', [int(index) for index in translated]

    def _locate(self, global_index):
        """Maps a combined index to ``(shard_index, local_index)``."""
        if global_index < 0 or global_index >= self._num_examples_total:
            raise ValueError(
                'index {} is out of range for a dataset with {} '
                'examples'.format(global_index, self._num_examples_total))
        # Empty shards share their start with the following shard, so
        # `bisect_right` naturally returns the next non-empty one.
        shard_index = bisect_right(self._shard_starts, global_index) - 1
        return shard_index, global_index - self._shard_starts[shard_index]

    @staticmethod
    def _map_rows(subset, local_indices):
        """Maps sorted local example indices to rows of a shard dataset."""
        if subset.is_list:
            return [int(subset.list_or_slice[index])
                    for index in local_indices]
        start, _, step = subset.slice_to_numerical_args(
            subset.list_or_slice, subset.original_num_examples)
        if isinstance(local_indices, range):
            return range(start + step * local_indices.start,
                         start + step * local_indices.stop,
                         step * local_indices.step)
        return [int(start + step * index) for index in local_indices]

    @staticmethod
    def _read_rows(dataset, rows):
        """Reads ``rows`` from ``dataset``, which must be increasing."""
        if isinstance(rows, range):
            if not len(rows):
                return dataset[0:0]
            return dataset[rows.start:rows.stop:rows.step]
        if not rows:
            return dataset[0:0]
        if all(rows[i] + 1 == rows[i + 1] for i in range(len(rows) - 1)):
            return dataset[rows[0]:rows[-1] + 1]
        return dataset[numpy.array(rows)]

    def _slice_plan(self, start, stop, step):
        """Splits a slice into per-shard (local range, output position)."""
        plans = []
        for shard_index, (shard_start, shard_stop) in enumerate(
                self._shard_bounds):
            low = max(start, shard_start)
            high = min(stop, shard_stop)
            if low >= high:
                continue
            # Smallest index of the progression that is >= low.
            first = start + ((low - start + step - 1) // step) * step
            if first >= high:
                continue
            num = (high - 1 - first) // step + 1
            last = first + (num - 1) * step
            local = range(first - shard_start, last - shard_start + 1, step)
            # Output positions are contiguous: they advance by one per
            # requested index.
            plans.append((shard_index, local, (first - start) // step, num))
        return plans

    # -- Handles -------------------------------------------------------------

    def _acquire_handle(self, shard_index):
        if shard_index in self._external_handles:
            return self._external_handles[shard_index], None
        path = self.shard_files[shard_index]
        return self._handle_pool.acquire(path, self.driver), path

    def _release_handle(self, lease):
        if lease is not None:
            self._handle_pool.release(lease)

    # -- Dataset interface ---------------------------------------------------

    @property
    def data_sources(self):
        if self.load_in_memory and self._data_sources is None:
            self.load()
        return self._data_sources

    @property
    def source_shapes(self):
        if self.load_in_memory and self._source_shapes is None:
            self.load()
        return self._source_shapes

    def load(self):
        """Loads the selected data in memory when ``load_in_memory`` is set."""
        if not self.load_in_memory:
            self._data_sources = None
            self._source_shapes = None
            return
        # Read the subset itself rather than the whole dataset followed by a
        # selection, so that only the requested examples are loaded.
        kind, value = self._translate_request(slice(None))
        if kind == 'int':
            data, shapes = self._read_int(value)
            data = [numpy.array([source_data]) for source_data in data]
            shapes = [None if source_shapes is None
                      else numpy.array([source_shapes])
                      for source_shapes in shapes]
        else:
            data, shapes = self._read(kind, value)
        self._data_sources = tuple(data)
        self._source_shapes = tuple(shapes)

    def open(self):
        if self.load_in_memory:
            return None
        self._handle_pool.open_state()
        return _ShardState(self.max_open_files)

    def close(self, state):
        if self.load_in_memory:
            return
        # Leases are per-request, so a stream holds no handle while idle.
        # When the last stream closes, idle cached handles are released.
        if state is not None:
            self._handle_pool.close_state()
        else:
            self._handle_pool.close_idle_if_unused()

    def get_data(self, state=None, request=None):
        if self.load_in_memory:
            data, shapes = self._in_memory_get_data(state, request)
        else:
            data, shapes = self._out_of_memory_get_data(state, request)
        for i in range(len(data)):
            if shapes[i] is not None:
                if isinstance(request, numbers.Integral):
                    data[i] = data[i].reshape(shapes[i])
                else:
                    for j in range(len(data[i])):
                        data[i][j] = data[i][j].reshape(shapes[i][j])
        return tuple(data)

    def _in_memory_get_data(self, state=None, request=None):
        if state is not None or request is None:
            raise ValueError
        if not isinstance(request, (numbers.Integral, slice, list)):
            raise ValueError
        identity = Subset(slice(None), self.num_examples)
        data = [identity.index_within_subset(data_source, request)
                for data_source in self.data_sources]
        shapes = [None if shape is None else
                  identity.index_within_subset(shape, request)
                  for shape in self.source_shapes]
        return data, shapes

    def _out_of_memory_get_data(self, state=None, request=None):
        if not isinstance(request, (numbers.Integral, slice, list)):
            raise ValueError()
        kind, value = self._translate_request(request)
        if kind == 'int':
            data, shapes = self._read_int(value)
        else:
            data, shapes = self._read(kind, value)
        return data, shapes

    # -- Reading -------------------------------------------------------------

    def _allocate(self, num_requested):
        data = []
        shapes = []
        for source in self.sources:
            if source in self._vlen_set:
                data.append(numpy.empty((num_requested,), dtype='object'))
                shapes.append(numpy.empty(
                    (num_requested,) + self._vlen_shape_shapes[source],
                    dtype='int64'))
            else:
                data.append(numpy.empty(
                    (num_requested,) + self._element_shapes[source],
                    dtype=self._numpy_dtypes[source]))
                shapes.append(None)
        return data, shapes

    def _read_int(self, global_index):
        shard_index, local_index = self._locate(global_index)
        handle, lease = self._acquire_handle(shard_index)
        data = []
        shapes = []
        try:
            for source in self.sources:
                subset = self._shard_subsets[source][shard_index]
                rows = self._map_rows(subset, [local_index])
                data.append(self._read_rows(handle[source], rows)[0])
                if source in self._vlen_set:
                    shapes.append(self._read_rows(
                        handle[source].dims[0]['shapes'], rows)[0])
                else:
                    shapes.append(None)
        finally:
            self._release_handle(lease)
        return data, shapes

    def _read(self, kind, value):
        """Reads a translated list or slice request.

        The returned arrays follow the order of the request. All sources of
        an example, including its variable-length shape, come from the same
        shard row.

        """
        if kind == 'list':
            num_requested = len(value)
            per_shard = OrderedDict()
            for position, global_index in enumerate(value):
                shard_index, local_index = self._locate(global_index)
                per_shard.setdefault(shard_index, []).append(
                    (local_index, position))
            plans = None
        else:
            start, stop, step = value
            num_requested = len(range(start, stop, step))
            per_shard = None
            plans = self._slice_plan(start, stop, step)

        data, shapes = self._allocate(num_requested)

        if kind == 'list':
            for shard_index, entries in per_shard.items():
                handle, lease = self._acquire_handle(shard_index)
                try:
                    # Read every distinct local index once, in increasing
                    # order as required by HDF5 fancy indexing.
                    local_indices = sorted(set(index for index, _ in entries))
                    slots = {index: slot for slot, index in
                             enumerate(local_indices)}
                    for source_index, source in enumerate(self.sources):
                        subset = self._shard_subsets[source][shard_index]
                        rows = self._map_rows(subset, local_indices)
                        values = self._read_rows(handle[source], rows)
                        if shapes[source_index] is not None:
                            value_shapes = self._read_rows(
                                handle[source].dims[0]['shapes'], rows)
                        else:
                            value_shapes = None
                        for local_index, position in entries:
                            slot = slots[local_index]
                            data[source_index][position] = values[slot]
                            if value_shapes is not None:
                                shapes[source_index][position] = \
                                    value_shapes[slot]
                finally:
                    self._release_handle(lease)
            return data, shapes

        for shard_index, local, position, num in plans:
            handle, lease = self._acquire_handle(shard_index)
            try:
                for source_index, source in enumerate(self.sources):
                    subset = self._shard_subsets[source][shard_index]
                    rows = self._map_rows(subset, local)
                    data[source_index][position:position + num] = \
                        self._read_rows(handle[source], rows)
                    if shapes[source_index] is not None:
                        shapes[source_index][position:position + num] = \
                            self._read_rows(
                                handle[source].dims[0]['shapes'], rows)
            finally:
                self._release_handle(lease)
        return data, shapes
