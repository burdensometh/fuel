from abc import ABCMeta, abstractmethod
from collections import Iterable

import numpy
from picklable_itertools import chain, repeat, imap, iter_
from picklable_itertools.extras import partition_all
from six import add_metaclass
from six.moves import xrange

from fuel import config


@add_metaclass(ABCMeta)
class IterationScheme(object):
    """An iteration scheme.

    Iteration schemes provide a dataset-agnostic iteration scheme, such as
    sequential batches, shuffled batches, etc. for datasets that choose to
    support them.

    Attributes
    ----------
    requests_examples : bool
        Whether requests produced by this scheme correspond to single
        examples (as opposed to batches).

    Notes
    -----
    Iteration schemes implement the :meth:`get_request_iterator` method,
    which returns an iterator type (e.g. a generator or a class which
    implements the `iterator protocol`_).

    Stochastic iteration schemes should generally not be shared between
    different data streams, because it would make experiments harder to
    reproduce.

    .. _iterator protocol:
       https://docs.python.org/3.3/library/stdtypes.html#iterator-types

    """
    @abstractmethod
    def get_request_iterator(self):
        """Returns an iterator type."""


@add_metaclass(ABCMeta)
class BatchSizeScheme(IterationScheme):
    """Iteration scheme that returns batch sizes.

    For infinite datasets it doesn't make sense to provide indices to
    examples, but the number of samples per batch can still be given.
    Hence BatchSizeScheme is the base class for iteration schemes
    that only provide the number of examples that should be in a batch.

    """
    requests_examples = False


@add_metaclass(ABCMeta)
class BatchScheme(IterationScheme):
    """Iteration schemes that return slices or indices for batches.

    For datasets where the number of examples is known and easily
    accessible (as is the case for most datasets which are small enough
    to be kept in memory, like MNIST) we can provide slices or lists of
    labels to the dataset.

    Parameters
    ----------
    examples : int or list
        Defines which examples from the dataset are iterated.
        If list, its items are the indices of examples.
        If an integer, it will use that many examples from the beginning
        of the dataset, i.e. it is interpreted as range(examples)
    batch_size : int
        The request iterator will return slices or list of indices in
        batches of size `batch_size` until the end of `examples` is
        reached.
        Note that this means that the last batch size returned could be
        smaller than `batch_size`. If you want to ensure all batches are
        of equal size, then ensure len(`examples`) or `examples` is a
        multiple of `batch_size`.

    """
    requests_examples = False

    def __init__(self, examples, batch_size):
        if isinstance(examples, Iterable):
            self.indices = examples
        else:
            self.indices = xrange(examples)
        self.batch_size = batch_size


class ConcatenatedScheme(IterationScheme):
    """Build an iterator by concatenating several schemes' iterators.

    Useful for iterating through different subsets of data in a specific
    order.

    Parameters
    ----------
    schemes : list
        A list of :class:`IterationSchemes`, whose request iterators
        are to be concatenated in the order given.

    Notes
    -----
    All schemes being concatenated must produce the same type of
    requests (batches or examples).

    """
    def __init__(self, schemes):
        if not len(set(scheme.requests_examples for scheme in schemes)) == 1:
            raise ValueError('all schemes must produce the same type of '
                             'requests (batches or examples)')
        self.schemes = schemes

    def get_request_iterator(self):
        return chain(*[sch.get_request_iterator() for sch in self.schemes])

    @property
    def requests_examples(self):
        return self.schemes[0].requests_examples


@add_metaclass(ABCMeta)
class IndexScheme(IterationScheme):
    """Iteration schemes that return single indices.

    This is for datasets that support indexing (like :class:`BatchScheme`)
    but where we want to return single examples instead of batches.

    """
    requests_examples = True

    def __init__(self, examples):
        if isinstance(examples, Iterable):
            self.indices = examples
        else:
            self.indices = xrange(examples)


class ConstantScheme(BatchSizeScheme):
    """Constant batch size iterator.

    This subset iterator simply returns the same constant batch size
    for a given number of times (or else infinitely).

    Parameters
    ----------
    batch_size : int
        The size of the batch to return.
    num_examples : int, optional
        If given, the request iterator will return `batch_size` until the
        sum reaches `num_examples`. Note that this means that the last
        batch size returned could be smaller than `batch_size`. If you want
        to ensure all batches are of equal size, then pass `times` equal to
        ``num_examples / batch-size`` instead.
    times : int, optional
        The number of times to return `batch_size`.

    """
    def __init__(self, batch_size, num_examples=None, times=None):
        if num_examples and times:
            raise ValueError
        self.batch_size = batch_size
        self.num_examples = num_examples
        self.times = times

    def get_request_iterator(self):
        if self.times:
            return repeat(self.batch_size, self.times)
        if self.num_examples:
            d, r = divmod(self.num_examples, self.batch_size)
            return chain(repeat(self.batch_size, d), [r] if r else [])
        return repeat(self.batch_size)


class SequentialScheme(BatchScheme):
    """Sequential batches iterator.

    Iterate over all the examples in a dataset of fixed size sequentially
    in batches of a given size.

    Notes
    -----
    The batch size isn't enforced, so the last batch could be smaller.

    """
    def get_request_iterator(self):
        return imap(list, partition_all(self.batch_size, self.indices))


class ShuffledScheme(BatchScheme):
    """Shuffled batches iterator.

    Iterate over all the examples in a dataset of fixed size in shuffled
    batches.

    Parameters
    ----------
    sorted_indices : bool, optional
        If `True`, enforce that indices within a batch are ordered.
        Defaults to `False`.

    Notes
    -----
    The batch size isn't enforced, so the last batch could be smaller.

    Shuffling the batches requires creating a shuffled list of indices in
    memory. This can be memory-intensive for very large numbers of examples
    (i.e. in the order of tens of millions).

    """
    def __init__(self, *args, **kwargs):
        self.rng = kwargs.pop('rng', None)
        if self.rng is None:
            self.rng = numpy.random.RandomState(config.default_seed)
        self.sorted_indices = kwargs.pop('sorted_indices', False)
        super(ShuffledScheme, self).__init__(*args, **kwargs)

    def get_request_iterator(self):
        indices = list(self.indices)
        self.rng.shuffle(indices)
        if self.sorted_indices:
            return imap(sorted, partition_all(self.batch_size, indices))
        else:
            return imap(list, partition_all(self.batch_size, indices))


class SequentialExampleScheme(IndexScheme):
    """Sequential examples iterator.

    Returns examples in order.

    """
    def get_request_iterator(self):
        return iter_(self.indices)


class ShuffledExampleScheme(IndexScheme):
    """Shuffled examples iterator.

    Returns examples in random order.

    """
    def __init__(self, *args, **kwargs):
        self.rng = kwargs.pop('rng', None)
        if self.rng is None:
            self.rng = numpy.random.RandomState(config.default_seed)
        super(ShuffledExampleScheme, self).__init__(*args, **kwargs)

    def get_request_iterator(self):
        indices = list(self.indices)
        self.rng.shuffle(indices)
        return iter_(indices)


def cross_validation(scheme_class, num_examples, num_folds, strict=True,
                     **kwargs):
    """Return pairs of schemes to be used for cross-validation.

    Parameters
    ----------
    scheme_class : subclass of :class:`IndexScheme` or :class:`BatchScheme`
        The type of the returned schemes. The constructor is called with an
        iterator and `**kwargs` as arguments.
    num_examples : int
        The number of examples in the datastream.
    num_folds : int
        The number of folds to return.
    strict : bool, optional
        If `True`, enforce that `num_examples` is divisible by `num_folds`
        and so, that all validation sets have the same size. If `False`,
        the size of the validation set is returned along the iteration
        schemes. Defaults to `True`.

    Yields
    ------
    fold : tuple
        The generator returns `num_folds` tuples. The first two elements of
        the tuple are the training and validation iteration schemes. If
        `strict` is set to `False`, the tuple has a third element
        corresponding to the size of the validation set.

    """
    if strict and num_examples % num_folds != 0:
        raise ValueError(("{} examples are not divisible in {} evenly-sized " +
                          "folds. To allow this, have a look at the " +
                          "`strict` argument.").format(num_examples,
                                                       num_folds))

    for i in xrange(num_folds):
        begin = num_examples * i // num_folds
        end = num_examples * (i+1) // num_folds
        train = scheme_class(list(chain(xrange(0, begin),
                                        xrange(end, num_examples))),
                             **kwargs)
        valid = scheme_class(xrange(begin, end), **kwargs)

        if strict:
            yield (train, valid)
        else:
            yield (train, valid, end - begin)


class TimeSeriesCrossValidation(object):
    """Purged and embargoed cross-validation for time-series examples.

    Time-series examples typically describe an interval of time: the
    features are observed starting at a given time while the labels
    cover a period extending into the future. Splitting such examples by
    position therefore leaks validation information into the training
    set, as a training example may overlap in time with a validation
    example. This cross-validation generator prevents that by excluding
    from each training fold every example that

    * shares its entity (e.g. a subject, a customer, a recording device)
      with any validation example of the fold, or
    * covers a half-open interval ``[feature_start, label_end)`` that
      intersects the half-open coverage interval of any validation
      example of the fold, or
    * intersects the embargo (purge) window starting at the fold's
      latest validation label end.

    All intervals are treated as left-closed and right-open, so
    ``[a, b)`` and ``[b, c)`` do not intersect.

    Parameters
    ----------
    feature_starts : array-like, 1D
        Feature start time of every example, inclusive.
    label_ends : array-like, 1D
        Label end time of every example, exclusive. Must have the same
        length as `feature_starts` and satisfy ``feature_start <=
        label_end`` for every example. Numeric or ``datetime64`` arrays
        are supported.
    entities : array-like, 1D
        Entity identifier of every example. Must have the same length as
        `feature_starts`.
    validation_folds : iterable of array-likes
        Each element contains the example indices that make up one
        validation fold. Folds must be non-empty, indices must be valid
        example indices, and no index may appear in more than one fold.
    embargo : int, float or :class:`numpy.timedelta64`, optional
        Length of the embargo window following the latest validation
        label end of each fold. Any training example whose interval
        intersects the half-open window
        ``[latest_validation_end, latest_validation_end + embargo)`` is
        excluded. Must be non-negative and expressed in the same units
        as `label_ends`. Defaults to 0, i.e. no embargo.
    scheme_class : subclass of :class:`IterationScheme`, optional
        Class used to build the train and validation schemes returned by
        :meth:`get_fold_schemes`. It must accept a list of example
        indices as its first argument, like :class:`SequentialScheme`
        or :class:`SequentialExampleScheme`. Defaults to
        :class:`SequentialExampleScheme`.
    scheme_kwargs : dict, optional
        Extra keyword arguments passed to `scheme_class`, e.g.
        ``{'batch_size': 128}`` for a :class:`BatchScheme`.

    Attributes
    ----------
    folds : list of tuples
        One ``(train_indices, validation_indices)`` tuple per fold, each
        a sorted 1D :class:`numpy.ndarray` of example indices.
    fold_exclusion_stats : list of dicts
        For each fold, the number of examples excluded for each reason
        (``validation``, ``same_entity``, ``interval_overlap`` and
        ``embargo``). Note that an example may be counted under more than
        one exclusion reason.

    Raises
    ------
    ValueError
        If the inputs are malformed, or if a fold ends up with an empty
        training set. The empty-fold error message reports the fold
        number and the number of exclusions per reason.

    Examples
    --------
    >>> cv = TimeSeriesCrossValidation(
    ...     feature_starts=[0, 2, 4], label_ends=[1, 5, 7],
    ...     entities=['a', 'a', 'b'], validation_folds=[[2]], embargo=1)
    >>> train_scheme, valid_scheme = next(iter(cv))
    >>> list(train_scheme.get_request_iterator())
    [0]
    >>> list(valid_scheme.get_request_iterator())
    [2]

    The returned schemes can be passed directly to a
    :class:`~fuel.streams.DataStream`::

        from fuel.streams import DataStream
        for train_scheme, valid_scheme in cv:
            train_stream = DataStream(dataset,
                                      iteration_scheme=train_scheme)
            valid_stream = DataStream(dataset,
                                      iteration_scheme=valid_scheme)

    """

    def __init__(self, feature_starts, label_ends, entities,
                 validation_folds, embargo=0,
                 scheme_class=SequentialExampleScheme, scheme_kwargs=None):
        starts = numpy.asarray(feature_starts)
        ends = numpy.asarray(label_ends)
        entity_array = numpy.asarray(entities)

        for name, array in (('feature_starts', starts),
                            ('label_ends', ends),
                            ('entities', entity_array)):
            if array.ndim != 1:
                raise ValueError(
                    '`{}` must be one-dimensional, got an array with '
                    'shape {}'.format(name, array.shape))
        if not (len(starts) == len(ends) == len(entity_array)):
            raise ValueError(
                '`feature_starts`, `label_ends` and `entities` must have '
                'the same length, got {}, {} and {}'.format(
                    len(starts), len(ends), len(entity_array)))
        if numpy.any(starts > ends):
            raise ValueError('every feature start must be smaller than or '
                             'equal to its label end')
        embargo_array = numpy.asarray(embargo)
        if numpy.issubdtype(embargo_array.dtype, numpy.timedelta64):
            is_negative = embargo_array < numpy.timedelta64(0)
        else:
            is_negative = embargo_array < 0
        if is_negative:
            raise ValueError('`embargo` must be non-negative')

        num_examples = len(starts)
        folds = []
        seen_indices = set()
        for fold_number, fold in enumerate(validation_folds):
            indices = numpy.asarray(fold)
            if (indices.ndim != 1 or
                    indices.dtype.kind not in ('i', 'u') or
                    indices.size == 0):
                raise ValueError(
                    'fold {} must be a non-empty one-dimensional array of '
                    'integer indices'.format(fold_number))
            if numpy.any(indices < 0) or numpy.any(indices >= num_examples):
                raise ValueError(
                    'fold {} contains indices outside the range '
                    '[0, {})'.format(fold_number, num_examples))
            if len(set(indices.tolist())) != len(indices):
                raise ValueError(
                    'fold {} contains duplicate indices'.format(fold_number))
            repeated = seen_indices.intersection(indices.tolist())
            if repeated:
                raise ValueError(
                    'indices {} appear in more than one fold'.format(
                        sorted(repeated)))
            seen_indices.update(indices.tolist())
            folds.append(numpy.sort(indices))
        if not folds:
            raise ValueError('at least one validation fold is required')

        self._feature_starts = starts
        self._label_ends = ends
        self._entities = entity_array
        self._embargo = embargo
        self.scheme_class = scheme_class
        self.scheme_kwargs = dict(scheme_kwargs) if scheme_kwargs else {}

        self.folds = []
        self.fold_exclusion_stats = []
        for fold_number, validation_indices in enumerate(folds):
            train_indices, stats = self._build_fold(validation_indices)
            if train_indices.size == 0:
                raise ValueError(
                    ('fold {fold} has an empty training set: of {n} '
                     'examples, {validation} are validation examples, '
                     '{same_entity} share an entity with a validation '
                     'example, {interval_overlap} have intervals '
                     'intersecting validation coverage and {embargo} '
                     'intersect the embargo window').format(
                        fold=fold_number, n=num_examples, **stats))
            self.folds.append((train_indices, validation_indices))
            self.fold_exclusion_stats.append(stats)

    @property
    def num_examples(self):
        """Number of examples the cross-validation is defined over."""
        return len(self._feature_starts)

    @property
    def num_folds(self):
        """Number of validation folds."""
        return len(self.folds)

    @staticmethod
    def _merge_intervals(intervals):
        """Merge half-open ``[start, end)`` intervals that overlap.

        Empty intervals (``start == end``) are dropped because they
        carry no coverage. Strictly touching half-open intervals such as
        ``[a, b)`` and ``[b, c)`` do not overlap and are kept separate.

        """
        ordered = sorted(
            ((start, end) for start, end in intervals if start < end),
            key=lambda interval: interval[0])
        merged = []
        for start, end in ordered:
            if merged and start < merged[-1][1]:
                if end > merged[-1][1]:
                    merged[-1] = (merged[-1][0], end)
            else:
                merged.append((start, end))
        return merged

    @staticmethod
    def _intersecting_mask(merged_intervals, starts, ends):
        """Boolean mask of examples intersecting any merged interval.

        Both the merged intervals and the example intervals are
        half-open ``[start, end)``. Empty example intervals (start equal
        to end) cover no time and therefore never intersect.

        """
        non_empty = ends > starts
        result = numpy.zeros(starts.shape, dtype=bool)
        if not merged_intervals:
            return result
        for interval_start, interval_end in merged_intervals:
            # Half-open intersection: example [s, e) and interval
            # [a, b) intersect iff s < b and e > a, restricted to
            # non-empty example intervals.
            result |= (non_empty &
                       (starts < interval_end) &
                       (ends > interval_start))
        return result

    def _build_fold(self, validation_indices):
        starts = self._feature_starts
        ends = self._label_ends
        entities = self._entities

        is_validation = numpy.zeros(self.num_examples, dtype=bool)
        is_validation[validation_indices] = True

        # Exclude every example sharing an entity with a validation
        # example, regardless of where it sits in time.
        same_entity = (numpy.in1d(entities, entities[validation_indices]) &
                       ~is_validation)

        # Exclude examples whose half-open coverage interval intersects
        # any validation coverage interval.
        validation_intervals = self._merge_intervals(
            list(zip(starts[validation_indices], ends[validation_indices])))
        interval_overlap = (
            self._intersecting_mask(validation_intervals, starts, ends) &
            ~is_validation)

        # Exclude examples intersecting the embargo window following the
        # latest validation label end: [latest_end, latest_end + embargo).
        if numpy.issubdtype(numpy.asarray(self._embargo).dtype,
                            numpy.timedelta64):
            has_embargo = numpy.asarray(self._embargo) > numpy.timedelta64(0)
        else:
            has_embargo = self._embargo > 0
        if has_embargo:
            latest_end = ends[validation_indices].max()
            embargo_intervals = self._merge_intervals(
                [(latest_end, latest_end + self._embargo)])
            embargo_overlap = (
                self._intersecting_mask(embargo_intervals, starts, ends) &
                ~is_validation)
        else:
            embargo_overlap = numpy.zeros(self.num_examples, dtype=bool)

        excluded = (is_validation | same_entity | interval_overlap |
                    embargo_overlap)
        train_indices = numpy.flatnonzero(~excluded)
        stats = {
            'validation': int(is_validation.sum()),
            'same_entity': int(same_entity.sum()),
            'interval_overlap': int(interval_overlap.sum()),
            'embargo': int(embargo_overlap.sum())}
        return train_indices, stats

    def get_fold_indices(self):
        """Return the example indices of every fold.

        Returns
        -------
        list of tuples
            One ``(train_indices, validation_indices)`` tuple per fold,
            each a sorted 1D :class:`numpy.ndarray` of example indices.

        """
        return [(train_indices.copy(), validation_indices.copy())
                for train_indices, validation_indices in self.folds]

    def get_fold_schemes(self):
        """Yield train and validation schemes ready to drive a stream.

        Yields
        ------
        tuple
            A ``(train_scheme, validation_scheme)`` tuple per fold. Both
            schemes are instances of `scheme_class` and can be passed as
            the ``iteration_scheme`` argument of a
            :class:`~fuel.streams.DataStream`.

        """
        for train_indices, validation_indices in self.folds:
            train_scheme = self.scheme_class(
                list(train_indices), **self.scheme_kwargs)
            validation_scheme = self.scheme_class(
                list(validation_indices), **self.scheme_kwargs)
            yield train_scheme, validation_scheme

    def __iter__(self):
        return self.get_fold_schemes()

    def save_hdf5_splits(self, file_or_path, sources,
                         split_name_format='fold{fold}_{kind}',
                         overwrite=False):
        """Save the fold indices as HDF5 splits following Fuel's layout.

        One index dataset is written per fold and per train/validation
        role. All sources of a given split reference the very same index
        dataset, which guarantees that every source selects the same
        real examples. The resulting file can be opened with
        :class:`~fuel.datasets.hdf5.H5PYDataset` using split names such
        as ``'fold0_train'`` and ``'fold0_valid'``.

        Parameters
        ----------
        file_or_path : :class:`h5py.File` or str
            Open HDF5 file handle or path to an HDF5 file. Every source
            must already exist in the file and have as many rows as
            examples were given to this cross-validation generator.
        sources : iterable of str
            Names of the sources in the HDF5 file for which splits are
            written. All of them use the same per-split index dataset.
        split_name_format : str, optional
            Format of the split names. It must accept ``fold`` and
            ``kind`` keyword fields, the latter being either
            ``'train'`` or ``'valid'``. Defaults to
            ``'fold{fold}_{kind}'``.
        overwrite : bool, optional
            If `True`, replace an existing ``split`` attribute of the
            root group. If `False` (the default), a file that already
            defines splits raises a :class:`ValueError`.

        Returns
        -------
        tuple of str
            The names of the splits that were written, in fold order.

        """
        import h5py
        from fuel.datasets.hdf5 import H5PYDataset

        if isinstance(file_or_path, h5py.File):
            h5file = file_or_path
            opened_here = False
        else:
            h5file = h5py.File(file_or_path, mode='a')
            opened_here = True
        try:
            sources = tuple(sources)
            if not sources:
                raise ValueError('at least one source is required')
            for source in sources:
                if source not in h5file:
                    raise ValueError(
                        "source '{}' does not exist in the HDF5 "
                        "file".format(source))
                if len(h5file[source]) != self.num_examples:
                    raise ValueError(
                        "source '{}' has {} rows but the cross-validation "
                        "was built on {} examples".format(
                            source, len(h5file[source]), self.num_examples))
            if 'split' in h5file.attrs and not overwrite:
                raise ValueError(
                    "the HDF5 file already defines splits; pass "
                    "overwrite=True to replace them")

            split_dict = {}
            split_names = []
            for fold_number, (train_indices, validation_indices) in enumerate(
                    self.folds):
                for kind, indices in (('train', train_indices),
                                      ('valid', validation_indices)):
                    split_name = split_name_format.format(
                        fold=fold_number, kind=kind)
                    index_dataset_name = '{}_indices'.format(split_name)
                    if index_dataset_name in h5file:
                        del h5file[index_dataset_name]
                    h5file[index_dataset_name] = numpy.asarray(
                        indices, dtype=numpy.int64)
                    index_reference = h5file[index_dataset_name].ref
                    # The same reference is reused for every source so
                    # that all sources select the same examples.
                    split_dict[split_name] = {
                        source: (-1, -1, index_reference, '.')
                        for source in sources}
                    split_names.append(split_name)
            h5file.attrs['split'] = H5PYDataset.create_split_array(
                split_dict)
        finally:
            if opened_here:
                h5file.close()
        return tuple(split_names)
