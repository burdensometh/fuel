"""Versioned, ID-addressed label corrections for HDF5 datasets.

This module addresses the situation where the feature HDF5 file is large
and essentially immutable, while its labels keep being manually corrected.
Rewriting the whole feature file for every correction is not an option, and
the labels seen by all batches of a given epoch must come from the same
label version.

Two objects are provided:

* :class:`LabelRevisionStore` manages a *sidecar* HDF5 file that holds
  complete, versioned correction sets. Every revision is published with a
  strictly increasing integer version. A revision only needs to contain the
  IDs whose labels changed; untouched IDs inherit their label from the
  previous version (the original HDF5 label being version ``0``). Invalid
  revisions (unknown IDs, contradictory labels within the same revision,
  non-incremental version numbers) are rejected wholesale, so a failed
  publication never leaves a partial version behind.
* :class:`VersionedLabelsH5PYDataset` is an :class:`.H5PYDataset` that
  reads its features from the original, untouched HDF5 file and overrides a
  label source with the labels of a fixed revision. Each epoch pins one
  completed version; revisions published while an epoch is in progress only
  take effect at the next epoch. Resetting the stream re-reads the current
  epoch's version, it never changes the pinned version.

The original feature HDF5 file is never modified, and every historical
revision is kept, so any past version can be re-read after the files are
closed and reopened.

"""
import contextlib
import os

import h5py
import numpy

from fuel.datasets.hdf5 import H5PYDataset
from fuel.exceptions import LabelRevisionError


def _normalize_id(identifier):
    """Returns a hashable, comparable form of a stable ID.

    HDF5 stores text as fixed-width byte strings, so byte strings and
    unicode strings carrying the same text are treated as identical.

    """
    if isinstance(identifier, bytes):
        return identifier.decode('utf8')
    return identifier


@contextlib.contextmanager
def _exclusive_file_lock(lock_path):
    """Locks ``lock_path`` for the duration of the context.

    This serializes concurrent publications against the same sidecar file.
    Locking is advisory and best-effort on platforms without
    :mod:`fcntl`; on those platforms the context is a no-op.

    """
    try:
        import fcntl
    except ImportError:
        yield
        return
    with open(lock_path, 'w') as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


class LabelRevisionStore(object):
    """A sidecar HDF5 file holding versioned label corrections.

    The sidecar file is fully self-contained: it stores a snapshot of the
    original labels and of the stable IDs taken at creation time (version
    ``0``), together with every published revision. Revisions only store the
    IDs that actually changed; labels of all other IDs are carried over from
    the previous version.

    The file layout is::

        /
        |- attrs: id_source, label_source, latest_version
        |- base
        |  |- ids      : the full, ordered array of stable IDs
        |  |- labels   : the original labels, row-aligned with `ids`
        |- versions
           |- 1
           |  |- ids    : IDs whose label changed in revision 1
           |  |- labels : the new labels for those IDs
           |- 2 ...

    Parameters
    ----------
    path : str
        Path to the sidecar HDF5 file. The file must have been created with
        :meth:`create` beforehand.

    """
    VERSIONS_GROUP = 'versions'
    PENDING_GROUP = 'pending'
    BASE_GROUP = 'base'

    def __init__(self, path):
        self.path = path
        if not os.path.isfile(path):
            raise IOError(
                "label revision file '{}' does not exist; create it with "
                "LabelRevisionStore.create first".format(path))

    @classmethod
    def create(cls, path, feature_file_or_path, id_source, label_source):
        """Creates a sidecar file from the original labels of a feature file.

        The feature file is opened read-only and never modified. A snapshot
        of its stable IDs and original labels is copied into the sidecar.

        Parameters
        ----------
        path : str
            Path of the sidecar HDF5 file to create. It must not already
            exist.
        feature_file_or_path : :class:`h5py.File` or str
            The original feature HDF5 file, or a path to it.
        id_source : str
            Name of the dataset in the feature file containing the stable
            IDs, one row per example.
        label_source : str
            Name of the dataset in the feature file containing the original
            labels, one row per example.

        Returns
        -------
        :class:`LabelRevisionStore`
            A store opened on the newly created sidecar file.

        """
        if os.path.exists(path):
            raise IOError(
                "label revision file '{}' already exists".format(path))
        external_handle = isinstance(feature_file_or_path, h5py.File)
        if external_handle:
            handle = feature_file_or_path
            should_close = False
        else:
            handle = h5py.File(feature_file_or_path, mode='r')
            should_close = True
        try:
            if id_source not in handle:
                raise LabelRevisionError(
                    "ID source '{}' is missing from the feature "
                    "file".format(id_source))
            if label_source not in handle:
                raise LabelRevisionError(
                    "label source '{}' is missing from the feature "
                    "file".format(label_source))
            ids = handle[id_source][...]
            labels = handle[label_source][...]
            if len(ids) != len(labels):
                raise LabelRevisionError(
                    "ID source '{}' ({} rows) and label source '{}' ({} "
                    "rows) have different lengths".format(
                        id_source, len(ids), label_source, len(labels)))
            # The stable IDs must be unique, otherwise labels cannot be
            # addressed unambiguously.
            normalized = numpy.array(
                [_normalize_id(value) for value in ids.tolist()],
                dtype=object)
            if len(numpy.unique(normalized)) != len(normalized):
                raise LabelRevisionError(
                    "ID source '{}' contains duplicate IDs".format(id_source))
        finally:
            if should_close:
                handle.close()

        sidecar = h5py.File(path, mode='w')
        try:
            base = sidecar.create_group(cls.BASE_GROUP)
            base.create_dataset('ids', data=ids)
            base.create_dataset('labels', data=labels)
            sidecar.create_group(cls.VERSIONS_GROUP)
            sidecar.attrs['id_source'] = id_source
            sidecar.attrs['label_source'] = label_source
            sidecar.attrs['latest_version'] = 0
            sidecar.flush()
        finally:
            sidecar.close()
        return cls(path)

    # -- introspection ---------------------------------------------------

    @contextlib.contextmanager
    def _open(self):
        handle = h5py.File(self.path, mode='r')
        try:
            yield handle
        finally:
            handle.close()

    @property
    def id_source(self):
        """Name of the stable ID dataset in the feature file."""
        with self._open() as handle:
            return self._read_attr(handle, 'id_source')

    @property
    def label_source(self):
        """Name of the label dataset being overridden."""
        with self._open() as handle:
            return self._read_attr(handle, 'label_source')

    @staticmethod
    def _read_attr(handle, name):
        value = handle.attrs[name]
        if isinstance(value, bytes):
            value = value.decode('utf8')
        return value

    @property
    def latest_version(self):
        """Version number of the latest completed revision (0 at creation)."""
        with self._open() as handle:
            return int(handle.attrs['latest_version'])

    @property
    def versions(self):
        """Tuple of all completed revision version numbers, sorted."""
        with self._open() as handle:
            return tuple(sorted(
                int(name) for name in
                handle[self.VERSIONS_GROUP].keys()))

    def base_ids(self):
        """Returns a copy of the full, ordered array of stable IDs."""
        with self._open() as handle:
            return handle[self.BASE_GROUP]['ids'][...]

    def base_labels(self):
        """Returns a copy of the original (version 0) label array."""
        with self._open() as handle:
            return handle[self.BASE_GROUP]['labels'][...]

    # -- publication -----------------------------------------------------

    def publish(self, ids, labels, version=None):
        """Publishes a complete correction set as a new revision.

        The correction set is a *complete* view of the corrected labels: it
        may contain every stable ID, but only the IDs whose label actually
        changes compared to the previous version are stored. IDs that are
        absent from ``ids`` keep their previous label (which is the original
        HDF5 label for version ``1``).

        The submission is validated before anything is written; on any
        validation error the whole submission is rejected and the sidecar
        file is left untouched.

        Parameters
        ----------
        ids : array-like
            Stable IDs whose labels are (potentially) corrected in this
            revision. One entry per corrected example.
        labels : numpy.ndarray
            The corrected labels, with as many rows as ``ids``. Rows are
            paired with IDs positionally, and their trailing shape and dtype
            must match the original labels.
        version : int, optional
            The version number to publish. It must be exactly
            ``latest_version + 1``. If omitted, the next version number is
            used automatically.

        Returns
        -------
        int
            The version number of the newly published revision.

        Raises
        ------
        :class:`.LabelRevisionError`
            If the revision is invalid (unknown or duplicated IDs,
            contradictory labels within the revision, bad label shape or
            dtype, non-incremental version number, or an empty/no-op
            revision).

        """
        ids = numpy.asarray(ids)
        labels = numpy.asarray(labels)
        lock_path = self.path + '.lock'
        with _exclusive_file_lock(lock_path):
            handle = h5py.File(self.path, mode='r+')
            try:
                latest = int(handle.attrs['latest_version'])
                if version is None:
                    version = latest + 1
                elif not isinstance(version, (int, numpy.integer)):
                    raise LabelRevisionError(
                        "revision version must be an integer, got "
                        "{!r}".format(version))
                if version != latest + 1:
                    raise LabelRevisionError(
                        "expected revision version {} (the latest published "
                        "version is {}), got {}".format(
                            latest + 1, latest, version))

                base_ids = handle[self.BASE_GROUP]['ids'][...]
                base_labels = handle[self.BASE_GROUP]['labels']
                ids, labels = self._validate_submission(
                    ids, labels, base_ids, base_labels)

                # Build the full effective labels of this revision, then
                # keep only the rows that actually changed.
                effective_labels, index_of_id = self._resolve(
                    handle, latest)
                normalized = [
                    _normalize_id(value) for value in ids.tolist()]
                changed_ids = []
                changed_labels = []
                for raw_id, new_label in zip(normalized, labels):
                    position = index_of_id[raw_id]
                    old_label = effective_labels[position]
                    if not self._labels_equal(old_label, new_label):
                        changed_ids.append(base_ids[position])
                        changed_labels.append(new_label)
                if not changed_ids:
                    raise LabelRevisionError(
                        "revision {} does not modify any label; empty "
                        "revisions are not allowed".format(version))
                changed_labels = numpy.array(
                    changed_labels, dtype=base_labels.dtype)

                # Write into a temporary group first and only commit it by
                # renaming, so a failure cannot leave a partial version.
                if self.PENDING_GROUP in handle:
                    del handle[self.PENDING_GROUP]
                pending = handle.create_group(self.PENDING_GROUP)
                pending.create_dataset(
                    'ids', data=numpy.array(changed_ids,
                                            dtype=base_ids.dtype))
                pending.create_dataset('labels', data=changed_labels)
                handle.flush()
                handle.move(self.PENDING_GROUP,
                            '{}/{}'.format(self.VERSIONS_GROUP, version))
                handle.attrs['latest_version'] = version
                handle.flush()
            except Exception:
                # Roll back anything a failed write may have left behind.
                if self.PENDING_GROUP in handle:
                    del handle[self.PENDING_GROUP]
                handle.flush()
                raise
            finally:
                handle.close()
        return version

    @staticmethod
    def _validate_submission(ids, labels, base_ids, base_labels):
        """Validates a correction set and returns its deduplicated form.

        Returns
        -------
        tuple of numpy.ndarray
            The input IDs and labels with redundant occurrences of the same
            ID removed (redundant occurrences are only legal when they carry
            identical labels).

        """
        if ids.ndim != 1:
            raise LabelRevisionError(
                "the ID array must be one-dimensional, got shape "
                "{}".format(ids.shape))
        if len(ids) == 0:
            raise LabelRevisionError(
                "a revision must contain at least one ID")
        if len(ids) != len(labels):
            raise LabelRevisionError(
                "got {} IDs but {} label rows; they must have the same "
                "length".format(len(ids), len(labels)))
        if labels.dtype != base_labels.dtype:
            raise LabelRevisionError(
                "label dtype {} does not match the original label dtype "
                "{}".format(labels.dtype, base_labels.dtype))
        expected_shape = base_labels.shape[1:]
        if labels.shape[1:] != expected_shape:
            raise LabelRevisionError(
                "label row shape {} does not match the original label row "
                "shape {}".format(labels.shape[1:], expected_shape))

        normalized = [_normalize_id(value) for value in ids.tolist()]
        # Same-version contradictions: an ID may only appear more than once
        # if every occurrence carries the exact same label. Valid repeats
        # are then collapsed to a single occurrence.
        first_position = {}
        keep_positions = []
        for position, identifier in enumerate(normalized):
            if identifier in first_position:
                previous = first_position[identifier]
                if not LabelRevisionStore._labels_equal(
                        labels[previous], labels[position]):
                    raise LabelRevisionError(
                        "contradictory labels for ID {!r} within the same "
                        "revision".format(identifier))
            else:
                first_position[identifier] = position
                keep_positions.append(position)

        known = set(
            _normalize_id(value) for value in base_ids.tolist())
        unknown = [identifier for identifier in normalized
                   if identifier not in known]
        if unknown:
            raise LabelRevisionError(
                "{} unknown ID(s) are not part of the feature file's "
                "stable IDs; first unknown IDs: {}".format(
                    len(unknown), unknown[:5]))

        keep_positions = numpy.array(keep_positions)
        return ids[keep_positions], labels[keep_positions]

    @staticmethod
    def _labels_equal(left, right):
        left = numpy.asarray(left)
        right = numpy.asarray(right)
        if left.shape != right.shape:
            return False
        return bool(numpy.array_equal(left, right))

    #-- resolution ------------------------------------------------------

    def resolve(self, version=None):
        """Returns the complete label array effective at a given version.

        Labels are aligned with the base stable IDs, i.e. row ``i`` of the
        returned array is the label of the ID at row ``i`` of
        :meth:`base_ids`. This is the same row ordering as in the original
        feature file.

        Parameters
        ----------
        version : int, optional
            Revision version to resolve. ``0`` returns the original labels.
            If omitted, the latest published version is used.

        Returns
        -------
        numpy.ndarray
            A copy of the full label array at that version.

        """
        with self._open() as handle:
            labels, _ = self._resolve(handle, version)
            return labels

    def _resolve(self, handle, version):
        latest = int(handle.attrs['latest_version'])
        if version is None:
            version = latest
        if not isinstance(version, (int, numpy.integer)):
            raise LabelRevisionError(
                "version must be an integer, got {!r}".format(version))
        if version < 0 or version > latest:
            raise LabelRevisionError(
                "version {} is not available; published versions are 0 "
                "through {}".format(version, latest))
        base_ids = handle[self.BASE_GROUP]['ids'][...]
        labels = handle[self.BASE_GROUP]['labels'][...]
        index_of_id = {
            _normalize_id(identifier): position
            for position, identifier in enumerate(base_ids.tolist())}
        for revision in range(1, version + 1):
            group = handle['{}/{}'.format(self.VERSIONS_GROUP, revision)]
            revision_ids = group['ids'][...]
            revision_labels = group['labels'][...]
            for identifier, new_label in zip(revision_ids, revision_labels):
                labels[index_of_id[_normalize_id(identifier)]] = new_label
        return labels, index_of_id


class VersionedLabelsH5PYDataset(H5PYDataset):
    """An :class:`.H5PYDataset` whose label source is versioned.

    Features are read from the original, untouched feature HDF5 file, while
    one label source is overridden by labels stored in a
    :class:`LabelRevisionStore` sidecar file. The two files are aligned on
    the stable IDs of the feature file, so the label returned for a request
    is always the label of the feature row's own ID.

    Epoch/version semantics:

    * Opening the dataset pins one completed revision: ``label_version`` if
      given, otherwise the latest revision available at that moment.
    * :meth:`reset` re-reads the pinned revision; it never changes it. All
      batches of an epoch therefore see the same label version, even if new
      revisions are published in the meantime.
    * :meth:`next_epoch` advances to the latest revision that has completed
      by then (when no explicit version was pinned); revisions published
      during an epoch only affect the next epoch.
    * An explicitly pinned version is kept for the whole lifetime of the
      dataset, including across epochs, resets and pickling. This is what
      makes an experiment reproducible after the files are reopened.

    Parameters
    ----------
    file_or_path : :class:`h5py.File` or str
        The original feature HDF5 file, as for :class:`.H5PYDataset`.
    which_sets : iterable of str
        Which split(s) to use, as for :class:`.H5PYDataset`.
    label_store : :class:`LabelRevisionStore` or str
        The sidecar store, or a path to the sidecar file.
    id_source : str, optional
        Name of the stable ID dataset in the feature file. Defaults to the
        ``id_source`` recorded in the sidecar.
    label_source : str, optional
        Name of the source to override with corrected labels. It must be a
        source provided by the feature file. Defaults to the
        ``label_source`` recorded in the sidecar.
    label_version : int, optional
        Revision to pin. If omitted, the latest revision at open time is
        pinned and :meth:`next_epoch` may advance to newer revisions.
    **kwargs
        Remaining keyword arguments are forwarded to
        :class:`.H5PYDataset`.

    """
    def __init__(self, file_or_path, which_sets, label_store,
                 id_source=None, label_source=None, label_version=None,
                 **kwargs):
        if isinstance(label_store, LabelRevisionStore):
            self.label_store = label_store
        else:
            self.label_store = LabelRevisionStore(label_store)
        self.id_source = id_source or self.label_store.id_source
        self.label_source = label_source or self.label_store.label_source
        self.explicit_label_version = label_version
        self._pinned_version = None
        super(VersionedLabelsH5PYDataset, self).__init__(
            file_or_path, which_sets, **kwargs)
        if self.label_source not in self.provides_sources:
            raise ValueError(
                "label source '{}' is not provided by the feature file for "
                "split(s) {}".format(self.label_source, which_sets))
        self._check_id_alignment()

    def _check_id_alignment(self):
        """Verifies the sidecar still matches the feature file's IDs."""
        if self.external_file_handle:
            handle = self.external_file_handle
            feature_ids = handle[self.id_source][...]
        else:
            with h5py.File(self.path, mode='r', driver=self.driver) as handle:
                if self.id_source not in handle:
                    raise LabelRevisionError(
                        "ID source '{}' is missing from the feature "
                        "file".format(self.id_source))
                feature_ids = handle[self.id_source][...]
        store_ids = self.label_store.base_ids()
        if len(feature_ids) != len(store_ids):
            raise LabelRevisionError(
                "the feature file's ID source '{}' has {} rows but the "
                "label store has {}".format(
                    self.id_source, len(feature_ids), len(store_ids)))
        feature_normalized = [
            _normalize_id(value) for value in feature_ids.tolist()]
        store_normalized = [
            _normalize_id(value) for value in store_ids.tolist()]
        if feature_normalized != store_normalized:
            raise LabelRevisionError(
                "the feature file's stable IDs do not match the IDs "
                "snapshotted in the label store; labels cannot be aligned "
                "with the features")

    @property
    def label_version(self):
        """The revision currently pinned for the ongoing epoch."""
        return self._pinned_version

    def _version_to_pin(self):
        if self.explicit_label_version is not None:
            return self.explicit_label_version
        return self.label_store.latest_version

    def _read_pinned_labels(self):
        labels = self.label_store.resolve(self._pinned_version)
        return self._pinned_version, labels

    def open(self):
        """Pins a completed revision and returns the epoch state.

        The state carries the pinned version number and the full,
        ID-aligned label array at that version.

        """
        super(VersionedLabelsH5PYDataset, self).open()
        self._pinned_version = self._version_to_pin()
        version, labels = self._read_pinned_labels()
        return _LabelEpochState(version=version, labels=labels)

    def reset(self, state):
        """Re-reads the current epoch's pinned revision without changing it."""
        # Do not call the inherited `reset`, which would call our own
        # `open` and re-pin the latest revision. Re-open the underlying
        # file explicitly instead, keeping the pinned version untouched.
        H5PYDataset.close(self, state)
        H5PYDataset.open(self)
        version, labels = self._read_pinned_labels()
        if version != self._pinned_version:
            # Defensive: reset must never silently switch versions.
            raise LabelRevisionError(
                "reset would change the label version from {} to {}; the "
                "pinned version must remain stable within an epoch".format(
                    self._pinned_version, version))
        return _LabelEpochState(version=version, labels=labels)

    def next_epoch(self, state):
        """Advances to the latest revision completed by now.

        With an explicitly pinned version the pin is kept; otherwise the
        newest published revision becomes the version of the next epoch.

        """
        # Same reason as in `reset`: bypass our own `open` override.
        H5PYDataset.close(self, state)
        H5PYDataset.open(self)
        if self.explicit_label_version is None:
            self._pinned_version = self.label_store.latest_version
        labels = self.label_store.resolve(self._pinned_version)
        return _LabelEpochState(
            version=self._pinned_version, labels=labels)

    def get_data(self, state=None, request=None):
        """Returns feature data together with the pinned-version labels.

        The label rows are selected with the exact same
        :class:`.Subset` machinery as every other source, which guarantees
        they stay aligned with the feature rows of the request.

        """
        if state is None:
            state = self.open()
        if self.load_in_memory:
            # The in-memory code path of H5PYDataset does not use the
            # state, so fetch every source through it directly.
            data = list(super(VersionedLabelsH5PYDataset, self).get_data(
                state=None, request=request))
        else:
            data = list(super(VersionedLabelsH5PYDataset, self).get_data(
                state=state, request=request))
        # Both `self.subsets` and the returned data tuple are aligned with
        # `self.sources`, so the same position indexes both.
        source_position = self.sources.index(self.label_source)
        data[source_position] = self.subsets[source_position]\
            .index_within_subset(
                state.labels, request,
                sort_indices=self.sort_indices and not self.load_in_memory)
        return tuple(data)


class _LabelEpochState(object):
    """Epoch state: the pinned version and its ID-aligned label array."""
    def __init__(self, version, labels):
        self.version = version
        self.labels = labels

    def __repr__(self):
        return '_LabelEpochState(version={})'.format(self.version)
