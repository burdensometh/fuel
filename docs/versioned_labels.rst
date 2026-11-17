Versioned label corrections
===========================

When the feature HDF5 file is large but its labels keep being manually
corrected, rewriting the whole file for every correction is impractical.
Fuel provides :class:`~.datasets.versioned_labels.LabelRevisionStore` and
:class:`~.datasets.versioned_labels.VersionedLabelsH5PYDataset` to manage
corrections in a separate *sidecar* HDF5 file while leaving the feature
file untouched.

The design guarantees that:

* the feature HDF5 file (including its original labels) is never modified;
* every correction set is published with a strictly increasing integer
  *version*, and invalid submissions are rejected wholesale;
* all batches of a given epoch see the same label version;
* any historical version can still be read after the files are reopened.

Setting up the store
--------------------

The feature file must contain a dataset of stable IDs, one row per example,
alongside the original label dataset.

>>> import h5py, numpy
>>> f = h5py.File('dataset.hdf5', mode='r+')
>>> num_examples = f['features'].shape[0]
>>> f['ids'] = numpy.array(
...     ['id_{}'.format(i).encode('utf8') for i in range(num_examples)])

The store snapshots the IDs and the original labels (version ``0``) into a
new, separate file:

>>> from fuel.datasets import LabelRevisionStore
>>> store = LabelRevisionStore.create(
...     'labels.hdf5', 'dataset.hdf5',
...     id_source='ids', label_source='targets')

Publishing corrections
----------------------

A correction set pairs stable IDs with their corrected labels. It may be a
*complete* set containing every ID: only the labels that actually change
compared to the previous version are stored, all other IDs inherit their
previous label.

>>> changed_ids = f['ids'][[3, 7]]
>>> changed_labels = numpy.array([[31], [72]], dtype='uint8')
>>> store.publish(changed_ids, changed_labels)
1

A submission is rejected as a whole (nothing is written) when it references
an unknown ID, contains contradictory labels for the same ID within one
revision, has the wrong label shape/dtype, or uses a version number that is
not exactly ``latest_version + 1``.

>>> from fuel.exceptions import LabelRevisionError
>>> try:
...     store.publish(numpy.array([b'unknown_id']),
...                   numpy.array([[1]], dtype='uint8'))
... except LabelRevisionError as error:
...     print(str(error)[:40])
unknown ID(s) are not part of the feature

Reading a fixed version
-----------------------

:meth:`~.datasets.versioned_labels.LabelRevisionStore.resolve` returns the
complete, ID-aligned label array of any published version:

>>> labels_v1 = store.resolve(1)
>>> original_labels = store.resolve(0)

Epoch-pinned dataset
--------------------

:class:`~.datasets.versioned_labels.VersionedLabelsH5PYDataset` behaves like
:class:`~.datasets.hdf5.H5PYDataset`, but overrides the label source with the
store's labels of one pinned revision:

>>> from fuel.datasets import VersionedLabelsH5PYDataset
>>> dataset = VersionedLabelsH5PYDataset(
...     'dataset.hdf5', which_sets=('train',), label_store=store)
>>> state = dataset.open()

Opening the dataset pins the latest completed revision. Calling
:meth:`~.datasets.versioned_labels.VersionedLabelsH5PYDataset.reset` re-reads
that same revision without changing it, so every batch of an epoch uses the
same label version even if new revisions are published in the meantime.
:meth:`~.datasets.versioned_labels.VersionedLabelsH5PYDataset.next_epoch`
advances to the newest revision that has completed by then; revisions
published during an epoch only affect the next one.

Passing ``label_version`` explicitly pins a historical version for the whole
lifetime of the dataset, including across epochs, resets and pickling. This
is how an experiment keeps reading a given old version after the files are
reopened:

>>> dataset = VersionedLabelsH5PYDataset(
...     'dataset.hdf5', which_sets=('train',),
...     label_store='labels.hdf5', label_version=1)

Labels are always selected through the same subset machinery as the
features, which keeps each returned label aligned with the stable ID of its
feature row.
