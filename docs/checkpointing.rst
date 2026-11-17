Checkpointing and resuming a data stream
==========================================

A training script frequently needs to save the model together with the exact
position of the data pipeline, so that after a restart it can continue with
the *next* batch and never skip or duplicate a sample.  Fuel data streams are
picklable, but pickling an epoch iterator does not by itself distinguish the
batches the training loop actually consumed from the batches that were merely
delivered (or read ahead), and some wrappers read far beyond what is consumed.

Fuel therefore exposes an explicit acknowledgement and checkpoint protocol on
:class:`~fuel.streams.AbstractDataStream`.

Acknowledgement
---------------

A batch only counts as processed once it is acknowledged.  A typical training
loop looks like this:

.. code-block:: python

   from fuel.datasets import MNIST
   from fuel.schemes import ShuffledScheme
   from fuel.streams import DataStream
   from fuel.transformers import Batch, Padding

   dataset = MNIST(('train',))
   stream = Padding(DataStream(
       dataset,
       iteration_scheme=ShuffledScheme(dataset.num_examples, batch_size=128,
                                       rng=numpy.random.RandomState(seed)))

   for epoch in stream.iterate_epochs():
       for batch in epoch:
           train_one_step(batch)
           # The batch has been consumed and the model is up to date.
           stream.acknowledge()

Batches that have been delivered but not acknowledged (for instance because
the process crashed between delivery and the update) are considered unsafe to
skip, and are delivered again after a restart.  This gives *at-least-once*
semantics for every batch.

Checkpoints
-----------

:meth:`~fuel.streams.AbstractDataStream.get_checkpoint` returns a plain,
JSON-serializable dictionary.  It records:

* the epoch the next batch belongs to (``epoch``);
* the number of batches acknowledged in that epoch (``acknowledged``);
* the random number generator state the epoch was shuffled from, so that
  replaying the acknowledged prefix reproduces the exact same sample order;
* a fingerprint binding the checkpoint to the data source, the iteration
  scheme and the transformer chain.

.. code-block:: python

   import json

   checkpoint = stream.get_checkpoint()
   with open('fuel_checkpoint.json', 'w') as f:
       json.dump(checkpoint, f)

   # ... later, possibly in a new process ...

   stream = Padding(DataStream(
       dataset,
       iteration_scheme=ShuffledScheme(dataset.num_examples, batch_size=128,
                                       rng=numpy.random.RandomState(seed)))
   with open('fuel_checkpoint.json') as f:
       stream.set_checkpoint(json.load(f))
   for epoch in stream.iterate_epochs():
       ...  # starts at the batch after the last acknowledged one

Restoring a checkpoint is refused with
:class:`~fuel.exceptions.IncompatibleCheckpointError` when the checkpoint does
not describe the current pipeline: a different dataset, split, subset, batch
size, scheme, transformer chain, or configuration value (``default_seed`` /
``floatX``) all trigger the error instead of silently continuing on different
samples.  When the data is stored in an HDF5 file, the checkpoint also binds a
digest of the selected data, so editing the file in place is detected too.

Resumed runs are guaranteed to match an uninterrupted run: sample order,
labels, padding masks and the trailing partial batch are identical, because
the epoch is rebuilt from its start with the recorded random state and the
acknowledged prefix is replayed (and discarded) before new data is returned.

Unsupported components
----------------------

Not every component can describe its *consumed* position.  Wrappers that read
ahead (:class:`~fuel.transformers.Cache`,
:class:`~fuel.transformers.MultiProcessing`), change the number of examples
(:class:`~fuel.transformers.Filter`, :class:`~fuel.transformers.Unpack`),
apply arbitrary callables (:class:`~fuel.transformers.Mapping`) or interleave
streams (:class:`~fuel.transformers.Merge`) set ``checkpoint_supported =
False``.  Asking for a checkpoint of such a pipeline raises
:class:`~fuel.exceptions.CheckpointNotSupportedError` *before* anything is
written, so a training script can never save a model together with a
checkpoint that pretends a read-ahead position is a consumed position.

Custom components can opt in by setting ``checkpoint_supported = True`` if they
are stateless and deterministic, or ``checkpoint_supported = False`` with a
``checkpoint_unsupported_reason`` attribute describing why they cannot be
resumed.

The supported, deterministic chain is ``H5PYDataset`` (or
``IndexableDataset``) -> ``SequentialScheme`` / ``ShuffledScheme`` ->
``Batch`` / ``Padding``, optionally wrapped in stateless sourcewise
transformers such as ``ScaleAndShift``, ``Cast``, ``ForceFloatX``,
``Flatten``, ``Rename`` and ``FilterSources``.
