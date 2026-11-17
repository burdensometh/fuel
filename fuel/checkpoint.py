"""Explicit batch acknowledgement and checkpointing for data pipelines.

Fuel data streams are picklable, which makes it possible to serialize an epoch
iterator and resume it later.  Pickling an iterator, however, does not give a
training loop any control over *which* batches count as processed: an iterator
that already handed a batch to the training loop looks exactly like one that
did not, and a stream that reads far ahead (e.g.
:class:`~fuel.transformers.Cache` or
:class:`~fuel.transformers.MultiProcessing`) has read much more data than was
ever consumed.

This module implements a stricter and explicit notion of progress:

* A batch is only considered processed once it is **acknowledged** with
  :meth:`~fuel.streams.AbstractDataStream.acknowledge`.
* :meth:`~fuel.streams.AbstractDataStream.get_checkpoint` returns a plain,
  JSON-serializable dictionary describing the acknowledged position, the
  current epoch, the random number generator states needed to reproduce the
  exact sample order, and a fingerprint binding the checkpoint to its data
  source and to the shape of the pipeline.
* :meth:`~fuel.streams.AbstractDataStream.set_checkpoint` validates that
  fingerprint and positions the stream so that the first batch returned by
  the next epoch iterator is the batch following the last acknowledged one.
  A batch that was delivered but never acknowledged is therefore delivered
  again (at-least-once semantics).
* Components that cannot faithfully report their *consumed* position raise
  :class:`~fuel.exceptions.CheckpointNotSupportedError` *before* a checkpoint
  is produced, instead of quietly passing a read-ahead position off as the
  consumed position.

Checkpoint format
-----------------
A checkpoint is a dictionary with the following top-level keys:

``fuel_checkpoint_version``
    Version of the checkpoint format (currently ``1``).

``epoch``
    Zero-based index of the epoch the next batch belongs to.

``acknowledged``
    Number of batches acknowledged in that epoch.

``epoch_start_rng_state``
    Encoded state of the iteration scheme's random number generator as it
    was *before* the current epoch was shuffled.  Restoring it reproduces
    that epoch's sample order exactly.

``sources``
    The stream's source names, as a list.

``produces_examples``
    Whether the stream produces examples instead of batches.

``environment``
    Configuration values that influence the pipeline (``default_seed`` and
    ``floatX``).

``node``
    A recursive description of the pipeline.  Each node stores its class and
    a structural ``identity`` (covering the iteration scheme, transformer
    configuration and dataset fingerprint) used to detect configuration
    changes.  Transforming streams store their wrapped stream under
    ``child``.

All values are plain Python/NumPy scalar types so a checkpoint can be saved
as JSON or embedded in a pickled model file.
"""
import hashlib
import json
import numbers
import re

import numpy

from fuel import config

CHECKPOINT_VERSION = 1

#: Matches Python's default ``repr`` memory addresses, which differ between
#: runs and must not take part in checkpoint fingerprints.
_MEMORY_ADDRESS_RE = re.compile(r'0x[0-9a-fA-F]+')

#: Corpora of ``(name, reason)`` for classes whose consumed position cannot
#: be reconstructed.  These are rejected before a checkpoint is written.
UNSUPPORTED_REASONS = {
    'Cache': 'it reads ahead into an internal cache, so its read position is '
             'not its consumed position',
    'MultiProcessing': 'it prefetches batches in a background process, which '
                       'is not acknowledged in lockstep with training',
    'BackgroundProcess': 'it reads batches ahead in a separate process',
    'Unpack': 'it buffers the current batch and hands out single examples, so '
              'its consumed position cannot be recovered from its read '
              'position',
    'Filter': 'it silently drops examples according to a predicate, so the '
              'number of consumed examples is not known in advance',
    'Merge': 'it interleaves several data streams whose individual positions '
             'are not tracked',
    'Mapping': 'an arbitrary Python callable is applied to the data and Fuel '
               'cannot know whether it is stateful',
    'ServerDataStream': 'data is pushed by a separate server process and the '
                        'client cannot know which batches were buffered',
    'ConcatenatedScheme': 'its sub-schemes are concatenated lazily and their '
                          'individual positions are not tracked',
    'PytablesDataset': 'its stateful node objects cannot be fingerprinted',
}

#: Classes that are known to be deterministic and safe to checkpoint.  This
#: is a fallback for classes that do not set ``checkpoint_supported``.
SAFE_COMPONENTS = frozenset([
    'DataStream', 'Batch', 'Padding', 'ScaleAndShift', 'Cast', 'ForceFloatX',
    'Flatten', 'Rename', 'FilterSources', 'ImagesFromBytes',
    'MinimumImageDimensions', 'SequentialScheme', 'ShuffledScheme',
    'ConstantScheme', 'SequentialExampleScheme', 'ShuffledExampleScheme',
])

#: Dataset classes whose subclasses are safe to checkpoint.
SAFE_DATASETS = frozenset(['H5PYDataset', 'IndexableDataset'])
def unsupported_message(name, detail=None):
    """Return a clear explanation for one unsupported component."""
    reason = detail or UNSUPPORTED_REASONS.get(
        name, 'it does not declare how to reconstruct its consumed position')
    return ("'{}' cannot be checkpointed: {}. Only deterministic pipelines "
            "built from H5PYDataset, SequentialScheme/ShuffledScheme, Batch "
            "and Padding (plus stateless transformers that opt in) are "
            "supported.").format(name, reason)


def format_unsupported(components):
    """Format a list of ``(name, detail)`` pairs into one error message."""
    return '\n'.join(unsupported_message(name, detail)
                     for name, detail in components)


def _plain(value):
    """Recursively convert numpy scalars/containers to plain Python values."""
    if isinstance(value, numpy.generic):
        return value.item()
    if isinstance(value, numpy.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return dict((k, _plain(v)) for k, v in value.items())
    return value


def _identity_value(value):
    """Convert a configuration value into a stable, plain description."""
    if isinstance(value, bool):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value)
    if isinstance(value, str):
        return value
    if value is None:
        return None
    if isinstance(value, numpy.generic):
        return _identity_value(value.item())
    if isinstance(value, numpy.ndarray):
        return {'array': digest_array(value),
                'shape': list(value.shape),
                'dtype': str(value.dtype)}
    if isinstance(value, range):
        return {'range': [value.start, value.stop, value.step]}
    if isinstance(value, slice):
        return {'slice': [value.start, value.stop, value.step]}
    if isinstance(value, (list, tuple)):
        if len(value) > 256:
            return {'sequence': digest_repr(value), 'length': len(value)}
        return [_identity_value(v) for v in value]
    if isinstance(value, dict):
        return dict((str(k), _identity_value(v))
                    for k, v in sorted(value.items(),
                                       key=lambda item: str(item[0])))
    if callable(value):
        return {'callable': '{}.{}'.format(
            getattr(value, '__module__', None),
            getattr(value, '__qualname__', getattr(value, '__name__', None)))}
    return {'repr': _MEMORY_ADDRESS_RE.sub('0xADDR', repr(value))}


def encode_rng_state(state):
    """Encode a :meth:`numpy.random.RandomState.get_state` tuple as plain data.

    Parameters
    ----------
    state : tuple or None
        The state returned by ``RandomState.get_state()``.

    Returns
    -------
    dict or None
        A JSON-serializable description of the state.

    """
    if state is None:
        return None
    kind, keys, pos, has_gauss, cached_gaussian = state
    return {
        'kind': str(kind),
        'keys': numpy.asarray(keys, dtype=numpy.uint32).tolist(),
        'pos': int(pos),
        'has_gauss': int(has_gauss),
        'cached_gaussian': float(cached_gaussian),
    }


def decode_rng_state(encoded):
    """Decode a state produced by :func:`encode_rng_state`."""
    if encoded is None:
        return None
    return (encoded['kind'],
            numpy.asarray(encoded['keys'], dtype=numpy.uint32),
            int(encoded['pos']),
            int(encoded['has_gauss']),
            float(encoded['cached_gaussian']))


def get_rng_state(rng):
    """Return an encoded RNG state, or ``None`` if there is no RNG."""
    if rng is None:
        return None
    return encode_rng_state(rng.get_state())


def set_rng_state(rng, encoded):
    """Restore an encoded RNG state (no-op when either side is ``None``)."""
    if rng is None or encoded is None:
        return
    rng.set_state(decode_rng_state(encoded))


def digest_repr(value):
    """Return a stable hexadecimal digest for a plain Python value."""
    hasher = hashlib.sha256()
    hasher.update(repr(_plain(value)).encode('utf8'))
    return hasher.hexdigest()


def digest_array(array_like):
    """Return a stable digest of a numpy array's dtype, shape and contents."""
    array = numpy.ascontiguousarray(array_like)
    if array.dtype == numpy.object_:
        return digest_repr([numpy.asarray(item).tolist() for item in array])
    hasher = hashlib.sha256()
    hasher.update(str(array.dtype).encode('utf8'))
    hasher.update(repr(array.shape).encode('utf8'))
    hasher.update(array.tobytes())
    return hasher.hexdigest()


def environment_identity():
    """Return configuration values that influence data ordering or layout."""
    try:
        floatx = str(config.floatX)
    except Exception:  # floatX has no default and may be unset
        floatx = None
    return {
        'default_seed': int(config.default_seed),
        'floatX': floatx,
    }


def scheme_identity(scheme):
    """Return a stable description of an iteration scheme's configuration."""
    if scheme is None:
        return None
    config_identity = {}
    for key, value in vars(scheme).items():
        if key.startswith('_') or key == 'rng':
            continue
        config_identity[key] = _identity_value(value)
    return {
        'class': type(scheme).__name__,
        'requests_examples': bool(scheme.requests_examples),
        'config': config_identity,
    }


#: Public attributes that are runtime state rather than configuration.
RUNTIME_ATTRIBUTES = frozenset([
    'data_stream', 'dataset', 'data_state', 'iteration_scheme',
    'child_epoch_iterator', 'child_epoch_iterators', 'cache', 'data',
    'background', 'proc', 'socket', 'batches', 'axis_labels',
    'external_file_handle', 'nodes', 'h5file', 'data_sources',
    'source_shapes', 'in_memory_subset', 'subsets', 'sources',
    'provides_sources', 'rng',
])


def config_identity(component):
    """Describe the public, non-runtime configuration of a component.

    Values that are streams, datasets or iteration schemes are handled
    separately by the checkpoint machinery, and callables are identified by
    their qualified name so that the checkpoint stays JSON-serializable.

    """
    identity = {}
    for key, value in vars(component).items():
        if key.startswith('_') or key in RUNTIME_ATTRIBUTES:
            continue
        if callable(value):
            continue
        identity[key] = _identity_value(value)
    return identity


def dataset_identity(dataset):
    """Return a stable description of a dataset, including its input.

    The result binds a checkpoint to the exact data source: the file (if
    any), the requested splits, the selection within them, and a digest of
    the selected data.  If the input changes, restoring the checkpoint is
    refused instead of silently continuing on different samples.

    """
    identity = {'class': type(dataset).__name__}
    for attr in ('path', 'which_sets', 'load_in_memory', 'driver',
                 'sort_indices', 'start', 'stop', 'data_node',
                 'sources_in_file'):
        if hasattr(dataset, attr):
            identity[attr] = _identity_value(getattr(dataset, attr))
    if getattr(dataset, 'user_given_subset', None) is not None:
        identity['subset'] = _identity_value(dataset.user_given_subset)
    try:
        identity['num_examples'] = int(dataset.num_examples)
    except Exception:
        pass
    try:
        identity['provides_sources'] = list(dataset.provides_sources)
    except Exception:
        pass
    try:
        identity['sources'] = list(dataset.sources)
    except Exception:
        pass
    digest = getattr(dataset, 'input_digest', None)
    if callable(digest):
        identity['input'] = digest()
    return identity


def dataset_identity(dataset):
    """Return a stable description of a dataset, including its input.

    The result binds a checkpoint to the exact data source: the file (if
    any), the requested splits, the selection within them, and a digest of
    the selected data.  If the input changes, restoring the checkpoint is
    refused instead of silently continuing on different samples.

    """
    identity = {'class': type(dataset).__name__}
    for attr in ('path', 'which_sets', 'load_in_memory', 'driver',
                 'sort_indices', 'start', 'stop', 'data_node',
                 'sources_in_file'):
        if hasattr(dataset, attr):
            identity[attr] = _identity_value(getattr(dataset, attr))
    if getattr(dataset, 'user_given_subset', None) is not None:
        identity['subset'] = _identity_value(dataset.user_given_subset)
    try:
        identity['num_examples'] = int(dataset.num_examples)
    except Exception:
        pass
    try:
        identity['provides_sources'] = list(dataset.provides_sources)
    except Exception:
        pass
    try:
        identity['sources'] = list(dataset.sources)
    except Exception:
        pass
    digest = getattr(dataset, 'input_digest', None)
    if callable(digest):
        identity['input'] = digest()
    return identity


def fingerprint(node):
    """Return the structural part of a node snapshot.

    Position and RNG state are deliberately excluded: they change between
    checkpoints of the same pipeline and must not cause a mismatch.

    Parameters
    ----------
    node : dict
        A node snapshot as produced by
        :meth:`~fuel.streams.AbstractDataStream._checkpoint_snapshot`.

    """
    if not isinstance(node, dict):
        return node
    result = {
        'class': node.get('class'),
        'identity': node.get('identity'),
    }
    if 'child' in node:
        result['child'] = fingerprint(node['child'])
    return result


def fingerprint_diff(saved, current, path=''):
    """Collect human-readable differences between two checkpoint identities.

    Parameters
    ----------
    saved, current : object
        Values from the checkpoint and from the live pipeline.
    path : str, optional
        Prefix describing the location of a difference.

    Returns
    -------
    list of str
        One line per difference.  An empty list means the values match.

    """
    differences = []
    if isinstance(saved, dict) and isinstance(current, dict):
        for key in sorted(set(saved) | set(current)):
            child_path = '{}.{}'.format(path, key) if path else str(key)
            if key not in saved:
                differences.append('{}: missing from checkpoint'.format(
                    child_path))
            elif key not in current:
                differences.append('{}: missing from current pipeline'.format(
                    child_path))
            else:
                differences.extend(
                    fingerprint_diff(saved[key], current[key], child_path))
        return differences
    if isinstance(saved, (list, tuple)) and isinstance(current, (list, tuple)):
        if len(saved) != len(current):
            differences.append('{}: {!r} != {!r}'.format(path, saved, current))
            return differences
        for i, (saved_item, current_item) in enumerate(zip(saved, current)):
            differences.extend(fingerprint_diff(
                saved_item, current_item, '{}[{}]'.format(path, i)))
        return differences
    if _plain(saved) != _plain(current):
        differences.append('{}: {!r} != {!r}'.format(path, saved, current))
    return differences


def save_checkpoint(checkpoint, file_or_path):
    """Save a checkpoint dictionary as JSON.

    Parameters
    ----------
    checkpoint : dict
        A checkpoint produced by ``stream.get_checkpoint()``.
    file_or_path : str or file-like
        Destination path or writable file object.

    """
    if hasattr(file_or_path, 'write'):
        json.dump(checkpoint, file_or_path)
    else:
        with open(file_or_path, 'w') as f:
            json.dump(checkpoint, f)


def load_checkpoint(file_or_path):
    """Load a checkpoint previously written by :func:`save_checkpoint`."""
    if hasattr(file_or_path, 'read'):
        return json.load(file_or_path)
    with open(file_or_path, 'r') as f:
        return json.load(f)
