from abc import ABCMeta, abstractmethod

import zmq
from six import add_metaclass, iteritems

from fuel.checkpoint import (CHECKPOINT_VERSION, SAFE_COMPONENTS,
                             SAFE_DATASETS, UNSUPPORTED_REASONS,
                             config_identity, dataset_identity,
                             environment_identity, fingerprint,
                             fingerprint_diff, format_unsupported,
                             get_rng_state, scheme_identity, set_rng_state)
from fuel.exceptions import (CheckpointNotSupportedError,
                             IncompatibleCheckpointError)
from fuel.iterator import DataIterator
from fuel.server import recv_arrays


def _component_supported(component):
    """Return ``(is_supported, reason)`` for a pipeline component."""
    cls = type(component)
    declared = getattr(cls, 'checkpoint_supported', None)
    if declared is True:
        return True, None
    if declared is False:
        reason = getattr(cls, 'checkpoint_unsupported_reason', None)
        return False, reason or UNSUPPORTED_REASONS.get(cls.__name__)
    name = cls.__name__
    if name in SAFE_COMPONENTS:
        return True, None
    if any(base.__name__ in SAFE_DATASETS for base in cls.__mro__):
        return True, None
    return False, UNSUPPORTED_REASONS.get(name)


def _stream_chain(stream):
    """Return the stream followed by each wrapped stream."""
    chain = [stream]
    while True:
        child = getattr(chain[-1], 'data_stream', None)
        if isinstance(child, AbstractDataStream):
            chain.append(child)
        else:
            return chain


def collect_unsupported(stream):
    """Return ``(name, reason)`` pairs for components that cannot resume."""
    problems = []

    def check(component):
        supported, reason = _component_supported(component)
        if not supported:
            problems.append((type(component).__name__, reason))

    for node in _stream_chain(stream):
        check(node)
        scheme = getattr(node, 'iteration_scheme', None)
        if scheme is not None:
            check(scheme)
        dataset = getattr(node, 'dataset', None)
        if dataset is not None:
            check(dataset)
    return problems


class CheckpointState(object):
    """The acknowledged progress shared by every stream in a pipeline.

    A single instance is owned by the innermost :class:`DataStream` and
    shared with every transformer wrapping it, so that one acknowledgement
    covers the whole chain.

    Attributes
    ----------
    epoch : int
        Zero-based index of the epoch the next batch belongs to.
    delivered : int
        Number of batches delivered in the current epoch.
    acknowledged : int
        Number of batches acknowledged in the current epoch.  This is the
        position checkpoints record; anything delivered but not
        acknowledged is delivered again after a restore.
    current_epoch_start_rng : dict or None
        Encoded state of the iteration scheme's RNG immediately before the
        current epoch was shuffled.  Restoring it reproduces the order of
        that epoch exactly.
    pending_resume : bool
        Whether the next epoch iterator should resume instead of starting a
        new epoch.
    pending_skip : int
        Number of already-acknowledged batches the next epoch iterator must
        replay and discard.
    pending_rng : dict or None
        Encoded RNG state to install before the next epoch iterator's
        shuffle.
    active_depth : int
        Nesting depth of epoch iterators currently fetching data.  Only the
        iterator at depth 0 (the one returned to training code) records
        deliveries.

    """
    def __init__(self):
        self.epoch = 0
        self.delivered = 0
        self.acknowledged = 0
        self.started = False
        self.current_epoch_start_rng = None
        self.pending_resume = False
        self.pending_skip = 0
        self.pending_rng = None
        self.active_depth = 0

    def begin_epoch(self):
        """Prepare the counters for a newly created epoch iterator."""
        if self.pending_resume:
            self.pending_resume = False
            self.delivered = self.acknowledged
            return
        if self.started:
            self.epoch += 1
        else:
            self.started = True
        self.delivered = 0
        self.acknowledged = 0
        self.current_epoch_start_rng = None
        self.pending_skip = 0
        self.pending_rng = None

    def reset(self):
        """Rewind to the start of epoch 0, discarding acknowledged progress."""
        self.epoch = 0
        self.delivered = 0
        self.acknowledged = 0
        self.started = False
        self.current_epoch_start_rng = None
        self.pending_resume = False
        self.pending_skip = 0
        self.pending_rng = None
        self.active_depth = 0

    def take_pending_skip(self):
        """Consume and return the number of batches to replay and discard."""
        skip = self.pending_skip
        self.pending_skip = 0
        return skip

    def record_delivery(self):
        self.delivered += 1

    def acknowledge(self):
        self.acknowledged = self.delivered

    def remaining(self, total):
        """Number of units left to deliver in the current epoch."""
        return total - self.delivered


@add_metaclass(ABCMeta)
class AbstractDataStream(object):
    """A stream of data separated into epochs.

    A data stream is an iterable stream of examples/minibatches. It shares
    similarities with Python file handles return by the ``open`` method.
    Data streams can be closed using the :meth:`close` method and reset
    using :meth:`reset` (similar to ``f.seek(0)``).

    Parameters
    ----------
    iteration_scheme : :class:`.IterationScheme`, optional
        The iteration scheme to use when retrieving data. Note that not all
        datasets support the same iteration schemes, some datasets require
        one, and others don't support any. In case when the data stream
        wraps another data stream, the choice of supported iteration
        schemes is typically even more limited. Be sure to read the
        documentation of the dataset or data stream in question.
    axis_labels : dict, optional
        Maps source names to tuples of strings describing axis semantics,
        one per axis. Defaults to `None`, i.e. no information is available.

    Attributes
    ----------
    iteration_scheme : :class:`.IterationScheme`
        The iteration scheme used to retrieve data. Can be ``None`` when
        not used.
    sources : tuple of strings
        The names of the data sources returned by this data stream, as
        given by the dataset.
    produces_examples : bool
        Whether this data stream produces examples (as opposed to batches
        of examples).

    """
    def __init__(self, iteration_scheme=None, axis_labels=None):
        self.iteration_scheme = iteration_scheme
        self.axis_labels = axis_labels

    @property
    def produces_examples(self):
        if self.iteration_scheme:
            return self.iteration_scheme.requests_examples
        elif not hasattr(self, '_produces_examples'):
            raise ValueError("cannot infer type of stream for {} instance; "
                             "set the produces_examples attribute to True "
                             "(for example streams) or False (for batch "
                             "streams).".format(self.__class__.__name__))
        else:
            return self._produces_examples

    @produces_examples.setter
    def produces_examples(self, value):
        if self.iteration_scheme:
            raise ValueError("cannot set produces_examples on {} instance; "
                             "determined by iteration scheme {}".format(
                                 self.__class__.__name__,
                                 self.iteration_scheme))
        self._produces_examples = value

    def get_data(self, request=None):
        """Request data from the dataset or the wrapped stream.

        Parameters
        ----------
        request : object
            A request fetched from the `request_iterator`.

        Notes
        -----
        It is possible to build a usable stream in terms of underlying
        streams for the purposes of training by only implementing
        `get_epoch_iterator`, thus this method is optional.

        """
        raise NotImplementedError

    def reset(self):
        """Reset the data stream."""

    def close(self):
        """Gracefully close the data stream, e.g. releasing file handles."""

    def next_epoch(self):
        """Switch the data stream to the next epoch."""

    @abstractmethod
    def get_epoch_iterator(self, as_dict=False):
        return DataIterator(
            self,
            self.iteration_scheme.get_request_iterator()
            if self.iteration_scheme else None,
            as_dict=as_dict,
            checkpoint_state=self._checkpoint_state())

    # ------------------------------------------------------------------
    # Checkpointing and batch acknowledgement protocol.
    #
    # The protocol is defined on the base class so that every stream can be
    # asked for a checkpoint.  Components that cannot faithfully describe
    # what they consumed declare ``checkpoint_supported = False`` and are
    # rejected before anything is written.
    #
    # All streams in a pipeline share a single :class:`CheckpointState`
    # object, owned by the innermost DataStream.  The epoch iterator at
    # nesting depth 0 (the one returned to training code) counts delivered
    # batches; a call to :meth:`acknowledge` promotes the delivered count to
    # the acknowledged count.  Because every transformer reads from its
    # child in lockstep, this single counter describes the whole chain.
    # ------------------------------------------------------------------

    #: Whether this component can report a faithful consumed position.
    checkpoint_supported = None

    def _checkpoint_state(self):
        """Return the shared :class:`CheckpointState` of the pipeline."""
        child = getattr(self, 'data_stream', None)
        if isinstance(child, AbstractDataStream):
            return child._checkpoint_state()
        if not hasattr(self, '_checkpoint_state_'):
            self._checkpoint_state_ = CheckpointState()
        return self._checkpoint_state_

    def _reset_checkpoint_state(self):
        """Rewind the acknowledged position to the start of epoch 0."""
        self._checkpoint_state().reset()

    def _identity(self):
        """Return a structural description used for checkpoint validation.

        Transforming streams are described by their class, their sources and
        their configuration, plus the identity of their wrapped stream.
        :class:`DataStream` adds the dataset identity.

        """
        identity = {
            'class': type(self).__name__,
            'sources': list(self.sources),
            'produces_examples': bool(self.produces_examples),
        }
        config = config_identity(self)
        if config:
            identity['config'] = config
        scheme = getattr(self, 'iteration_scheme', None)
        if scheme is not None:
            identity['scheme'] = scheme_identity(scheme)
        return identity

    def _checkpoint_extras(self):
        """Return the checkpoint fields specific to this component."""
        return {}

    def _checkpoint_snapshot(self):
        """Return a recursive description of this stream and its children."""
        snapshot = {
            'class': type(self).__name__,
            'identity': self._identity(),
        }
        snapshot.update(self._checkpoint_extras())
        child = getattr(self, 'data_stream', None)
        if isinstance(child, AbstractDataStream):
            snapshot['child'] = child._checkpoint_snapshot()
        return snapshot

    def acknowledge(self):
        """Mark every batch delivered so far in this epoch as processed.

        Training loops should call this once they have durably consumed the
        most recently returned batch (e.g. after gradients were applied and
        the model was saved).  Only the acknowledged position is recorded in
        :meth:`get_checkpoint`; anything delivered but not acknowledged is
        delivered again after a restore.

        """
        self._checkpoint_state().acknowledge()

    def get_checkpoint(self):
        """Return a JSON-serializable checkpoint of the acknowledged position.

        Returns
        -------
        checkpoint : dict
            Describes the epoch, the number of acknowledged batches, the
            random number generator states required to reproduce the exact
            sample order, and a fingerprint binding the checkpoint to the
            data source and the shape of the pipeline.

        Raises
        ------
        CheckpointNotSupportedError
            If any pipeline component cannot faithfully report its consumed
            position.  Nothing is produced in that case, so a partially
            valid checkpoint is never written.

        """
        problems = collect_unsupported(self)
        if problems:
            raise CheckpointNotSupportedError(format_unsupported(problems))
        state = self._checkpoint_state()
        snapshot = self._checkpoint_snapshot()
        return {
            'fuel_checkpoint_version': CHECKPOINT_VERSION,
            'epoch': int(state.epoch),
            'acknowledged': int(state.acknowledged),
            'epoch_start_rng_state': state.current_epoch_start_rng,
            'sources': list(self.sources),
            'produces_examples': bool(self.produces_examples),
            'environment': environment_identity(),
            'node': snapshot,
        }

    def set_checkpoint(self, checkpoint):
        """Restore a checkpoint produced by :meth:`get_checkpoint`.

        The next batch produced by this stream will be the one following the
        last acknowledged batch.  Any batch delivered but not acknowledged
        before the checkpoint was taken is therefore produced again.

        Parameters
        ----------
        checkpoint : dict
            Checkpoint to restore.

        Raises
        ------
        IncompatibleCheckpointError
            If the checkpoint was taken for a different data source, a
            different iteration scheme, a different transformer chain, or
            under different configuration values.
        CheckpointNotSupportedError
            If the current pipeline contains components that cannot be
            resumed.

        """
        problems = collect_unsupported(self)
        if problems:
            raise CheckpointNotSupportedError(format_unsupported(problems))
        if not isinstance(checkpoint, dict):
            raise IncompatibleCheckpointError(
                'checkpoint must be a dictionary, got {!r}'.format(
                    type(checkpoint).__name__))
        version = checkpoint.get('fuel_checkpoint_version')
        if version != CHECKPOINT_VERSION:
            raise IncompatibleCheckpointError(
                'checkpoint format version {!r} is not supported (expected '
                '{})'.format(version, CHECKPOINT_VERSION))
        differences = []
        differences.extend(fingerprint_diff(
            fingerprint(checkpoint.get('node') or {}),
            fingerprint(self._checkpoint_snapshot()), 'node'))
        differences.extend(fingerprint_diff(
            checkpoint.get('sources'), list(self.sources), 'sources'))
        differences.extend(fingerprint_diff(
            bool(checkpoint.get('produces_examples')),
            bool(self.produces_examples), 'produces_examples'))
        differences.extend(fingerprint_diff(
            checkpoint.get('environment'), environment_identity(),
            'environment'))
        if differences:
            raise IncompatibleCheckpointError(
                'checkpoint is incompatible with the current data stream:\n'
                + '\n'.join(differences))
        self._restore(checkpoint)

    def _restore(self, checkpoint):
        """Install the acknowledged position and RNG states of a checkpoint."""
        state = self._checkpoint_state()
        state.epoch = int(checkpoint.get('epoch', 0))
        state.acknowledged = int(checkpoint.get('acknowledged', 0))
        state.delivered = state.acknowledged
        state.started = True
        state.current_epoch_start_rng = checkpoint.get(
            'epoch_start_rng_state')
        state.pending_resume = True
        state.pending_skip = state.acknowledged
        state.pending_rng = checkpoint.get('epoch_start_rng_state')
        self._restore_snapshot(checkpoint.get('node') or {})

    def _restore_snapshot(self, snapshot):
        """Restore per-component state from a checkpoint node.

        The RNG that determines the sample order lives on the iteration
        scheme and is restored through the top-level
        ``epoch_start_rng_state``; nodes only need to recurse into their
        children so that custom components can override this hook.

        """
        child = getattr(self, 'data_stream', None)
        if isinstance(child, AbstractDataStream) and 'child' in snapshot:
            child._restore_snapshot(snapshot['child'])

    def iterate_epochs(self, as_dict=False):
        """Allow iteration through all epochs.

        Notes
        -----
        This method uses the :meth:`get_epoch_iterator` method to retrieve
        the :class:`DataIterator` for each epoch. The default
        implementation of this method resets the state of the data stream
        so that the new epoch can read the data from the beginning.
        However, this behavior only works as long as the ``epochs``
        property is iterated over using e.g. ``for epoch in
        stream.epochs``. If you create the data iterators in advance (e.g.
        using ``for i, epoch in zip(range(10), stream.epochs`` in legacy
        Python) you must call the :meth:`reset` method yourself.

        """
        while True:
            yield self.get_epoch_iterator(as_dict=as_dict)


class DataStream(AbstractDataStream):
    """A stream of data from a dataset.

    Parameters
    ----------
    dataset : instance of :class:`Dataset`
        The dataset from which the data is fetched.

    """
    def __init__(self, dataset, **kwargs):
        if dataset.axis_labels:
            kwargs.setdefault('axis_labels', dataset.axis_labels.copy())
        super(DataStream, self).__init__(**kwargs)
        # A DataStream with no iteration scheme is considered an example stream
        # by default
        if not self.iteration_scheme:
            self.produces_examples = True
        # If the data stream produces examples, remove 'batch' from axis labels
        if self.produces_examples and self.axis_labels:
            for source, labels in iteritems(self.axis_labels):
                self.axis_labels[source] = tuple(
                    label for label in labels if label != 'batch')
        self.dataset = dataset
        self.data_state = self.dataset.open()
        self._fresh_state = True

    @property
    def sources(self):
        if hasattr(self, '_sources'):
            return self._sources
        return self.dataset.sources

    @sources.setter
    def sources(self, value):
        self._sources = value

    def close(self):
        self.data_state = self.dataset.close(self.data_state)

    def reset(self):
        self.data_state = self.dataset.reset(self.data_state)
        self._fresh_state = True
        # A reset rewinds the pipeline; discard any acknowledged position.
        self._reset_checkpoint_state()

    def next_epoch(self):
        self.data_state = self.dataset.next_epoch(self.data_state)

    def get_data(self, request=None):
        """Get data from the dataset."""
        return self.dataset.get_data(self.data_state, request)

    def get_epoch_iterator(self, **kwargs):
        """Get an epoch iterator for the data stream."""
        state = self._checkpoint_state()
        state.begin_epoch()
        if not self._fresh_state:
            self.next_epoch()
        else:
            self._fresh_state = False
        # When resuming, reinstall the RNG state the saved epoch was shuffled
        # from so that replaying the acknowledged prefix reproduces exactly
        # the same samples.  Otherwise remember the state *before* this
        # epoch's shuffle, which is what a later checkpoint will record.
        if state.pending_rng is not None:
            set_rng_state(getattr(self.iteration_scheme, 'rng', None),
                          state.pending_rng)
            state.pending_rng = None
        else:
            state.current_epoch_start_rng = get_rng_state(
                getattr(self.iteration_scheme, 'rng', None))
        return super(DataStream, self).get_epoch_iterator(**kwargs)

    def _identity(self):
        identity = super(DataStream, self)._identity()
        identity['dataset'] = dataset_identity(self.dataset)
        return identity

    @classmethod
    def default_stream(cls, dataset, **kwargs):
        data_stream = cls(dataset, **kwargs)
        return dataset.apply_default_transformers(data_stream)


class ServerDataStream(AbstractDataStream):
    """A data stream that receives batches from a Fuel server.

    Parameters
    ----------
    sources : tuple of strings
        The names of the data sources returned by this data stream.
    produces_examples : bool
        Whether this data stream produces examples (as opposed to batches
        of examples).
    host : str, optional
        The host to connect to. Defaults to ``localhost``.
    port : int, optional
        The port to connect on. Defaults to 5557.
    hwm : int, optional
        The `ZeroMQ high-water mark (HWM)
        <http://zguide.zeromq.org/page:all#High-Water-Marks>`_ on the
        receiving socket. Increasing this increases the buffer, which can
        be useful if your data preprocessing times are very random.
        However, it will increase memory usage. There is no easy way to
        tell how many batches will actually be queued with a particular
        HWM. Defaults to 10. Be sure to set the corresponding HWM on the
        server's end as well.
    axis_labels : dict, optional
        Maps source names to tuples of strings describing axis semantics,
        one per axis. Defaults to `None`, i.e. no information is available.

    """
    def __init__(self, sources, produces_examples, host='localhost', port=5557,
                 hwm=10, axis_labels=None):
        super(ServerDataStream, self).__init__(axis_labels=axis_labels)
        self.sources = sources
        self.produces_examples = produces_examples
        self.host = host
        self.port = port
        self.hwm = hwm
        self.connected = False

    def connect(self):
        context = zmq.Context()
        self.socket = socket = context.socket(zmq.PULL)
        socket.set_hwm(self.hwm)
        socket.connect("tcp://{}:{}".format(self.host, self.port))
        self.connected = True

    def get_data(self, request=None):
        if request is not None:
            raise ValueError
        if not self.connected:
            self.connect()
        data = recv_arrays(self.socket)
        return tuple(data)

    def get_epoch_iterator(self, **kwargs):
        return super(ServerDataStream, self).get_epoch_iterator(**kwargs)

    def close(self):
        pass

    def next_epoch(self):
        pass

    def reset(self):
        pass

    def __getstate__(self):
        state = self.__dict__.copy()
        state['connected'] = False
        if 'socket' in state:
            del state['socket']
        return state
