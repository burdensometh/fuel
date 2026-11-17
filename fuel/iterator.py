import six

from fuel.exceptions import CheckpointNotSupportedError


class DataIterator(six.Iterator):
    """An iterator over data, representing a single epoch.

    Parameters
    ----------
    data_stream : :class:`DataStream` or :class:`Transformer`
        The data stream over which to iterate.
    request_iterator : iterator
        An iterator which returns the request to pass to the data stream
        for each step.
    as_dict : bool, optional
        If `True`, return dictionaries mapping source names to data
        from each source. If `False` (default), return tuples in the
        same order as `data_stream.sources`.
    checkpoint_state : :class:`~fuel.streams.CheckpointState`, optional
        Shared acknowledged-progress state.  Every iterator in a pipeline
        receives the same object; only the outermost iterator (the one
        returned to training code) records deliveries and replays the
        acknowledged prefix when resuming.

    """
    def __init__(self, data_stream, request_iterator=None, as_dict=False,
                 checkpoint_state=None):
        self.data_stream = data_stream
        self.request_iterator = request_iterator
        self.as_dict = as_dict
        self.checkpoint_state = checkpoint_state
        self._resume_done = False

    def __iter__(self):
        return self

    def _fetch(self):
        if self.request_iterator is not None:
            return self.data_stream.get_data(next(self.request_iterator))
        return self.data_stream.get_data()

    def _format(self, data):
        if self.as_dict:
            return dict(zip(self.data_stream.sources, data))
        return data

    def _resume(self):
        """Replay and discard the batches acknowledged before the checkpoint.

        The epoch is rebuilt from its start with the same seed and schedule,
        so the discarded batches are identical to the ones the training loop
        already consumed.  Whatever was delivered but *not* acknowledged is
        not skipped, and is therefore returned again (at-least-once).

        """
        state = self.checkpoint_state
        for _ in range(state.take_pending_skip()):
            self._fetch()

    def __next__(self):
        state = self.checkpoint_state
        outermost = state is not None and state.active_depth == 0
        if not outermost:
            return self._format(self._fetch())
        state.active_depth += 1
        try:
            if not self._resume_done:
                self._resume_done = True
                self._resume()
            data = self._fetch()
        finally:
            state.active_depth -= 1
        state.record_delivery()
        return self._format(data)

    def acknowledge(self):
        """Mark the most recently returned batch as durably consumed.

        This is a convenience shortcut for
        :meth:`fuel.streams.AbstractDataStream.acknowledge`; the training
        loop may call either one.  Batches that were returned but never
        acknowledged are delivered again after restoring a checkpoint.

        """
        if self.checkpoint_state is not None:
            self.checkpoint_state.acknowledge()
        else:
            raise CheckpointNotSupportedError(
                'this epoch iterator has no checkpoint state; acknowledge '
                'on the outermost stream returned to the training loop')
