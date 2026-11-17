class AxisLabelsMismatchError(ValueError):
    """Raised when a pair of axis labels tuples do not match."""


class ConfigurationError(Exception):
    """Error raised when a configuration value is requested but not set."""


class MissingInputFiles(Exception):
    """Exception raised by a converter when input files are not found.

    Parameters
    ----------
    message : str
        The error message to be associated with this exception.
    filenames : list
        A list of filenames that were not found.

    """
    def __init__(self, message, filenames):
        self.filenames = filenames
        super(MissingInputFiles, self).__init__(message, filenames)


class NeedURLPrefix(Exception):
    """Raised when a URL is not provided for a file."""


class CheckpointError(Exception):
    """Base class for errors related to checkpointing data streams."""


class IncompatibleCheckpointError(CheckpointError):
    """Raised when a checkpoint does not describe the current data stream.

    This happens when the data source (e.g. the HDF5 file), the iteration
    scheme, the transformer chain or a relevant configuration value changed
    between the moment a checkpoint was saved and the moment it is restored.

    """


class CheckpointNotSupportedError(CheckpointError):
    """Raised when a stream component cannot be checkpointed.

    Fuel refuses to save a checkpoint for a pipeline containing a component
    whose state it cannot faithfully reproduce (or whose read-ahead makes it
    impossible to distinguish what has been read from what has been
    consumed). This is raised *before* anything is written, so a partially
    valid checkpoint can never be produced.

    """


# Alias kept for readability in user code.
UnsupportedCheckpointError = CheckpointNotSupportedError
