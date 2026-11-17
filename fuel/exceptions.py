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


class LabelRevisionError(ValueError):
    """Raised when a versioned label revision is invalid.

    This is raised e.g. when a correction set references unknown stable
    IDs, contains contradictory labels for the same ID within a single
    revision, uses a non-incremental version number, or when an unknown
    revision version is requested. In every case the whole submission is
    rejected without modifying the label store.

    """
