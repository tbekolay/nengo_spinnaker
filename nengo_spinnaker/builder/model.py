"""Objects used to represent Nengo networks as instantiated on SpiNNaker.
"""
import collections
import enum
import numpy as np

from ..utils.hash import hash_slice


ObjectPort = collections.namedtuple("ObjectPort", "obj port")
"""Source or sink of a signal.

Parameters
----------
obj : intermediate object
    Intermediate representation of a Nengo object, or other object, which is
    the source or sink of a signal.
port : port
    Port that is the source or sink of a signal.
"""


class OutputPort(enum.Enum):
    """Indicate the intended transmitting part of an executable."""
    standard = 0
    """Standard, value-based, output port."""


class InputPort(enum.Enum):
    """Indicate the intended receiving part of an executable."""
    standard = 0
    """Standard, value-based, output port."""


class TransmissionParameters(object):
    """Basic transmission parameters that can be applied to a signal.

    Attributes
    ----------
    pre_slice : slice or None
        Slice which should be applied to any values before multiplying them by
        the transform.
    transform : array
        Transform which should be applied to yield the vector to transmit
        across the network.
    """
    def __init__(self, pre_slice, transform):
        # Store the slice
        self.pre_slice = pre_slice

        # Copy the array and make it immutable
        self.transform = np.array(transform)
        self.transform.flags.writeable = False

    def __hash__(self):
        # Hash by type, pre_slice and transform data
        return hash((self.__class__, hash_slice(self.pre_slice),
                     self.transform.shape, self.transform.data.tobytes()))

    def __eq__(self, b):
        # Equivalent if they are of exactly the same type
        if type(self) is not type(b):
            return False

        # Equivalent if the transform is the same shape and the same value
        if self.transform.shape != b.transform.shape:
            return False

        if np.any(self.transform != b.transform):
            return False

        # Equivalent if the slice is equivalent
        if self.pre_slice != b.pre_slice:
            return False

        # Otherwise equivalent
        return True

    def __ne__(self, b):
        return not self == b


ReceptionParameters = collections.namedtuple("ReceptionParameters", "filter")
"""Basic reception parameters than can be applied to a signal.

Attributes
----------
filter : :py:class:`~nengo.synapses.Synapse`
    Synaptic filter which should be applied to received values.
"""
