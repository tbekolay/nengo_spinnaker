"""Objects used to represent Nengo networks as instantiated on SpiNNaker.
"""
import collections
import enum
import numpy as np

from ..utils.hash import hash_slice


# Connection related objects
# --------------------------
# The following objects are concerned with specifying how connections are
# translated into streams of packets transmitted across the SpiNNaker network.
class Signal(object):
    """Stream of multicast packets associated with one source and multiple
    sinks.

    Attributes
    ----------
    source : ObjectPort
        The source of a stream of packets.
    transmission_parameters :
        Source-specific parameters which will be used to determine how the
        packets to be sent across the SpiNNaker network should be formed.
    sinks : [(ObjectPort, parameters), ...]
        List of pairs of sinks and parameters which are specific sinks and
        indicate how the received packets are to be treated.
    keyspace : keyspace or None
        Keyspace used for packets.  If None a keyspace will be allocated later.
    weight : int
        Number of packets expected to represent the signal during a single
        time-step.
    latching : bool
        Indicates that the receiving buffer must *not* be reset every
        simulation timestep but must hold its value until it next receives a
        packet. This parameters applies to all sinks.
    """
    def __init__(self, source, transmission_parameters, sinks=list(),
                 keyspace=None, weight=0, latching=False):
        # Simply store all the parameters
        self.source = source
        self.transmission_parameters = transmission_parameters
        self.sinks = list(sinks)
        self.keyspace = keyspace
        self.weight = weight
        self.latching = latching


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
                     self.transform.shape, self.transform.data))

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
