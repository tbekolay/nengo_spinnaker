"""Objects used to represent Nengo networks as instantiated on SpiNNaker.
"""
import collections
import enum
from itertools import chain
from six import iteritems, itervalues

from ..utils.collections import counter


class ConnectionMap(object):
    """A container which represents all of the connections in a model and the
    parameters which are associated with them.

    A `ConnectionMap` maps from source objects to their outgoing ports to lists
    of transmission parameters and the objects which receive packets. This may
    be best expressed as::

        {source_object:
            {source_port: [
                ((signal_parameters, transmit_parameters),
                 [(sink_object, sink_port, reception_parameters), ...]),
                ...],
            ...},
        ...}

    """
    def __init__(self):
        """Create a new empty connection map."""
        # Construct the connection map internal structure
        self._connections = collections.defaultdict(
            lambda: collections.defaultdict(list))

    def add_connection(self, source_object, source_port, signal_parameters,
                       transmission_parameters, sink_object, sink_port,
                       reception_parameters):
        """Add a new connection to the map of connections.

        Parameters
        ----------
        signal_parameters : :py:class:`~.SignalParameters`
            Parameters describing how the signal will be transmitted (keyspace,
            weight, etc.).
        transmission_parameters :
            Source-specific parameters of how the signal is to be created
            (e.g., decoders, transform, etc.).
        reception_parameters : :py:class:`~.ReceptionParameters`
            Sink-specific parameters of how the received packets are to be
            treated.
        """
        # Combine the signal parameters with the transmission parameters
        signal_parameters = (signal_parameters, transmission_parameters)

        # See if we can combine the connection with an existing set of
        # transmission parameters.
        pars_list = self._connections[source_object][source_port]
        for pars, sinks in pars_list:
            # If the transmission parameters are equivalent then we can add the
            # sink parameters to the list of sinks.
            if pars == signal_parameters:
                break
        else:
            # If we can't combine with an existing connection then we need to
            # add a new transmission parameters and set of sinks.
            psp = _ParsSinksPair(signal_parameters)
            pars_list.append(psp)
            sinks = psp.sinks

        # Add the parameters to the list of sinks
        sinks.append(_SinkPars(sink_object, sink_port, reception_parameters))

    def add_default_keyspace(self, keyspace):
        """Add a unique entry to all signals which have not yet been assigned a
        keyspace.
        """
        # Create an object which can return a unique ID for each objects as we
        # request it.
        obj_ids = collections.defaultdict(counter())

        # For each source object
        for obj, ptps in iteritems(self._connections):
            # Hold the count of connection IDs
            conn_id = counter()

            # For each transmission parameter, sinks pair
            for tps_sinks in chain(*itervalues(ptps)):
                # If the signal parameter doesn't have a keyspace assign one
                # using the default keyspace.
                sp, _ = tps_sinks.parameters

                if sp.keyspace is None:
                    # Assign the keyspace
                    sp.keyspace = keyspace(object=obj_ids[obj],
                                           connection=conn_id())

    def get_signals_from(self, source_object):
        """Get the signals transmitted by a source object.

        Returns
        -------
        {port : [signal_parameters, ...], ...}
            Dictionary mapping ports to lists of parameters for the signals
            that originate from them.
        """
        signals = dict()

        # For every port and list of (transmission pars, sinks) associated with
        # it add the transmission parameters to the correct list of signals.
        for port, sigs in iteritems(self._connections[source_object]):
            signals[port] = list(pars.parameters for pars in sigs)

        return signals

    def get_signals_to(self, sink_object):
        """Get the signals received by a sink object.

        Returns
        -------
        {port : [ReceptionSpec, ...], ...}
            Dictionary mapping ports to the lists of objects specifying
            incoming signals.
        """
        signals = collections.defaultdict(list)

        # For all connections we have reference to identify those which
        # terminate at the given object. For those that do add a new entry to
        # the signal dictionary.
        tps_sinks = chain(*chain(*(itervalues(x) for x in
                                   itervalues(self._connections))))
        for tp_sinks in tps_sinks:
            # tp_sinks are pairs of transmission parameters and sinks
            # Extract the transmission parameters
            sp, _ = tp_sinks.parameters

            # For each sink, if the sink object is the specified object
            # then add signal to the list.
            for sink in tp_sinks.sinks:
                if sink.sink_object is sink_object:
                    # This is the desired sink object, so remember the
                    # signal. First construction the reception
                    # specification.
                    signals[sink.port].append(
                        ReceptionSpec(sp, sink.reception_parameters)
                    )

        return signals

    def get_signals(self):
        """Extract all the signals from the connection map.

        Returns
        -------
        [Signal, ...]
            List of signal objects derived from the contents of the connection
            map.
        """
        # Prepare to store a list of signals
        signals = list()

        # For each source object and set of sinks create a new signal.
        for source, port_conns in iteritems(self._connections):
            # For each connection look at the sinks and the signal parameters.
            for (sig_pars, _), par_sinks in chain(*itervalues(port_conns)):
                # Create a new signal and append to the list
                signals.append(Signal(
                    source,
                    (ps.sink_object for ps in par_sinks),  # Extract the sinks
                    sig_pars.keyspace,
                    sig_pars.weight
                ))

        return signals


class OutputPort(enum.Enum):
    """Indicate the intended transmitting part of an executable."""
    standard = 0
    """Standard, value-based, output port."""


class InputPort(enum.Enum):
    """Indicate the intended receiving part of an executable."""
    standard = 0
    """Standard, value-based, output port."""


class SignalParameters(object):
    """Basic parameters that can be applied to a signal.

    Attributes
    ----------
    latching : bool
        If False (the default) any receiving filters should reset their inputs
        every time-step.  If True then receiving filters should hold their
        inputs until new values are received.
    weight : int
        Number of packets expected to be sent every time-step.
    keyspace : :py:class:`~rig.bitfield.BitField` or None
        Keyspace which will be used to assign keys to packets.
    """
    def __init__(self, latching=False, weight=0, keyspace=None):
        # Store the parameters
        self.latching = latching
        self.weight = weight
        self.keyspace = keyspace

    def __eq__(self, other):
        # Equivalent if the latching spec is the same, the weight is the same
        # and the keyspaces are equivalent.
        if ((self.latching is other.latching) and
                (self.weight == other.weight) and
                (self.keyspace == other.keyspace)):
            return True
        else:
            return False

    def __ne__(self, b):
        return not self == b


class TransmissionParameters(object):
    """Basic parameters that relate to the transmission of a series of
    multicast packets.
    """
    def __eq__(self, other):
        # Equivalent if the types are the same
        return type(self) is type(other)

    def __ne__(self, other):
        return not (self == other)


ReceptionParameters = collections.namedtuple("ReceptionParameters", "filter")
"""Basic reception parameters that relate to the reception of a series of
multicast packets.

Attributes
----------
filter : :py:class:`~nengo.synapses.Synapse`
    Synaptic filter which should be applied to received values.
"""


class _ParsSinksPair(collections.namedtuple("_PSP", "parameters, sinks")):
    """Pair of transmission parameters and sink tuples."""
    def __new__(cls, signal_parameters, sinks=list()):
        # Copy the sinks list before calling __new__
        sinks = list(sinks)
        return super(_ParsSinksPair, cls).__new__(cls, signal_parameters,
                                                  sinks)


_SinkPars = collections.namedtuple("_SinkPars", ["sink_object", "port",
                                                 "reception_parameters"])
"""Collection of parameters for a sink."""


ReceptionSpec = collections.namedtuple(
    "ReceptionSpec", ["signal_parameters",
                      "reception_parameters"]
)
"""Specification of an incoming connection.

Attributes
----------
signal_parameters : :py:class:`~.SignalParameters`
    Description of how the signal will be transmitted.
reception_parameters : :py:class:`~.ReceptionParameters`
    Object specific description of how the received signal is to be handled
    (e.g., the type of filter to use).
"""


class Signal(object):
    """Details of a stream of multicast packets that will be transmitted across
    the SpiNNaker system.

    Attributes
    ----------
    source :
        Object representing the source of the stream of packets.
    sinks :
        List of objects representing the sinks of the stream of packets.
    keyspace : :py:class:`~rig.bitfield.BitField`
        Keyspace used to derive keys for the packets.
    weight : int
        Number of packets expected to be sent across the packet each time-step.
    """
    def __init__(self, source, sinks, keyspace, weight):
        """Create a new signal."""
        # Store all the parameters, copying the list of sinks.
        self.source = source
        self.sinks = list(sinks)
        self.keyspace = keyspace
        self.weight = weight
