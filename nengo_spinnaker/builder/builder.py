"""SpiNNaker builder for Nengo models."""
import collections
import itertools
import nengo
from nengo.cache import NoDecoderCache
from nengo.utils import numpy as npext
import numpy as np
from six import itervalues

from . import model
from nengo_spinnaker.netlist import Net, Netlist
from nengo_spinnaker.utils import collections as collections_ext
from nengo_spinnaker.utils.keyspaces import KeyspaceContainer

BuiltConnection = collections.namedtuple(
    "BuiltConnection", "decoders, eval_points, transform, solver_info"
)
"""Parameters which describe a Connection."""


def get_seed(obj, rng):
    seed = rng.randint(npext.maxint)
    return (seed if getattr(obj, "seed", None) is None else obj.seed)


class Model(object):
    """Model which has been built specifically for simulation on SpiNNaker.

    Attributes
    ----------
    dt : float
        Simulation timestep in seconds.
    machine_timestep : int
        Real-time duration of a simulation timestep in microseconds.
    decoder_cache :
        Cache used to reduce the time spent solving for decoders.
    params : {object: build details, ...}
        Map of Nengo objects (Ensembles, Connections, etc.) to their built
        equivalents.
    seeds : {object: int, ...}
        Map of Nengo objects to the seeds used in their construction.
    keyspaces : {keyspace_name: keyspace}
        Map of keyspace names to the keyspace which they may use.
    objects_operators : {object: operator, ...}
        Map of objects to the operators which will simulate them on SpiNNaker.
    extra_operators: [operator, ...]
        Additional operators.
    connection_map :
        Connection manager.
    """

    builders = collections_ext.registerabledict()
    """Builders for Nengo objects.

    Each object in the Nengo network is built by calling a builder function
    registered in this dictionary.  The builder function must be of the form:

        .. py:function:: builder(model, object)

    It is free to modify the model as required (including doing nothing to
    suppress SpiNNaker simulation of the object).
    """

    transmission_parameter_builders = collections_ext.registerabledict()
    """Functions which can provide the parameters for transmitting values to
    simulate a connection.

    The parameters required to form multicast packets to simulate a Nengo
    Connection vary depending on the type of the object at the start of the
    connection. Functions to build these parameters can be registered in this
    dictionary against the type of the originating object. Functions must be of
    the form:

        .. py:function:: builder(model, connection)

    It is recommended that functions set the value of
    `model.params[connection]` to an instance of :py:class:`~.BuiltConnection`
    alongside returning an appropriate value to use as the transmission
    parameters.
    """

    source_getters = collections_ext.registerabledict()
    """Functions to retrieve the specifications for the sources of signals.

    Before a connection is built an attempt is made to determine where the
    signal it represents on SpiNNaker will originate; a source getter is called
    to perform this task.  A source getter should resemble:

        .. py:function:: getter(model, connection)

    The returned item can be one of two things:
     * `None` will suppress simulation of the connection on SpiNNaker -- an
       example of this being useful is in optimising out connections from
       constant valued Nodes to ensembles or reusing an existing connection.
     * a :py:class:`~.spec` object which will be used to determine nature of
       the signal (in particular, the key and mask that it should use, whether
       it is latching or otherwise and the cost of the signal in terms of the
       frequency of packets across it).
    """

    reception_parameter_builders = collections_ext.registerabledict()
    """Functions which can provide the parameters for receiving values which
    simulate a connection.

    The parameters required to interpret multicast packets can vary based on
    the type of the object at the end of a Nengo Connection. Functions to build
    these parameters can be registered in this dictionary against the type of
    the terminating object.  Functions must of the form:

        .. py:function:: builder(model, connection)
    """

    sink_getters = collections_ext.registerabledict()
    """Functions to retrieve the specifications for the sinks of signals.

    A sink getter is analogous to a `source_getter`, but refers to the
    terminating end of a signal.
    """

    probe_builders = collections_ext.registerabledict()
    """Builder functions for probes.

    Probes can either require the modification of an existing object or the
    insertion of a new object into the model. A probe builder can be registered
    against the target of the probe and must be of the form:

        .. py:function:: probe_builder(model, probe)

    And is free the modify the model and existing objects as required.
    """

    def __init__(self, dt=0.001, machine_timestep=1000,
                 decoder_cache=NoDecoderCache(), keyspaces=None):
        self.dt = dt
        self.machine_timestep = machine_timestep
        self.decoder_cache = decoder_cache

        self.params = dict()
        self.seeds = dict()
        self.rngs = dict()
        self.rng = None

        self.config = None
        self.object_operators = dict()
        self.extra_operators = list()
        self.connection_map = model.ConnectionMap()

        if keyspaces is None:
            keyspaces = KeyspaceContainer()
        self.keyspaces = keyspaces

        # Builder dictionaries
        self._builders = dict()
        self._transmission_parameter_builders = dict()
        self._source_getters = dict()
        self._reception_parameter_builders = dict()
        self._sink_getters = dict()
        self._probe_builders = dict()

    def build(self, network, **kwargs):
        """Build a Network into this model.

        Parameters
        ----------
        network : :py:class:`~nengo.Network`
            Nengo network to build.  Passthrough Nodes will be removed.
        """
        # Store the network config
        self.config = network.config

        # Get a clean set of builders and getters
        self._builders = collections_ext.mrolookupdict()
        self._builders.update(self.builders)
        self._builders.update(kwargs.get("extra_builders", {}))

        self._transmission_parameter_builders = \
            collections_ext.mrolookupdict()
        self._transmission_parameter_builders.update(
            self.transmission_parameter_builders)
        self._transmission_parameter_builders.update(
            kwargs.get("extra_transmission_parameter_builders", {}))

        self._source_getters = collections_ext.mrolookupdict()
        self._source_getters.update(self.source_getters)
        self._source_getters.update(kwargs.get("extra_source_getters", {}))

        self._reception_parameter_builders = collections_ext.mrolookupdict()
        self._reception_parameter_builders.update(
            self.reception_parameter_builders)
        self._reception_parameter_builders.update(
            kwargs.get("extra_reception_parameter_builders", {}))

        self._sink_getters = collections_ext.mrolookupdict()
        self._sink_getters.update(self.sink_getters)
        self._sink_getters.update(kwargs.get("extra_sink_getters", {}))

        self._probe_builders = dict()
        self._probe_builders.update(self.probe_builders)
        self._probe_builders.update(kwargs.get("extra_probe_builders", {}))

        # Build
        self._build_network(network)

    def _build_network(self, network):
        # Get the seed for the network
        self.seeds[network] = get_seed(network, np.random)

        # Build all subnets
        for subnet in network.networks:
            self._build_network(subnet)

        # Get the random number generator for the network
        self.rngs[network] = np.random.RandomState(self.seeds[network])
        self.rng = self.rngs[network]

        # Build all objects
        for obj in itertools.chain(network.ensembles, network.nodes):
            self.make_object(obj)

        # Build all the connections
        for connection in network.connections:
            self.make_connection(connection)

        # Build all the probes
        for probe in network.probes:
            self.make_probe(probe)

    def make_object(self, obj):
        """Call an appropriate build function for the given object.
        """
        self.seeds[obj] = get_seed(obj, self.rng)
        self._builders[type(obj)](self, obj)

    def make_connection(self, conn):
        """Make a Connection and add a new signal to the Model.

        This method will build a connection and construct a new signal which
        will be included in the model.
        """
        # Set the seed for the connection
        self.seeds[conn] = get_seed(conn, self.rng)

        # Get the transmission parameters and reception parameters for the
        # connection.
        pre_type = type(conn.pre_obj)
        tps = self._transmission_parameter_builders[pre_type](self, conn)
        post_type = type(conn.post_obj)
        rps = self._reception_parameter_builders[post_type](self, conn)

        # Get the source and sink specification, then make the signal provided
        # that neither of specs is None.
        source = self._source_getters[pre_type](self, conn)
        sink = self._sink_getters[post_type](self, conn)

        if not (source is None or sink is None):
            # Construct the signal parameters
            sps = _make_signal_parameters(source, sink)

            # Add the connection to the connection map, this will automatically
            # merge connections which are equivalent.
            self.connection_map.add_connection(
                source.target.obj, source.target.port, sps, tps,
                sink.target.obj, sink.target.port, rps
            )

    def make_probe(self, probe):
        """Call an appropriate build function for the given probe."""
        self.seeds[probe] = get_seed(probe, self.rng)

        # Get the target type
        target_obj = probe.target
        if isinstance(target_obj, nengo.base.ObjView):
            target_obj = target_obj.obj

        # Build
        self._probe_builders[type(target_obj)](self, probe)

    def make_netlist(self, *args, **kwargs):
        """Convert the model into a netlist for simulating on SpiNNaker.

        Returns
        -------
        :py:class:`~nengo_spinnaker.netlist.Netlist`
            A netlist which can be placed and routed to simulate this model on
            a SpiNNaker machine.
        """
        # Call each operator to make vertices
        operator_vertices = dict()
        vertices = collections_ext.flatinsertionlist()
        load_functions = collections_ext.noneignoringlist()
        before_simulation_functions = collections_ext.noneignoringlist()
        after_simulation_functions = collections_ext.noneignoringlist()

        for op in itertools.chain(itervalues(self.object_operators),
                                  self.extra_operators):
            vxs, load_fn, pre_fn, post_fn = op.make_vertices(
                self, *args, **kwargs
            )

            operator_vertices[op] = vxs
            vertices.append(vxs)

            load_functions.append(load_fn)
            before_simulation_functions.append(pre_fn)
            after_simulation_functions.append(post_fn)

        # Construct the groups set
        groups = list()
        for vxs in itervalues(operator_vertices):
            # If multiple vertices were provided by an operator then we add
            # them as a new group.
            if isinstance(vxs, collections.Iterable):
                groups.append(set(vxs))

        # Construct nets from the signals
        nets = list()
        for signal in itertools.chain(itervalues(self.connections_signals),
                                      self.extra_signals):
            # Get the source and sink vertices
            sources = operator_vertices[signal.source.obj]
            if not isinstance(sources, collections.Iterable):
                sources = (sources, )

            sinks = collections_ext.flatinsertionlist()
            for sink in signal.sinks:
                sinks.append(operator_vertices[sink.obj])

            # Create the net(s)
            for source in sources:
                nets.append(Net(source, list(sinks),
                            signal.weight, signal.keyspace))

        # Return a netlist
        return Netlist(
            nets=nets,
            vertices=vertices,
            keyspaces=self.keyspaces,
            groups=groups,
            load_functions=load_functions,
            before_simulation_functions=before_simulation_functions,
            after_simulation_functions=after_simulation_functions
        )


class netlistspec(collections.namedtuple(
        "netlistspec", "vertices, load_function, before_simulation_function, "
                       "after_simulation_function")):
    """Specification of how an operator should be added to a netlist."""
    def __new__(cls, vertices, load_function=None,
                before_simulation_function=None,
                after_simulation_function=None):
        return super(netlistspec, cls).__new__(
            cls, vertices, load_function, before_simulation_function,
            after_simulation_function
        )


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


class spec(collections.namedtuple("spec",
                                  "target, keyspace, weight, latching")):
    """Specification of a signal which can be returned by either a source or
    sink getter.

    Attributes
    ----------
    target : :py:class:`ObjectPort`
        Source or sink of a signal.

    The other attributes and arguments are as for :py:class:`~.Signal`.
    """
    def __new__(cls, target, keyspace=None, weight=0, latching=False):
        return super(spec, cls).__new__(cls, target, keyspace,
                                        weight, latching)


def _make_signal_parameters(source_spec, sink_spec):
    """Create parameters for a signal using specifications provided by the
    source and sink.

    Parameters
    ----------
    source_spec : spec
        Signal specification parameters from the source of the signal.
    sink_spec : spec
        Signal specification parameters from the sink of the signal.

    Returns
    -------
    :py:class:`~.SignalParameters`
        Description of the signal.
    """
    # Raise an error if keyspaces are specified by the source and sink
    if source_spec.keyspace is not None and sink_spec.keyspace is not None:
        raise NotImplementedError("Cannot merge keyspaces")

    # Create the signal parameters
    return model.SignalParameters(
        latching=source_spec.latching or sink_spec.latching,
        weight=max((source_spec.weight, sink_spec.weight)),
        keyspace=source_spec.keyspace or sink_spec.keyspace,
    )
