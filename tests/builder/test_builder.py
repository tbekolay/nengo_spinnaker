import mock
from mock import patch
import nengo
from nengo.cache import NoDecoderCache
import numpy as np
import pytest

from nengo_spinnaker.builder.builder import (
    Model, spec, ObjectPort, netlistspec, _make_signal_parameters
)
from nengo_spinnaker.builder.model import SignalParameters
from nengo_spinnaker.netlist import Vertex, VertexSlice


def test_model_init():
    """Test initialising a model, should be completely empty."""
    model = Model()
    assert model.dt == 0.001
    assert model.machine_timestep == 1000

    assert model.params == dict()
    assert model.seeds == dict()

    assert dict(model.object_operators) == dict()
    assert model.extra_operators == list()

    assert isinstance(model.decoder_cache, NoDecoderCache)
    assert len(model.keyspaces) == 1


def test_model_init_with_keyspaces():
    """Test initialising a model, should be completely empty."""
    keyspaces = mock.Mock()
    model = Model(keyspaces=keyspaces)

    assert model.keyspaces is keyspaces


class TestBuild(object):
    @pytest.mark.parametrize("use_make_object", [False, True])
    def test_builder_dictionaries_are_combined(self, use_make_object):
        """Test that the builder and extra_builders dictionaries are combined
        and that a mrolookupdict is used.
        """
        class A(object):
            seed = 101

        class B(object):
            pass

        builders = {A: mock.Mock()}
        extra_builders = {B: mock.Mock()}

        a = A()
        b = B()

        network = mock.Mock()
        network.seed = None
        network.connections = []
        network.ensembles = [a]
        network.nodes = [b]
        network.networks = []
        network.probes = []
        network.config = mock.Mock(name="config")

        if not use_make_object:
            # Patch the default builders
            with patch.object(Model, "builders", new=builders):
                # Create a model and build the mock network
                model = Model()
                model.build(network, extra_builders=extra_builders)

            # Assert that the config was stored in the model
            assert model.config is network.config
        else:
            # Create the model
            model = Model()
            model.rng = np.random

            # When using `make_object` directly `_builders` should be defined
            # and used.
            model._builders.update(builders)
            model._builders.update(extra_builders)

            # Build the objects
            model.make_object(a)
            model.make_object(b)

        # Assert that seeds were supplied
        assert model.seeds[a] == a.seed
        assert model.seeds[b] is not None

        # Assert the builders got called
        builders[A].assert_called_once_with(model, a)
        extra_builders[B].assert_called_once_with(model, b)

    def test_builds_hierarchy(self):
        """Test that networks are built hierarchically.
        """
        class A(object):
            seed = 101

        class B(object):
            pass

        builders = {A: mock.Mock()}
        extra_builders = {B: mock.Mock()}

        a = A()
        b = B()

        network_a = mock.Mock()
        network_a.seed = None
        network_a.connections = []
        network_a.ensembles = [a]
        network_a.nodes = []
        network_a.networks = []
        network_a.probes = []
        network_a.config = mock.Mock(name="config")

        network_b = mock.Mock()
        network_b.seed = None
        network_b.connections = []
        network_b.ensembles = []
        network_b.nodes = [b]
        network_b.networks = []
        network_b.probes = []
        network_b.config = mock.Mock(name="config")

        network = mock.Mock()
        network.seed = None
        network.connections = []
        network.ensembles = []
        network.nodes = []
        network.networks = [network_a, network_b]
        network.probes = []
        network.config = mock.Mock(name="config")

        # Patch the default builders
        with patch.object(Model, "builders", new=builders):
            # Create a model and build the mock network
            model = Model()
            model.build(network, extra_builders=extra_builders)

        # Assert that the config was stored in the model
        assert model.config is network.config

        # Assert that seeds were supplied
        assert model.seeds[a] == a.seed
        assert model.seeds[b] is not None

        # Assert the builders got called
        builders[A].assert_called_once_with(model, a)
        extra_builders[B].assert_called_once_with(model, b)


class TestMakeConnection(object):
    """Test the building of connections."""
    @pytest.mark.parametrize("use_make_connection", (True, False))
    def test_standard(self, use_make_connection):
        """Test building a single connection, ensure that all appropriate
        methods are called and that the signal is added to the connection map.
        """
        class A(object):
            pass

        # Create the connection (as a mock)
        connection_source = A()
        connection_sink = A()

        connection = mock.Mock()
        connection.pre_obj = connection_source
        connection.post_obj = connection_sink

        # Create the Model which we'll build with
        m = Model()

        # Modify the Model so that we can interpret calls to the connection map
        m.connection_map = mock.Mock(name="ConnectionMap")

        source = mock.Mock(name="Source Object")
        source_port = mock.Mock(name="Source Port")
        sink = mock.Mock(name="Sink Object")
        sink_port = mock.Mock(name="Sink Port")

        # Add some build methods
        def source_getter(model, conn):
            assert model is m
            assert conn is connection
            return spec(ObjectPort(source, source_port))

        def sink_getter(model, conn):
            assert model is m
            assert conn is connection
            return spec(ObjectPort(sink, sink_port))

        source_getters = {A: mock.Mock(side_effect=source_getter)}
        sink_getters = {A: mock.Mock(side_effect=sink_getter)}

        transmission_parameters = mock.Mock(name="Transmission Params")

        def transmission_builder(model, conn):
            assert model is m
            assert conn is connection
            return transmission_parameters

        reception_parameters = mock.Mock(name="Reception Params")

        def reception_builder(model, conn):
            assert model is m
            assert conn is connection
            return reception_parameters

        transmission_parameter_builders = {
            A: mock.Mock(side_effect=transmission_builder)
        }
        reception_parameter_builders = {
            A: mock.Mock(side_effect=reception_builder)
        }

        # Make the connection
        if use_make_connection:
            # Set an RNG to build with
            m.rng = np.random

            # Set the builders
            m._source_getters = source_getters
            m._sink_getters = sink_getters
            m._transmission_parameter_builders = \
                transmission_parameter_builders
            m._reception_parameter_builders = reception_parameter_builders

            # Build the connection directly
            m.make_connection(connection)
        else:
            # Embed the connection in a mock Nengo network and build that
            # instead.
            network = mock.Mock()
            network.seed = None
            network.connections = [connection]
            network.ensembles = []
            network.nodes = []
            network.networks = []
            network.probes = []

            # Build this (having overridden the builders)
            with mock.patch.object(m, "source_getters", source_getters), \
                    mock.patch.object(m, "sink_getters", sink_getters), \
                    mock.patch.object(m, "transmission_parameter_builders",
                                      transmission_parameter_builders), \
                    mock.patch.object(m, "reception_parameter_builders",
                                      reception_parameter_builders):
                m.build(network)

        # Assert the connection map received an appropriate call
        m.connection_map.add_connection.assert_called_once_with(
            source, source_port, SignalParameters(),
            transmission_parameters, sink, sink_port, reception_parameters
        )

    @pytest.mark.parametrize("no_source, no_sink", ((True, False), (False, True)))
    def test_source_is_none(self, no_source, no_sink):
        """Test that if either the source or sink is none no connection is
        added to the model.
        """
        class A(object):
            pass

        # Create the connection (as a mock)
        connection_source = A()
        connection_sink = A()

        connection = mock.Mock()
        connection.pre_obj = connection_source
        connection.post_obj = connection_sink

        # Create the Model which we'll build with
        m = Model()

        # Modify the Model so that we can interpret calls to the connection map
        m.connection_map = mock.Mock(name="ConnectionMap")

        obj = mock.Mock(name="Object")
        obj_port = mock.Mock(name="Port")

        # Add some build methods
        m._source_getters = ({A: lambda m, c: None} if no_source else
                             {A: lambda m, c: ObjectPort(obj, obj_port)})
        m._sink_getters = ({A: lambda m, c: None} if no_sink else
                           {A: lambda m, c: ObjectPort(obj, obj_port)})
        m._transmission_parameter_builders = {A: lambda m, c: None}
        m._reception_parameter_builders = {A: lambda m, c: None}

        # Make the connection
        # Set an RNG to build with
        m.rng = np.random

        # Build the connection directly
        m.make_connection(connection)

        # Assert no call was made to add_connection
        assert not m.connection_map.add_connection.called


class TestBuildProbe(object):
    """Test the building of probes."""
    @pytest.mark.parametrize("use_arguments", [False, True])
    @pytest.mark.parametrize("with_slice", [False, True])
    def test_standard(self, use_arguments, with_slice):
        # Create test network
        with nengo.Network() as network:
            a = nengo.Ensemble(100, 2)

            if not with_slice:
                p_a = nengo.Probe(a)
            else:
                p_a = nengo.Probe(a[0])

            p_n = nengo.Probe(a.neurons)

        # Create a model
        model = Model()

        # Dummy neurons builder
        ens_build = mock.Mock(name="ensemble builder")

        # Define two different probe build functions
        def build_ens_probe_fn(model, probe):
            assert ens_build.called
            assert model is model
            assert probe is p_a

        build_ens_probe = mock.Mock(wraps=build_ens_probe_fn)

        def build_neurons_probe_fn(model, probe):
            assert ens_build.called
            assert model is model
            assert probe is p_n

        build_neurons_probe = mock.Mock(wraps=build_neurons_probe_fn)

        # Build the model
        probe_builders = {nengo.Ensemble: build_ens_probe,
                          nengo.ensemble.Neurons: build_neurons_probe}
        with patch.object(model, "builders", new={nengo.Ensemble: ens_build}):
            if not use_arguments:
                with patch.object(model, "probe_builders", new=probe_builders):
                    model.build(network)
            else:
                with patch.object(model, "probe_builders", new={}):
                    model.build(network, extra_probe_builders=probe_builders)

        # Assert the probe functions were built
        assert p_a in model.seeds
        assert p_n in model.seeds
        assert build_ens_probe.call_count == 1
        assert build_neurons_probe.call_count == 1


def test_spec():
    """Test specifying the source or sink of a signal."""
    # With minimal arguments
    s = spec(None)
    assert s.target is None
    assert s.keyspace is None
    assert not s.latching
    assert s.weight == 0

    # With all arguments
    target = mock.Mock(name="target")
    keyspace = mock.Mock(name="keyspace")
    weight = 5
    latching = True

    s = spec(target, keyspace=keyspace, weight=weight, latching=latching)
    assert s.target is target
    assert s.keyspace is keyspace
    assert s.weight == weight
    assert s.latching is latching


class TestMakeSignalParameters(object):
    """Test constructing signal parameters from spec objects."""
    @pytest.mark.parametrize("a_is_latching, b_is_latching, latching",
                             [(False, False, False),
                              (True, False, True),
                              (False, True, True),
                              (True, True, True)])
    def test_latching(self, a_is_latching, b_is_latching, latching):
        # Construct the specs
        a_spec = spec(None, latching=a_is_latching)
        b_spec = spec(None, latching=b_is_latching)

        # Make the signal parameters, check they are correct
        sig_pars = _make_signal_parameters(a_spec, b_spec)
        assert sig_pars.latching is latching

    @pytest.mark.parametrize("source_weight, sink_weight",
                             [(4, 7), (5, 2), (2, 2)])
    def test_weight(self, source_weight, sink_weight):
        """Test that the greatest specified weight is used."""
        # Construct the specs
        a_spec = spec(None, weight=source_weight)
        b_spec = spec(None, weight=sink_weight)

        # Make the signal parameters, check they are correct
        sig_pars = _make_signal_parameters(a_spec, b_spec)
        assert sig_pars.weight == max((source_weight, sink_weight))

    def test_keyspace_from_source(self):
        """Check that the source keyspace is used if provided."""
        ks = mock.Mock(name="Keyspace")
        a_spec = spec(None, keyspace=ks)
        b_spec = spec(None)

        # Make the signal parameters, check they are correct
        sig_pars = _make_signal_parameters(a_spec, b_spec)
        assert sig_pars.keyspace is ks

    def test_keyspace_from_sink(self):
        """Check that the sink keyspace is used if provided."""
        ks = mock.Mock(name="Keyspace")
        a_spec = spec(None)
        b_spec = spec(None, keyspace=ks)

        # Make the signal parameters, check they are correct
        sig_pars = _make_signal_parameters(a_spec, b_spec)
        assert sig_pars.keyspace is ks

    def test_keyspace_collision(self):
        """Test that if both the source and spec provide a keyspace an error is
        raised.
        """
        a_spec = spec(None, keyspace=mock.Mock())
        b_spec = spec(None, keyspace=mock.Mock())

        # Make the signal parameters, this should raise an error
        with pytest.raises(NotImplementedError):
            _make_signal_parameters(a_spec, b_spec)


class TestMakeNetlist(object):
    """Test production of netlists from operators and signals."""
    def test_single_vertices(self):
        """Test that operators which produce single vertices work correctly and
        that all functions and signals are correctly collected and included in
        the final netlist.
        """
        raise NotImplementedError

    def test_extra_operators_and_signals(self):
        """Test the operators and signals in the extra_operators and
        extra_signals lists are included when building netlists.
        """
        raise NotImplementedError

    def test_multiple_sink_vertices(self):
        """Test that each of the vertices associated with a sink is correctly
        included in the sinks of a net.
        """
        raise NotImplementedError

    def test_multiple_source_vertices(self):
        """Test that each of the vertices associated with a source is correctly
        included in the sources of a net.
        """
        raise NotImplementedError
