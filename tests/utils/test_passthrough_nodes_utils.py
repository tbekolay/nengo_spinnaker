import nengo
import numpy as np
import pytest

from nengo_spinnaker.utils.passthrough_nodes import (
    add_passthrough_node_to_probe_connections, combine_lti_synapses,
    remove_passthrough_nodes
)


def test_add_pasthrough_node_to_probe_connections():
    """Test that connections from passthrough Nodes to Probes are added
    correctly.
    """
    # Create a network with some awkward nesting
    with nengo.Network() as network:
        with nengo.Network():
            ptn = nengo.Node(size_in=3)

        with nengo.Network():
            probe = nengo.Probe(ptn[:], synapse=0.05)
            probe.seed = 1234

        probe2 = nengo.Probe(ptn)

        # And another completely unrelated probe
        e = nengo.Ensemble(100, 1)
        nengo.Probe(e)

    # Add the connection
    add_passthrough_node_to_probe_connections(network)

    # Check it is correct
    for conn in network.all_connections:
        if conn.post is probe:
            assert conn.pre_obj is ptn
            assert conn.pre_slice == slice(None)
            assert conn.seed == probe.seed
            assert conn.synapse.tau == 0.05
        else:
            assert conn.pre is ptn
            assert conn.post is probe2


@pytest.mark.parametrize(
    "synapse_1, synapse_2, expected_synapse",
    [(nengo.Lowpass(0.05), nengo.Lowpass(0.01),
      nengo.synapses.LinearFilter([1], [0.05*0.01, 0.05+0.01, 1]))]
)
def test_combine_lti_synapses(synapse_1, synapse_2, expected_synapse):
    # Combine the synapses
    new_synapse = combine_lti_synapses(synapse_1, synapse_2)

    # Assert this is as expected
    assert isinstance(new_synapse, nengo.synapses.LinearFilter)
    assert np.all(new_synapse.num == expected_synapse.num)
    assert np.all(new_synapse.den == expected_synapse.den)


def test_remove_passthrough_nodes():
    """Test that passthrough nodes are correctly removed from the following
    network:

       E01 --\                /--> E11
              > PN1 ---> PN2 >
       E02 --/                \--> E12
    """
    # Create the network
    with nengo.Network() as network:
        input_node = nengo.Node([0, 1])
        e01 = nengo.Ensemble(100, 2)
        e02 = nengo.Ensemble(100, 2)

        pn1 = nengo.Node(size_in=4)
        pn2 = nengo.Node(size_in=4)

        e11 = nengo.Ensemble(100, 2)
        e12 = nengo.Ensemble(100, 2)

        # Add the connections
        nengo.Connection(input_node, e01)
        c1 = nengo.Connection(e01, pn1[:2], synapse=None)
        c2 = nengo.Connection(e02, pn1[2:], synapse=None)
        nengo.Connection(pn1, pn2[[2, 3, 0, 1]], synapse=None)
        nengo.Connection(pn2[:2], e12, synapse=0.03)
        nengo.Connection(pn2[2:], e11, synapse=0.02)

        # Probe one of the ensembles
        p = nengo.Probe(e12)

    # Remove the passthrough nodes, this should return a simplified network and
    # a mapping from connections to the connection whose decoder they should
    # use.
    new_net, decoder_sources = remove_passthrough_nodes(network)

    assert set(new_net.all_ensembles) == set(network.all_ensembles)
    assert new_net.all_nodes == [input_node]
    assert new_net.all_probes == [p]
    assert len(new_net.all_connections) == 3

    for conn in new_net.all_connections:
        if conn.pre_obj is e01:
            assert conn.post_obj is e11
            assert conn.synapse.tau == 0.02
            assert np.all(conn.transform == np.eye(2))

            assert decoder_sources[conn] is c1
        elif conn.pre_obj is e02:
            assert conn.post_obj is e12
            assert conn.synapse.tau == 0.03
            assert np.all(conn.transform == np.eye(2))

            assert decoder_sources[conn] is c2
        else:
            assert conn.pre_obj is input_node


def test_remove_passthrough_nodes_combines_lti_synapses():
    with nengo.Network() as network:
        a = nengo.Node([1.0])
        b = nengo.Node(size_in=1)
        c = nengo.Ensemble(100, 1)

        ab = nengo.Connection(a, b, synapse=0.01)
        nengo.Connection(b, c, synapse=0.02)

    # Remove passthrough Nodes
    new_net, conns_decoders = remove_passthrough_nodes(network)

    # Check the new single connection
    assert len(new_net.all_connections) == 1
    assert np.all(new_net.all_connections[0].synapse.num == [1])
    assert np.all(new_net.all_connections[0].synapse.den ==
                  [0.01*0.02, 0.01+0.02, 1.0])


def test_remove_passthrough_nodes_fails_combines_synapses():
    with nengo.Network() as network:
        a = nengo.Ensemble(100, 1)
        b = nengo.Node(size_in=1)
        c = nengo.Ensemble(100, 1)

        ab = nengo.Connection(a, b, synapse=nengo.synapses.Triangle(0.01))
        nengo.Connection(b, c, synapse=nengo.synapses.Triangle(0.02))

    # Remove passthrough Nodes
    with pytest.raises(NotImplementedError) as excinfo:
        remove_passthrough_nodes(network)
    assert "Triangle" in str(excinfo.value)


def test_remove_passthrough_nodes_with_function():
    with nengo.Network() as network:
        a = nengo.Ensemble(100, 5)
        b = nengo.Node(size_in=2)
        c = nengo.Ensemble(100, 2)

        ab = nengo.Connection(a, b, function=lambda x: x[:2], synapse=None)
        nengo.Connection(b, c, synapse=None)

    # Remove passthrough Nodes
    new_net, conns_decoders = remove_passthrough_nodes(network)

    # Check the new single connection
    assert len(new_net.all_connections) == 1
    assert new_net.all_connections[0].function is ab.function


def test_remove_passthrough_nodes_with_preslices_no_function():
    with nengo.Network() as network:
        a = nengo.Ensemble(100, 5)
        b = nengo.Node(size_in=2)
        c = nengo.Node(size_in=1)
        d = nengo.Ensemble(100, 1)

        ab = nengo.Connection(a[:2], b, synapse=None)
        bc = nengo.Connection(b[0], c, synapse=None)
        cd = nengo.Connection(c[:], d, synapse=None)

    # Remove passthrough Nodes
    new_net, conns_decoders = remove_passthrough_nodes(network)

    # Check the new single connection
    assert len(new_net.all_connections) == 1
    assert new_net.all_connections[0].pre is a
    assert new_net.all_connections[0].post_obj is d
    assert np.all(new_net.all_connections[0].transform == [[1, 0, 0, 0, 0]])


def test_remove_passthrough_nodes_with_preslices_with_function():
    with nengo.Network() as network:
        a = nengo.Ensemble(100, 5)
        b = nengo.Node(size_in=2)
        c = nengo.Node(size_in=1)
        d = nengo.Ensemble(100, 1)

        ab = nengo.Connection(a[:2], b, synapse=None, function=lambda x: x)
        bc = nengo.Connection(b[0], c, synapse=None)
        cd = nengo.Connection(c[:], d, synapse=None)

    # Remove passthrough Nodes
    new_net, conns_decoders = remove_passthrough_nodes(network)

    # Check the new single connection
    assert len(new_net.all_connections) == 1
    assert new_net.all_connections[0].pre_obj is a
    assert new_net.all_connections[0].pre_slice == slice(2)
    assert new_net.all_connections[0].post_obj is d
    assert np.all(new_net.all_connections[0].transform == [[1, 0]])
