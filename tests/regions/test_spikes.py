import nengo
import numpy as np
import pytest
import struct
import tempfile

from nengo_spinnaker.builder.builder import BuiltConnection, Model, Signal
from nengo_spinnaker.utils import type_casts as tp
from nengo_spinnaker.utils.keyspaces import KeyspaceContainer
from nengo_spinnaker.regions import spikes


def test_make_synaptic_regions():
    """Test the synaptic regions can be correctly constructed from existing
    connections and signals.
    """
    # Create a model with some spike based connections in it.
    with nengo.Network():
        # Create three ensembles
        a = nengo.Ensemble(5, 1)
        b = nengo.Ensemble(6, 1)
        c = nengo.Ensemble(7, 1)

        # Create some spike connections
        # A->B and A->C will share a signal
        a_b_weights = np.random.normal(size=(a.n_neurons, b.n_neurons)).T
        a_b = nengo.Connection(a.neurons, b.neurons,
                               transform=a_b_weights)
        a_c_weights = np.random.normal(size=(a.n_neurons, c.n_neurons)).T
        a_c = nengo.Connection(a.neurons, c.neurons,
                               transform=a_c_weights, synapse=0.01)

        # B->C
        b_c_weights = np.random.normal(size=(b.n_neurons, c.n_neurons)).T
        b_c = nengo.Connection(b.neurons, c.neurons,
                               transform=b_c_weights, synapse=0.02)

    # Create two signals to combine with the connections
    ksc = KeyspaceContainer()
    ks = ksc["nengo.spikes"]
    sig_a = Signal(None, [None], ks(object=0))
    sig_b = Signal(None, [None], ks(object=1))

    # Create an empty model to contain all of the parameters
    model = Model(keyspaces=ksc)
    model.params[a_b] = BuiltConnection(None, None, a_b_weights, None)
    model.params[a_c] = BuiltConnection(None, None, a_c_weights, None)
    model.params[b_c] = BuiltConnection(None, None, b_c_weights, None)

    # Create the synaptic regions for B, this should contain one weight matrix
    # because there is only one incoming connection.
    b_filters, b_wm, b_routing = spikes.make_synaptic_regions(
        b, {sig_a: [a_b, a_c]}, model
    )

    # B should have only one filter
    assert len(b_filters.filters) == 1
    assert b_filters.dt == model.dt
    assert b_filters.sliced

    # B should have one synaptic matrix equal to a_b_weights
    assert b_wm.data.shape[0] == a.n_neurons
    assert np.all(b_wm.data[:, 1:] == tp.np_to_fix(a_b_weights.T))
    assert np.all(b_wm.data[:, 0] == 0)

    # B should have one synaptic routing entry
    assert b_routing.keyspace_routes == [(sig_a.keyspace, 0)]

    # Create the synaptic weight regions for C, this should contain two weight
    # matrices because there are two incoming connections.
    c_filters, c_wm, c_routing = spikes.make_synaptic_regions(
        c, {sig_a: [a_b, a_c], sig_b: [b_c]}, model
    )

    # C should have two filters
    assert len(c_filters.filters) == 2

    # C should have two synaptic routing entries, and these should match the
    # expected weight matrices.
    assert len(c_routing.keyspace_routes) == 2
    r0, r1 = c_routing.keyspace_routes

    # Get the weight matrices
    wm0 = c_wm.data[slice(r0[1], r1[1])]
    wm1 = c_wm.data[r1[1]:]

    if r0[0] == sig_a.keyspace:
        assert np.all(wm0[:, 1:] == tp.np_to_fix(a_c_weights.T))
        assert np.all(wm1[:, 1:] == tp.np_to_fix(b_c_weights.T))
    else:
        assert np.all(wm0[:, 1:] == tp.np_to_fix(b_c_weights.T))
        assert np.all(wm1[:, 1:] == tp.np_to_fix(a_c_weights.T))

    # Check the filter indices
    assert np.all(wm0[:, 0] == 0)
    assert np.all(wm1[:, 0] == 1)


@pytest.mark.parametrize(
    "vertex_slice, n_neurons",
    [(slice(0, 10), 10),
     (slice(10, 25), 15),
     ]
)
def test_synaptic_weight_matrix_region(vertex_slice, n_neurons):
    """Create a synaptic weight matrix region, ensure that it reports size
    correctly and can be written out to memory correctly.
    """
    # Use an identity matrix as a synaptic weight matrix as it's pretty easy to
    # determine what's going on.
    full_matrix = np.eye(50)

    # We require filter indices as a list or vector
    filter_indices = list(range(full_matrix.shape[1]))

    # Create the region
    swmr = spikes.SynapticWeightMatrixRegion(filter_indices, full_matrix)

    # Check that the size is reported correctly, this should be 1 word * (1 +
    # number of rows) * number of columns where the number of columns is
    # determined by the slicing.
    assert swmr.sizeof(vertex_slice) == 4 * (1 + n_neurons) * 50

    # Try writing the synaptic weight matrices out to file to ensure that the
    # data is written correctly.
    with tempfile.TemporaryFile() as fp:
        # Write out the data
        swmr.write_subregion_to_file(fp, vertex_slice)

        # Read back all of the data and reconstruct it as a Numpy array of
        # integers.
        fp.seek(0)
        data = np.frombuffer(fp.read(), dtype=np.int32)
        data.shape = (50, 1 + n_neurons)

        # Check the filter indices are correct
        assert np.all(data[:, 0] == np.array(filter_indices, dtype=np.int32))

        # Assert the weights are correct
        assert np.all(data[:, 1:] ==
                      tp.np_to_fix(full_matrix)[:, vertex_slice])


def test_synaptic_routing_region():
    """Create a synaptic routing region, ensure that it correctly reports its
    size and writes out correctly.
    """
    # Get a keyspace for spike packets
    ksc = KeyspaceContainer()
    ks = ksc["nengo.spikes"]

    # Create several entries for the routing region
    entries = [
        (ks(object=1, cluster=0), 0),
        (ks(object=5, cluster=0), 0),
        (ks(object=7, cluster=0), 0),
    ]

    # Create a keyspace which will set several fields to be non-zero
    ks(object=1, cluster=100, index=255)
    ksc.assign_fields()

    # Create the routing region
    srr = spikes.SynapticRoutingRegion(entries)

    # Check that the reported size of the region is correct - expecting 4 words
    # per entry plus 4 words of count.
    assert srr.sizeof() == (len(entries) * 4 + 1) * 4

    # Write the region out into memory and assert that it is written out
    # correctly
    with tempfile.TemporaryFile() as fp:
        # Perform the write
        srr.write_subregion_to_file(fp, slice(None))

        # Seek back to the start and process the data
        fp.seek(0)
        assert struct.unpack("<I", fp.read(4))[0] == len(entries)

        # Read each entry in turn and check that it is correct
        for (ks, block_offset) in entries:
            # Unpack the data from the file
            key, mask, read_offset, neuron_mask = \
                struct.unpack("<4I", fp.read(16))

            # Check these values are sane
            assert key == ks.get_value(tag=ksc.filter_routing_tag)
            assert mask == ks.get_mask(tag=ksc.filter_routing_tag)
            assert read_offset == block_offset
            assert neuron_mask == ks.get_mask(field="index")
