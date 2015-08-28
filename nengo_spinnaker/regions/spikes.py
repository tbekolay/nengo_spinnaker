"""Regions to store information about spike transmission and utilities to
construct them.
"""
import numpy as np
from six import iteritems
import struct

from .region import Region
from .filters import add_filter, FilterRegion
from ..utils import type_casts as tp


def make_synaptic_regions(target_ens, signals_and_connections, model):
    """Construct the regions required for representing synaptic routing,
    synaptic filtering and synaptic weight matrices.

    Parameters
    ----------
    target_ens : :py:class:`nengo.Ensemble`
        Ensemble to retrieve the synaptic regions for.

    Returns
    -------
    FilterRegion
        Region specifying the synaptic filters.
    SynapticWeightMatrixRegion
        Synaptic weight matrices.
    SynapticRoutingRegion
        Synaptic routing information.
    """
    filters = list()
    filter_indices = list()
    weight_matrices = list()
    synaptic_routing = list()

    # For each signal and connection which target the given Ensemble
    for signal, connections in iteritems(signals_and_connections):
        for conn in connections:
            # If this connection doesn't target the desired ensemble then skip
            # it.
            if conn.post_obj is not target_ens.neurons:
                continue

            # Otherwise add the information pertaining to this connection to
            # what we already have.
            # Add the filter to the list of filters
            index = add_filter(filters, conn, latching=False,
                               width=0, minimise=True)

            # Add a routing entry for the weight matrix
            synaptic_routing.append((signal.keyspace, len(filter_indices)))

            # Get the synaptic weight matrix and add it to the list of weight
            # matrices.
            matrix = model.params[conn].transform.T
            weight_matrices.append(matrix)

            # Add the right number of filter indices to the list
            filter_indices.extend([index] * matrix.shape[0])

    # Combine the weight matrices into a single block
    weight_matrices = np.vstack(weight_matrices)

    # Finally create the regions
    filter_region = FilterRegion(filters, model.dt, sliced_filters=True)
    wm_region = SynapticWeightMatrixRegion(filter_indices, weight_matrices)
    wm_routing = SynapticRoutingRegion(
        synaptic_routing,
        filter_routing_tag=model.keyspaces.filter_routing_tag
    )

    return filter_region, wm_region, wm_routing


class SynapticWeightMatrixRegion(Region):
    """Holds a synaptic weight matrix.

    The first word of each row (when written out) will indicate the synaptic
    filter which should be used with the row.  The remaining entries will be
    the synaptic weights themselves.

    Parameters
    ----------
    filter_indices : list or array
        Integer indices of the filter that each row should be processed with.
    weight_matrix : NumPy array
        Full weight matrix.
    """
    def __init__(self, filter_indices, weight_matrix):
        rows, cols = weight_matrix.shape

        # Assert the sizes are correct
        filter_indices = np.array(filter_indices, dtype=np.int32)
        assert filter_indices.size == rows

        # Combine the filter indices and the weight matrix into one matrix.
        # First form the filter indices into a column
        filter_indices.shape = (rows, 1)

        # Then convert the weight matrix into fixed point form
        weight_matrix = tp.np_to_fix(weight_matrix)

        # Finally, stack horizontally and store
        self.data = np.zeros(shape=(rows, cols+1), dtype=np.int32)
        self.data[:, 0] = filter_indices[:, 0]
        self.data[:, 1:] = weight_matrix

    def _extend_slice(self, vertex_slice):
        """Convert the vertex slice into an appropriate slice into the columns
        of the weight matrix.
        """
        indices = [0] + list(range(vertex_slice.start + 1,
                                   vertex_slice.stop + 1))
        return (slice(None), indices)

    def sizeof(self, vertex_slice):
        """Get the memory requirements of this region as a number of bytes."""
        return self.data[self._extend_slice(vertex_slice)].nbytes

    def write_subregion_to_file(self, fp, vertex_slice):
        """Write a slice of this matrix to the file-like."""
        # Perform the write directly
        fp.write(self.data[self._extend_slice(vertex_slice)].tostring())


class SynapticRoutingRegion(Region):
    """Represents the routes which allow a spike packet to activate a row in a
    synaptic weight matrix.

    When written out each entry consists of four words:

        1. Routing key
        2. Routing mask
        3. Index into the rows of the weight matrix (`block_offset`)
        4. Mask to extract the specific row of the weight matrix
           (`neuron_mask`)

    A spike packet "matches" a row if `(spike_key & routing_mask) ==
    routing_key`.  If a match occurs then row `block_offset + (spike_key &
    neuron_mask)` should be retrieved from the weight matrix.

    In Python this data is represented as a list of pairs of bitfields and
    block offsets.  The routing key and mask are extracted using the routing
    tag and the neuron mask is extracted using the index field..

    Attributes
    ----------
    keyspaces_routes : [(Bitfield, int), ...]
        Pairs of bitfields (keyspaces) and integers which index into rows of a
        synaptic weight matrix.
    """
    def __init__(self, keyspace_routes, filter_routing_tag="filter_routing",
                 index_field="index"):
        """Create a new synaptic routing region."""
        # Store the parameters
        self.keyspace_routes = keyspace_routes
        self.filter_routing_tag = filter_routing_tag
        self.index_field = index_field

    def sizeof(self, *args):
        """Get the memory requirements of this region as a number of bytes."""
        # 1 word + 4 words per entry
        return 4 * (1 + 4*len(self.keyspace_routes))

    def write_subregion_to_file(self, fp, *args, **kwargs):
        """Write the routing region to a file-like object."""
        data = bytearray(self.sizeof())

        # Write the number of entries
        struct.pack_into("<I", data, 0, len(self.keyspace_routes))

        # Write each entry in turn
        for i, (ks, block_offset) in enumerate(self.keyspace_routes):
            struct.pack_into("<4I", data, 4 + 16*i,
                             ks.get_value(tag=self.filter_routing_tag),
                             ks.get_mask(tag=self.filter_routing_tag),
                             block_offset,
                             ks.get_mask(field=self.index_field))

        # Write to file
        fp.write(data)
