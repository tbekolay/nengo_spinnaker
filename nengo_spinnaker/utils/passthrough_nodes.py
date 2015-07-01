"""Collection of tools necessary to modify networks to remove passthrough
Nodes.
"""
import nengo
from nengo.synapses import LinearFilter
from nengo.utils.builder import find_all_io, full_transform
import numpy as np
from numpy.polynomial.polynomial import polymul

from .probes import probe_target


def add_passthrough_node_to_probe_connections(network):
    """Adds new connections to the network to connect passthrough Nodes probes
    which target them.
    """
    # For all of the probes in the network, if any of them refer to a
    # passthrough Node then add a connection from the passthrough Node to the
    # probe.
    for probe in network.all_probes:
        obj = probe_target(probe)

        # If the target is a passthrough Node then add a connection
        if isinstance(obj, nengo.Node) and obj.output is None:
            with network:
                nengo.Connection(probe.target, probe, seed=probe.seed,
                                 synapse=probe.synapse)


def combine_lti_synapses(a, b):
    """Combine two LTI filters."""
    # Assert that both the synapses are LTI synapses
    assert (isinstance(a, nengo.synapses.LinearFilter) and
            isinstance(b, nengo.synapses.LinearFilter))

    # Combine
    return nengo.synapses.LinearFilter(polymul(a.num, b.num),
                                       polymul(a.den, b.den))


def remove_passthrough_nodes(network):
    """Return a new network with all the passthrough Nodes removed and a
    mapping from new connections to connections the decoders of which they can
    use.
    """
    # Get a list of the connections and a complete map of objects to inputs and
    # outputs.
    conns = list(network.all_connections)
    inputs, outputs = find_all_io(conns)

    # Prepare to create a map of which connections can just use the decoders
    # from an earlier connection.
    connection_decoders = dict()

    # Create a new (flattened) network containing all elements from the
    # original network apart from passthrough Nodes.
    with nengo.Network() as m:
        # Add all of the original ensembles
        for ens in network.all_ensembles:
            m.add(ens)

        # Add all of the original probes
        for probe in network.all_probes:
            m.add(probe)

    # For all the Nodes, if the Node is not a passthrough Node we add it as
    # usual - otherwise we combine remove it and multiply together its
    # input and output connections.
    for node in network.all_nodes:
        if node.output is not None:
            with m:
                m.add(node)
            continue

        # Remove the original connections associated with this passthrough
        # Node from both the list of connections but also the lists
        # associated with their pre- and post- objects.
        conns_in = list(inputs[node])
        conns_out = list(outputs[node])

        for c in conns_in:
            conns.remove(c)
            outputs[c.pre_obj].remove(c)

        for c in conns_out:
            conns.remove(c)
            inputs[c.post_obj].remove(c)

        # For every outgoing connection
        for out_conn in outputs[node]:
            # For every incoming connection
            for in_conn in inputs[node]:
                use_pre_slice = in_conn.function is not None

                # Create a new transform for the combined connections.  If
                # the transform is zero then we don't bother adding a new
                # connection and instead move onto the next combination. If the
                # in connection doesn't have a function then we include the
                # pre-slice in the transform, otherwise we ignore it.
                transform = np.dot(
                    full_transform(out_conn),
                    full_transform(in_conn, slice_pre=not use_pre_slice)
                )

                if np.all(transform == 0.0):
                    continue

                # We determine if we can combine the synapses.  If we can't
                # we raise an error because we can't do anything at the
                # moment.
                if out_conn.synapse is None or in_conn.synapse is None:
                    # Trivial combination of synapses
                    new_synapse = out_conn.synapse or in_conn.synapse
                elif (isinstance(in_conn.synapse, LinearFilter) and
                        isinstance(out_conn.synapse, LinearFilter)):
                    # Combination of LTI systems
                    print("Combining synapses of {} and {}".format(
                        in_conn, out_conn))
                    new_synapse = combine_lti_synapses(in_conn.synapse,
                                                       out_conn.synapse)
                else:
                    # Can't combine these filters
                    raise NotImplementedError(
                        "Can't combine synapses of types {} and {}".format(
                            in_conn.synapse.__class__.__name__,
                            out_conn.synapse.__class__.__name__
                        )
                    )

                # Create a new connection that combines the inputs and outputs.
                new_c = nengo.Connection(
                    in_conn.pre if use_pre_slice else in_conn.pre_obj,
                    out_conn.post_obj,
                    function=in_conn.function,
                    synapse=new_synapse,
                    transform=transform,
                    add_to_container=False
                )

                # Add this connection to the list of connections to add to the
                # model and the lists of outgoing and incoming connections for
                # objects.
                conns.append(new_c)
                inputs[new_c.post_obj].append(new_c)
                outputs[new_c.pre_obj].append(new_c)

                # Determine which decoders should be used for this connection
                # if the pre object is an ensemble.
                if isinstance(in_conn.pre_obj, nengo.Ensemble):
                    x = in_conn
                    while x in connection_decoders:
                        x = connection_decoders[x]
                    connection_decoders[new_c] = x

    # Add all the connections
    with m:
        for c in conns:
            m.add(c)

    # Return the new network and map of connections
    return m, connection_decoders
