"""Model Optimisations
"""
from __future__ import absolute_import

from collections import defaultdict
import itertools
import nengo.synapses
from six import iteritems, itervalues

from nengo_spinnaker.operators import Filter

MAX_FAN_IN = 8
"""The maximum number of signals which will be allowed to terminate on any
given object.
"""


def remove_sinkless_signals(model):
    """Remove all Signals which do not have any sinks from a
    :py:class:`~nengo_spinnaker.builder.Model`.
    """
    # Create a list of signals to remove by iterating through the signals which
    # are related to connections and finding any with no sinks.
    sinkless_signals = [(c, s) for c, ss in
                        iteritems(model.connections_signals) for s in ss
                        if len(s.sinks) == 0]

    # Now remove all sinkless signals
    for conn, sig in sinkless_signals:
        model.connections_signals.pop(conn)

    # Create a list of signals to remove by finding signals which are not
    # related to connections and which have no sinks.
    sinkless_signals = [s for s in model.extra_signals if len(s.sinks) == 0]

    # Now remove all sinkless signals
    for sig in sinkless_signals:
        model.extra_signals.remove(sig)


def remove_childless_filters(model):
    """Remove all Filter operators which do not transmit to anything from a
    :py:class:`~nengo_spinnaker.builder.Model`.

    Transmitting values to a filter which then doesn't forward them is a waste
    of network bandwidth. This method optimises out all filters which do not
    transmit to at least one other operator. This method will not remove cycles
    of Filters which have no output.
    """
    # We loop through removing filters that aren't connected to anything so
    # that we can deal with the case:
    #
    #     Filter -> Filter -> Filter -> Filter
    #
    # Which should result in the removal of all of the above filters. We break
    # as soon as there are no more childless filters.
    while True:
        # Childless Filters are those in EITHER the dictionary of object
        # operators or the set of extra operators which have no outgoing
        # signals.
        childless_filters = [
            (k, v) for k, v in
            itertools.chain(iteritems(model.object_operators),
                            ((None, v) for v in model.extra_operators)) if
            (isinstance(v, Filter) and  # It's a filter
             model.get_signals_connections_from_object(v) == {})  # Unconnected
        ]

        if not childless_filters:
            # If there are no childless filters then we have nothing to do
            break

        # Remove each of the childless filters in turn.
        for obj, filt in childless_filters:
            # Remove the filter from the list of sinks of each of the signals
            # which target it.
            for sig in model.all_signals():
                # Prepare and remove sinks which target the object we're
                # removing.
                sinks = [s for s in sig.sinks if s.obj is filt]
                for sink in sinks:
                    sig.sinks.remove(sink)

            # Remove the Filter operator itself
            if obj is None:
                model.extra_operators.remove(filt)
            else:
                model.object_operators.pop(obj)

        # Removing filters from the lists of sinks of signals may produce
        # signals which have no sinks, we should remove these as it will allow
        # us to find further filters with no children.
        remove_sinkless_signals(model)


def _get_possible_merges(model, max_fan_in=MAX_FAN_IN):
    """Get a list of signals which could be merged in merge trees to reduce the
    fan-ins in the model.

    Returns
    -------
    [(object, [signal, ...]), ...]
        Pairs of objects with respective incoming signals which could be merged
        to reduce the fan in.
    """
    # Build up a dict {obj: {port: {filter: [signals, ...]}}} and a dict {obj:
    # fan_in} which will allow us to identify cases where merge trees are
    # necessary and which signals may be merged.
    opfs = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    obj_fanin = defaultdict(lambda: 0)

    # Create an iterator of all signals combined with related connections, or
    # None if there is no associated connection.
    def all_signals_and_connections(model):
        # Build a dictionary which maps signals to all of the connections with
        # which they are associated. Signals from the list of extra signals are
        # associated with the empty list.
        signal_connections = defaultdict(list)

        # All signals associated with connections
        for conn, sigs in iteritems(model.connections_signals):
            for sig in sigs:
                signal_connections[sig].append(conn)

        # All other signals
        for sig in model.extra_signals:
            # Accessing the key is enough to ensure that it exists
            signal_connections[sig]

        # Now we just return an iterator over this dictionary, this will yield
        # (signal, [connection, ...]) pairs.
        return iteritems(signal_connections)

    # For every (signal, connections) pair in the model increment the fan-in of
    # all its sinks and include the signal in the dictionary of possible
    # merges.
    for signal, connections in all_signals_and_connections(model):
        # For every sink in the signal we perform the same processing
        for sink in signal.sinks:
            # First we increment the fan in to the sink object
            obj_fanin[sink.obj] += 1

        # Now create a hashable representation of the synapse, if we can't then
        # we just skip to the next signal.
        if len(connections) == 0:
            # We treat no connections as if there were a `None` synapse
            synapse = None
        elif len(connections) == 1:
            # We grab a representation of the synapse as a tuple and use this
            conn = connections[0]
            if conn.synapse is None:
                synapse = conn.synapse
            elif isinstance(conn.synapse, nengo.synapses.LinearFilter):
                synapse = (type(conn.synapse),
                           tuple(conn.synapse.num),
                           tuple(conn.synapse.den))
            else:
                raise NotImplementedError("Unknown synapse type")
        else:
            # It's not clear what we should do, so we do nothing
            continue

        # Now we include the signal in the dictionary of possible merges
        for sink in signal.sinks:
            opfs[sink.obj][sink.port][synapse].append(signal)

    # To finish off we return a list of the objects with a fan-in greater than
    # the maximum fan-in and their possible merges.
    return list(
        # Pairs of objects and signals, extracted by ignoring ports and filters
        # in the large dictionary we created above.
        (obj, sigs) for obj, pfs in iteritems(opfs)
        for fs in itervalues(pfs) for sigs in itervalues(fs)
        # Provided there's more than 1 signal and the object has a fan-in
        # greater than the max we include this list of signals and the object.
        if len(sigs) > 1 and obj_fanin[obj] > max_fan_in
    )


def _get_equal_merges(n_in, max_fan_in=MAX_FAN_IN):
    """Generate slices which can be used to collect signals into merge nodes.

    Parameters
    ----------
    n_in : int
        Number of signals to combine.
    max_fan_in : int
        Maximum allowed fan in.
    """
    # Special case, we have less than the maximum fan in, just yield
    if n_in <= max_fan_in:
        yield slice(0, n_in)
    else:
        # Calculate the remainder, then split the missing items over two
        # entries
        n = 0
        r = n_in % max_fan_in
        if r != 0:
            r += max_fan_in
            yield slice(0, (r // 2))
            yield slice((r // 2), (r // 2) + -(-r // 2))

        # Yield the remaining merge lengths
        n = r
        n_in -= r
        while n_in > 0:
            # Next slice
            length = min((max_fan_in, n_in))
            yield slice(n, n+length)
            n_in -= length
            n += length


def _insert_merge_tree(model, obj, signals, max_fan_in=MAX_FAN_IN):
    """Insert a signal merge tree into the model.

    This will insert new operators into the model, will disconnect each of the
    listed signals from the given object and attach them to the merge objects
    and will connect the merge objects to the original object.  This will
    happen repeatedly until the merge tree obeys the max fan in requirement.
    """
