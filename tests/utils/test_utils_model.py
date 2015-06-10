import mock
import nengo
import pytest

from nengo_spinnaker.builder.builder import Model, ObjectPort, Signal
from nengo_spinnaker.operators import Filter
from nengo_spinnaker.utils.model import (
    remove_childless_filters, remove_sinkless_signals, _get_possible_merges,
    _get_equal_merges, _insert_merge_tree
)


def test_remove_sinkless_signals():
    """Signals with no sink should be removed."""
    # Create a netlist including some signals with no sinks, these signals
    # should be removed.
    o1 = mock.Mock(name="O1")
    o2 = mock.Mock(name="O2")

    # Create 4 signals (2 associated with connections, 2 not)
    cs1 = Signal(ObjectPort(o1, None), ObjectPort(o2, None), None)
    cs2 = Signal(ObjectPort(o1, None), [], None)
    ss1 = Signal(ObjectPort(o1, None), ObjectPort(o2, None), None)
    ss2 = Signal(ObjectPort(o1, None), [], None)

    # Create two mock connections
    c1 = mock.Mock(name="Connection 1")
    c2 = mock.Mock(name="Connection 2")

    # Create the model
    model = Model()
    model.extra_operators = [o1, o2]
    model.connections_signals = {c1: [cs1], c2: [cs2]}
    model.extra_signals = [ss1, ss2]

    # Remove sinkless signals
    remove_sinkless_signals(model)

    # Check that signals were removed as necessary
    assert model.connections_signals == {c1: [cs1]}
    assert model.extra_signals == [ss1]


def test_remove_childless_filters():
    """Filter operators which don't transmit to anything, and their incoming
    signals, can be removed.
    """
    # Create a netlist including some filters that do and don't transmit to
    # other objects, check that all the filters which don't connect to anything
    # are removed.
    #
    #          -S1---             F3
    #        /       \       S4  ^  \  S5
    #       /        v          /    v
    #     F1         O1 +S3-> F2     F5
    #      ^        /   |      \     ^
    #      \       /    |   S4  v   /  S6
    #       \-S2---     v        F4
    #                  O2
    #
    # F1 should remain, O1 and O2 should be untouched and F2..5 should be
    # removed.  S1 and S2 should be unchanged, S3 should have F2 removed from
    # its sinks and S4..6 should be removed entirely.

    # Create the filter operators
    f1 = mock.Mock(name="F1", spec=Filter)
    f2 = mock.Mock(name="F2", spec=Filter)
    f3 = mock.Mock(name="F3", spec=Filter)
    f4 = mock.Mock(name="F4", spec=Filter)
    f5 = mock.Mock(name="F5", spec=Filter)

    # The other operator
    o1 = mock.Mock(name="O1")
    o2 = mock.Mock(name="O2")

    # Create some objects which map to some of the operators
    oo1 = mock.Mock()
    of3 = mock.Mock()

    # Create the signals
    s1 = Signal(ObjectPort(f1, None), ObjectPort(o1, None), None)
    s2 = Signal(ObjectPort(o1, None), ObjectPort(f1, None), None)
    s3 = Signal(ObjectPort(o1, None), [ObjectPort(f2, None),
                                       ObjectPort(o2, None)], None)
    s4 = Signal(ObjectPort(f2, None), [ObjectPort(f3, None),
                                       ObjectPort(f4, None)], None)
    s5 = Signal(ObjectPort(f3, None), ObjectPort(f5, None), None)
    s6 = Signal(ObjectPort(f4, None), ObjectPort(f5, None), None)

    # Create some connections which map to the signals
    cs4 = mock.Mock()
    cs5 = mock.Mock()

    # Create the model
    model = Model()
    model.object_operators = {
        oo1: o1,
        of3: f3,
    }
    model.extra_operators = [f1, f2, f4, f5]
    model.connections_signals = {
        cs4: [s4],
        cs5: [s5],
    }
    model.extra_signals = [s1, s2, s3, s6]

    # Perform the optimisation
    remove_childless_filters(model)

    # Check that objects have been removed
    assert model.object_operators == {oo1: o1}
    assert model.extra_operators == [f1]
    assert model.connections_signals == {}
    assert model.extra_signals == [s1, s2, s3]
    assert [s.obj for s in s3.sinks] == [o2]


def test_get_possible_merges():
    """Test that we correctly identify sets of signals which can merged
    together to reduce fan-in.

        (filter 1, latching)-[s0]->\
              ----(filter 1)-[s1]->|
              ----(filter 1)-[s2]->+--> (o1)
              ----(filter 2)-[s3]->|
              ----(filter 2)-[s4]->|
        --(filter 2, port b)-[s5]->/

              ----(filter 1)-[s6]->\
              ----(filter 1)-[s7]->+--> (o2)
              ----(latching)-[s8]->/

    We should identify that the signals into `o1` require merging but that only
    (`s0`, `s1` and `s2`) and (`s3` and `s4`) can actually be merged.
    """
    # Create the objects
    o1 = mock.Mock()
    o2 = mock.Mock()

    # Create the filters
    filter1 = nengo.Lowpass(0.005)
    filter2 = nengo.Lowpass(0.03)

    # Create the signals
    s0 = Signal(ObjectPort(None, None), ObjectPort(o1, None), None,
                latching=True)
    s1 = Signal(ObjectPort(None, None), ObjectPort(o1, None), None)
    s2 = Signal(ObjectPort(None, None), ObjectPort(o1, None), None)
    s3 = Signal(ObjectPort(None, None), ObjectPort(o1, None), None)
    s4 = Signal(ObjectPort(None, None), ObjectPort(o1, None), None)
    s5 = Signal(ObjectPort(None, None), ObjectPort(o1, mock.Mock()), None)

    s6 = Signal(ObjectPort(None, None), ObjectPort(o2, None), None)
    s7 = Signal(ObjectPort(None, None), ObjectPort(o2, None), None)
    s8 = Signal(ObjectPort(None, None), ObjectPort(o2, None), None,
                latching=True)

    # Create some connections with which to associate the signals
    c1 = mock.Mock(spec=nengo.Connection)
    c1.synapse = filter1
    c2 = mock.Mock(spec=nengo.Connection)
    c2.synapse = filter2
    c3 = mock.Mock(spec=nengo.Connection)
    c3.synapse = filter1

    # Construct the model
    model = Model()
    model.connections_signals[c1] = [s0, s1, s2, s6, s7]
    model.connections_signals[c2] = [s3, s4, s5]
    model.extra_signals.append(s8)

    # Get the possible merges, check they are valid
    merges = _get_possible_merges(model, max_fan_in=3)
    assert len(merges) == 2
    for (obj, sigs) in merges:
        assert obj is o1
        assert set(sigs) in set([frozenset([s0, s1, s2]), frozenset([s3, s4])])


@pytest.mark.parametrize(
    "n_in, max_fan_in, expected",
    [(4, 4, [slice(0, 4)]),
     (2, 4, [slice(0, 2)]),
     (3, 4, [slice(0, 3)]),
     (5, 4, [slice(0, 2), slice(2, 5)]),
     (6, 4, [slice(0, 3), slice(3, 6)]),
     (7, 4, [slice(0, 3), slice(3, 7)]),
     (9, 4, [slice(0, 2), slice(2, 5), slice(5, 9)]),
     (10, 4, [slice(0, 3), slice(3, 6), slice(6, 10)]),
     (11, 4, [slice(0, 3), slice(3, 7), slice(7, 11)]),
     ]
)
def test_get_equal_merges(n_in, max_fan_in, expected):
    assert list(_get_equal_merges(n_in, max_fan_in)) == expected


def test_insert_merge_tree():
    """Tests that a merge tree is correctly inserted for some specified
    signals.
    
    We construct the model:

        --- (c1 => s0) ----\--> o2
        --- (c1 => s1) ----|
        --- (c1 => s2) ----+--> o1
        --- (c1 => s3) ----|
        --- (   => s4) ----|
        --- (c2 => s5) --- /

    And check that with a maximum fan in of 3 and specifying all (c1) signals
    we create:

        --- (c1 => s0) ----\-------------> o2
        --- (c1 => s1) -------> m1 --\
                                      +--> o1
        --- (c1 => s2) ----\          | 
        --- (c1 => s3) ----+--> m2 ---+
        --- (   => s4) ----/          |
                                      |
        --- (c2 => s5) --------------/  
    """
    # Create the terminating object
    o1 = mock.Mock(name="o1")
    o2 = mock.Mock(name="o2")

    # Create all the signals
    ss = [Signal(ObjectPort(None, None), ObjectPort(o1, None), None) for _ in
          range(6)]
    ss[0].sinks.append(ObjectPort(o2, None))

    # Create two connections
    c1 = mock.Mock(name="c1")
    c2 = mock.Mock(name="c2")

    # Create the model
    model = Model()
    model.connections_signals[c1] = ss[0:4]
    model.extra_signals.append(ss[4])
    model.connections_signals[c2] = [ss[5]]

    # Insert a merge tree
    _insert_merge_tree(model, o1, ss[0:5], max_fan_in=3)

    # Check that merge nodes were inserted
    assert len(model.extra_operators) == 2  # 2 merge nodes

    # Check that some extra signals were inserted
    assert len(model.connections_signals[c1]) == 8
    assert len(model.extra_signals) == 2
    assert len(model.connections_signals[c2]) == 1  # Unchanged

    # Check that each of the signals associated with c1 is either a signal we
    # had before, or that it is a new signal with appropriate start and end
    # points.
    for sig in model.connections_signals[c1]:
        if sig is ss[0]:
            assert o2 in [s.obj for s in sig.sinks]
        elif sig not in ss:
            assert sig.source.obj in model.extra_operators
            assert [o1] == [s.obj for s in sig.sinks]
        else:
            assert sig in ss
