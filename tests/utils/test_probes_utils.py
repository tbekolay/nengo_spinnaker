import nengo
from nengo_spinnaker.utils.probes import probe_target


def test_probe_target_no_slice():
    # Create a network containing a probe
    with nengo.Network():
        a = nengo.Ensemble(100, 1)
        p = nengo.Probe(a)

    assert probe_target(p) is a


def test_probe_target_with_slice():
    # Create a network containing a probe
    with nengo.Network():
        a = nengo.Ensemble(100, 2)
        p = nengo.Probe(a[0])

    assert probe_target(p) is a
