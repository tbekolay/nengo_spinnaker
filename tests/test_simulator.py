import mock
from mock import patch
import nengo
import pytest

from nengo_spinnaker import SpiNNakerSimulator


@pytest.mark.parametrize("dt", [0.001, 0.002])
def test_init(dt):
    """Check that creating a simulator accesses the correct values from the RC
    file, creates an appropriate controller, boots the machine correctly (if
    necessary), and builds the model correctly.
    """
    # Create a mock network and mock config
    network = mock.Mock(name="network")
    config = network.config = dict()
    config[SpiNNakerSimulator] = mock.Mock()

    # Create a NodeIOController
    config[SpiNNakerSimulator].node_io = NodeIOController = \
        mock.Mock(name="NodeIOController")
    config[SpiNNakerSimulator].node_io_kwargs = {"arthur": "King"}
    nioc = NodeIOController.return_value = mock.Mock("nioc")
    nioc.builder_kwargs = {"spam": "a lot"}
    nioc.host_network = nengo.Network()

    # Create a mock RC file that will be read from
    def rc_get(section, parameter):
        assert section == "spinnaker_machine"
        assert parameter == "hostname"
        return "bob"

    def rc_getint(section, parameter):
        assert section == "spinnaker_machine"

        if parameter == "width":
            return 4
        elif parameter == "height":
            return 5
        else:  # pragma: no cover
            assert False, "Unexpected config request!"

    rc = mock.Mock(name="rc")
    rc.get = mock.Mock(wraps=rc_get)
    rc.getint = mock.Mock(wraps=rc_getint)

    # Create a mock Model class and instance
    Model = mock.Mock(name="Model", spec_set=[])
    model = Model.return_value = mock.Mock(name="model",
                                           spec_set=['build', 'dt',
                                                     'decoder_cache'])
    model.dt = dt

    # Create a mock Controller class and instance
    MachineController = mock.Mock(name="MachineController")
    controller = MachineController.return_value = mock.Mock(name="controller")

    # Create a test_and_boot patch
    def test_and_boot_fn(cn, hn, w, h):
        assert cn is controller
        assert hn == "bob"
        assert w == 4
        assert h == 5

    test_and_boot = mock.Mock(wraps=test_and_boot_fn)

    # Create the Simulator
    with \
            patch("nengo_spinnaker.simulator.rc", rc), \
            patch("nengo_spinnaker.simulator.MachineController",
                  MachineController), \
            patch("nengo_spinnaker.simulator.test_and_boot",
                  test_and_boot), \
            patch("nengo_spinnaker.simulator.Model", Model):
        sim = SpiNNakerSimulator(network, dt)

    # Check the simulator is sane
    assert sim.dt == dt
    assert sim.data == dict()
    assert sim.io_controller is nioc

    # Ensure that all the calls were correct
    assert rc.get.call_count == 1
    assert rc.getint.call_count == 2
    assert test_and_boot.call_count == 1

    assert Model.call_count == 1
    NodeIOController.assert_called_once_with(arthur="King")
    model.build.assert_called_once_with(network, spam="a lot")