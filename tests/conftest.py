from __future__ import print_function

import fnmatch
import os
import sys

import nengo
from nengo.tests.conftest import *
from nengo.utils.testing import find_modules, allclose, load_functions

import nengo_spinnaker.simulator


# A monkey patch hack to solve a problem in Nengo master
def get_filename(self, ext=''):
    modparts = self.module_name.split('.')[1:]
    return "%s.%s.%s" % ('.'.join(modparts), self.function_name, ext)
nengo.utils.testing.Recorder.get_filename = get_filename


@pytest.fixture
def Simulator(request):
    if len(nengo_spinnaker.simulator.Simulator._open_simulators) > 0:
        # This shouldn't happen, but just in case it does...
        print("Sim still open, closing...", file=sys.stderr)
        nengo_spinnaker.simulator._close_open_simulators()
        # If something is still open, there's a bigger problem. Bail!
        assert len(nengo_spinnaker.simulator.Simulator._open_simulators) == 0
    request.addfinalizer(nengo_spinnaker.simulator._close_open_simulators)
    return nengo_spinnaker.simulator.Simulator


def pytest_runtest_setup(item):
    if not item.config.getvalue("nengo") and "Simulator" in item.funcargnames:
        pytest.skip("Nengo tests not requested.")


def pytest_generate_tests(metafunc):
    if "nl" in metafunc.funcargnames:
        metafunc.parametrize("nl", [nengo.LIF])
    if "nl_nodirect" in metafunc.funcargnames:
        metafunc.parametrize("nl_nodirect", [nengo.LIF])
