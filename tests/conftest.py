import fnmatch
import os

import nengo
from nengo.tests.conftest import *
from nengo.utils.testing import find_modules, allclose, load_functions

import nengo_spinnaker


# A monkey patch hack to solve a problem in Nengo master
def get_filename(self, ext=''):
    modparts = self.module_name.split('.')[1:]
    return "%s.%s.%s" % ('.'.join(modparts), self.function_name, ext)
nengo.utils.testing.Recorder.get_filename = get_filename


@pytest.fixture
def Simulator():
    return nengo_spinnaker.Simulator


def pytest_generate_tests(metafunc):
    if "nl" in metafunc.funcargnames:
        metafunc.parametrize("nl", [nengo.LIF])
    if "nl_nodirect" in metafunc.funcargnames:
        metafunc.parametrize("nl_nodirect", [nengo.LIF])
