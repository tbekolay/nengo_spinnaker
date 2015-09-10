import nengo.tests.options

def pytest_addoption(parser):
    parser.addoption('--nengo', nargs='?', default=False, const=True,
                     help='Also run the Nengo test suite.')
    nengo.tests.options.pytest_addoption(parser)
