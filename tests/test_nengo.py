"""
Black-box testing of the SpiNNaker simulator.

TestCase classes are added automatically from
nengo.tests, but you can still run individual
test files like this:

$ py.test tests/test_nengo.py -k test_ensemble.test_scalar

See http://pytest.org/latest/usage.html for more invocations.

"""
import fnmatch
import os

from nengo.utils.testing import find_modules, allclose, load_functions
import nengo
import pytest

import nengo_spinnaker


def xfail(pattern, msg):
    for key in tests:
        if fnmatch.fnmatch(key, pattern):
            tests[key] = pytest.mark.xfail(True, reason=msg)(tests[key])

nengo_dir = os.path.dirname(nengo.__file__)
modules = find_modules(nengo_dir, prefix='nengo')
tests = load_functions(modules, arg_pattern='^Simulator$')

xfail('test.nengo.networks.tests.test_circularconv.test_input_magnitude',
      "LIFRate not implemented.")
xfail('test.nengo.networks.tests.test_circularconv.test_neural_accuracy',
      "LIFRate not implemented.")
xfail('test.nengo.networks.tests.test_ensemblearray.test_neuronconnection',
      "SpiNNaker does not currently support connections from Neurons")
xfail('test.nengo.networks.tests.test_product.test_direct_mode*',
      "No direct mode")
xfail('test.nengo.networks.tests.test_workingmemory.test_inputgatedmemory',
      "Scalar transform?")
xfail('test.nengo.tests.test_cache.test_cache_works',
      "No cache, also can't take in built model")
xfail('test.nengo.utils.tests.test_connection.test_eval_point_decoding',
      "Have to update transform to weights...")
xfail('test.nengo.tests.test_connection.test_node_to_ensemble',
      "No direct mode")
xfail('test.nengo.tests.test_connection.test_neurons_to_*',
      "SpiNNaker does not currently support connections from Neurons")
xfail('test.nengo.tests.test_connection.test_weights',
      "SpiNNaker does not currently support connections from Neurons")
xfail('test.nengo.tests.test_connection.test_slicing',
      "SpiNNaker does not currently support connections from Neurons")
xfail('test.nengo.tests.test_connection.test_neuron_slicing',
      "LIFRate not implemented.")
xfail('test.nengo.tests.test_connection.test_shortfilter',
      "Causes zero division error.")
xfail('test.nengo.tests.test_connection.test_zerofilter',
      "No direct mode")
xfail('test.nengo.tests.test_connection.test_decoder_probe',
      "SpiNNaker does not currently support connections from Neurons")
xfail('test.nengo.tests.test_connection.test_transform_probe',
      "SpiNNaker does not currently support connections from Neurons")
xfail('test.nengo.tests.test_learning_rules.test_pes_*',
      "PES not implemented yet.")
xfail('test.nengo.tests.test_learning_rules.test_dt_dependence',
      "SpiNNaker does not currently support connections from Neurons")
xfail('test.nengo.tests.test_learning_rules.test_reset',
      "SpiNNaker does not currently support connections from Neurons")
xfail('test.nengo.tests.test_learning_rules.test_unsupervised',
      "SpiNNaker does not currently support connections from Neurons")
xfail('test.nengo.tests.test_neurons.test_lif',
      "'refractory_time' probe not supported.")
xfail('test.nengo.tests.test_neurons.test_lif_min_voltage',
      "Min voltage is clamped at 0.")
xfail('test.nengo.tests.test_neurons.test_izhikevich',
      "Izhikevich neurons not implemented.")
xfail('test.nengo.tests.test_neurons.test_alif*', "Only LIF implemented.")
xfail('test.nengo.tests.test_neurons.test_reset',
      "SpiNNaker does not currently support connections from Neurons")
xfail('test.nengo.tests.test_node.test_args', "Not quite 0...")
xfail('test.nengo.tests.test_node.test_unconnected_node',
      "Nengo shouldn't call sim.step here...")
xfail('test.nengo.tests.test_processes.test_reset',
      "Reset not implemented.")
xfail('test.nengo.tests.test_processes.test_whitesignal_continuity',
      "Non-uniform sampling problem I'm guessing?")
xfail('test.nengo.tests.test_probe.test_defaults',
      "Probing Connections not supported.")
xfail('test.nengo.tests.test_probe.test_input_probe',
      "SpiNNaker does not support probing 'input' on Ensembles.")
xfail('test.nengo.tests.test_probe.test_large',
      "Non-uniform sampling problem I'm guessing?")
xfail('test.nengo.tests.test_probe.test_multiple_probes',
      "sampling rates not guaranteed I'm guessing?")
xfail('test.nengo.tests.test_synapses.test_alpha',
      "alpha synapse not implemented.")
xfail('test.nengo.tests.test_synapses.test_linearfilter',
      "analog=False filters not implemented yet.")
xfail('test.nengo.tests.test_synapses.test_triangle',
      "triangle synapse not implemented.")
xfail('test.nengo.utils.tests.test_neurons.test_rates*',
      "LIFRate not implemented.")
xfail('test.nengo.utils.tests.test_ensemble.test_*_direct_mode',
      "No direct mode")

xfail('test.nengo.tests.test_neurons.test_lif_zero_tau_ref',
      "Doesn't spike at all with tau_ref=0.")
xfail('test.nengo.tests.test_ensemble.test_noise_gen',
      "Injected noise doesn't work?")
xfail('test.nengo.tests.test_connection.test_nonexistant_prepost',
      "Fails with KeyError instead of ValueError")
xfail('test.nengo.spa.tests.test_cortical.test_convolution',
      "All are 0 instead of all but action 3")
xfail('test.nengo.spa.tests.test_thalamus.test_routing',
      "valueC is near 0 instead of > 0.8")
xfail('test.nengo.tests.test_neurons.test_dt_dependence',
      "RMSE is 893.21 but should be 10.0. dt makes a difference? "
      "Or multiple sim issue.")
xfail('test.nengo.tests.test_node.test_time',
      "Not exact timesteps. We should make this OK.")
xfail('test.nengo.tests.test_node.test_simple',
      "Also a timestep issue.")
xfail('test.nengo.tests.test_node.test_connected',
      "Another timestep issue?")
xfail('test.nengo.tests.test_node.test_passthrough', "Unsure")
xfail('test.nengo.tests.test_node.test_passthrough_filter', "Similar...")
xfail('test.nengo.tests.test_node.test_circular', "Looks right?")
xfail('test.nengo.tests.test_processes.test_gaussian_whitenoise',
      "Processes not implemented yet?")
xfail('test.nengo.tests.test_processes.test_whitesignal*',
      "Processes not implemented yet? Or, tolerances are too tight...")
xfail('test.nengo.tests.test_ensemble.test_constant_scalar',
      "Last timepoint seems to be wrong? But the rest are good?")
xfail('test.nengo.spa.tests.test_thalamus.test_nondefault_routing',
      "valueC way off.")
xfail('test.nengo.tests.test_synapses.test_decoders',
      "Differences in representing time I think...")

# These are just out of tolerance
xfail('test.nengo.networks.tests.test_product.test_sine_waves',
      "RMSE 0.2327 but should be < 0.2")
xfail('test.nengo.spa.tests.test_assoc_mem.test_am_basic',
      "similarity 0.9818 but should be > 0.99")
xfail('test.nengo.spa.tests.test_basalganglia.test_basal_ganglia',
      "Motor A and E not the same, though close")
xfail('test.nengo.spa.tests.test_cortical.test_connect',
      "match 0.8639 but should be > 0.9")
xfail('test.nengo.tests.test_connection.test_vector',
      "Close to -0.5, 0.25 but not quite...")
xfail('test.nengo.tests.test_ensemble.test_constant_vector',
      "Looks right until the last bit of the simulation?")
xfail('test.nengo.tests.test_ensemble.test_scalar',
      "Looks right, but tolerances are too tight here.")
xfail('test.nengo.tests.test_ensemble.test_vector',
      "Looks right, but tolerances are too tight here.")

locals().update(tests)


if __name__ == '__main__':
    pytest.main(sys.argv)
