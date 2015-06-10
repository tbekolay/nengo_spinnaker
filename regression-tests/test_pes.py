import logging
logging.basicConfig(level=logging.INFO)

import math
import nengo
import nengo_spinnaker
import numpy as np

from nengo.processes import WhiteSignal

def test_pes():
    model = nengo.Network()
    with model:
        inp = nengo.Node(WhiteSignal(60, high=5), size_out=1)
        pre = nengo.Ensemble(60, dimensions=1, label="pre")
        nengo.Connection(inp, pre)

        inp_p = nengo.Probe(inp)
        pre_p = nengo.Probe(pre, synapse=0.01)

        post = nengo.Ensemble(60, dimensions=1, label="post")
        conn = nengo.Connection(pre, post, function=lambda x: np.random.random())

        error = nengo.Ensemble(60, dimensions=1, label="error")

        # Error = actual - target = post - pre
        nengo.Connection(post, error)
        nengo.Connection(pre, error, transform=-1)

        # Add the learning rule to the connection
        conn.learning_rule_type = nengo.PES()

        # Connect the error into the learning rule
        nengo.Connection(error, conn.learning_rule)

        post_p = nengo.Probe(post, synapse=0.01)

        nengo_spinnaker.add_spinnaker_params(model.config)
        model.config[inp].function_of_time = True

        sim = nengo_spinnaker.Simulator(model)
        with sim:
            sim.run(10.0)

    # Read data
    pre_data = sim.data[pre_p]
    post_data = sim.data[post_p]

    # Calculate squared error
    error = np.power(pre_data - post_data, 2.0)

    # After about 4s, learning should have converged
    learnt_times = sim.trange() > 4.0

    # Check that mean error in this period is less than 0.1
    assert math.sqrt(np.mean(error[learnt_times])) < 0.1


if __name__=="__main__":
    test_pes()