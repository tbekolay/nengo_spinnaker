import logging
logging.basicConfig(level=logging.INFO)

import nengo
import nengo_spinnaker
import numpy as np
import matplotlib.pyplot as plt

from nengo.processes import WhiteSignal

spinnaker = True

model = nengo.Network()
with model:
    learning_rates = [(1e-4, "r"), (0.5e-4, "g")]
    
    inp = nengo.Node(WhiteSignal(60, high=5), size_out=2)
    pre = nengo.Ensemble(60, dimensions=2, label="pre")
    nengo.Connection(inp, pre)
    
    inp_p = nengo.Probe(inp)
    pre_p = nengo.Probe(pre, synapse=0.01)
    
    post_p = []
    for learning_rate, _ in learning_rates:
        post = nengo.Ensemble(60, dimensions=2, label="post")
        conn = nengo.Connection(pre, post, function=lambda x: np.random.random(2))
        
        error = nengo.Ensemble(60, dimensions=2, label="error")
     
        # Error = actual - target = post - pre
        nengo.Connection(post, error)
        nengo.Connection(pre, error, transform=-1)
        
        # Add the learning rule to the connection
        conn.learning_rule_type = nengo.PES(learning_rate=learning_rate)
        
        # Connect the error into the learning rule
        nengo.Connection(error, conn.learning_rule)
        
        post_p.append(nengo.Probe(post, synapse=0.01))
    
if spinnaker:
    nengo_spinnaker.add_spinnaker_params(model.config)
    model.config[inp].function_of_time = True

    sim = nengo_spinnaker.Simulator(model)
    with sim:
        sim.run(10.0)
else:
    sim = nengo.Simulator(model)
    sim.run(10.0)


plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(sim.trange(), sim.data[inp_p].T[0], c='k', label='Input')
plt.plot(sim.trange(), sim.data[pre_p].T[0], c='b', label='Pre')
for p, (r, c) in zip(post_p, learning_rates):
    plt.plot(sim.trange(), sim.data[p].T[0], c=c, label='Post %f' % r)
plt.ylabel("Dimension 1")
plt.legend(loc='best')

plt.subplot(2, 1, 2)
plt.plot(sim.trange(), sim.data[inp_p].T[1], c='k', label='Input')
plt.plot(sim.trange(), sim.data[pre_p].T[1], c='b', label='Pre')
for p, (r, c) in zip(post_p, learning_rates):
    plt.plot(sim.trange(), sim.data[p].T[1], c=c, label='Post %f' % r)
plt.ylabel("Dimension 2")
plt.legend(loc='best');
plt.show()