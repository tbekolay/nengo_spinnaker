"""LIF Ensemble

Takes an intermediate representation of a LIF ensemble and returns a vertex and
appropriate callbacks to load and prepare the ensemble for simulation on
SpiNNaker.  The build method also manages the partitioning of the ensemble into
appropriate sized slices.
"""

from bitarray import bitarray
import collections
import math
import nengo
from nengo.base import ObjView
import numpy as np
from rig.machine import Cores, SDRAM
from six import iteritems
import struct

from nengo.connection import LearningRule

from nengo_spinnaker.builder.builder import InputPort, netlistspec, OutputPort
from nengo_spinnaker.builder.ports import EnsembleInputPort, EnsembleOutputPort
from nengo_spinnaker.regions.filters import (
    make_filter_regions, add_filters, FilterRegion, FilterRoutingRegion
)
from .. import regions
from nengo_spinnaker.netlist import VertexSlice
from nengo_spinnaker import partition_and_cluster as partition
from nengo_spinnaker.utils.application import get_application
from nengo_spinnaker.utils import type_casts as tp


class EnsembleLIF(object):
    """Controller for an ensemble of LIF neurons."""
    def __init__(self, ensemble):
        """Create a new LIF ensemble controller."""
        self.ensemble = ensemble
        self.direct_input = np.zeros(ensemble.size_in)
        self.local_probes = list()

    def make_vertices(self, model, n_steps):  # TODO remove n_steps
        """Construct the data which can be loaded into the memory of a
        SpiNNaker machine.
        """
        # Build encoders, gain and bias regions
        params = model.params[self.ensemble]

        # Convert the gains to S1615 before creating the region
        self.gain_region = regions.MatrixRegion(
            tp.np_to_fix(params.gain),
            sliced_dimension=regions.MatrixPartitioning.rows
        )

        # Extract all the filters from the incoming connections
        incoming = model.get_signals_connections_to_object(self)

        # Filter out incoming modulatory connections
        incoming_modulatory = {port: signal
                               for (port, signal) in iteritems(incoming)
                               if isinstance(port, LearningRule)}

        # Build input filters filter routing regions for incoming connections
        #  that sink into this ensembles standard input ports
        self.input_filters, self.input_filter_routing = make_filter_regions(
            incoming[InputPort.standard], model.dt, True,
            model.keyspaces.filter_routing_tag, width=self.ensemble.size_in
        )
        self.inhib_filters, self.inhib_filter_routing = make_filter_regions(
            incoming[EnsembleInputPort.global_inhibition], model.dt, True,
            model.keyspaces.filter_routing_tag, width=1
        )

        # Extract all the decoders for the outgoing connections and build the
        # regions for the decoders and the regions for the output keys.
        outgoing = model.get_signals_connections_from_object(self)
        decoders, output_keys = \
            get_decoders_and_keys(model, outgoing[OutputPort.standard], True)

        # Extract pre-scaled encoders from parameters
        encoders_with_gain = params.scaled_encoders

        # Create filtered activity region
        self.filtered_activity_region = FilteredActivityRegion(model.dt)

        # Create, initially empty, PES region
        self.pes_region = PESRegion()

        # Loop through outgoing learnt connections
        mod_filters = list()
        mod_keyspace_routes = list()
        learnt_encoder_filters = list()
        learnt_encoder_routes = list()
        for sig, conns in iteritems(outgoing[EnsembleOutputPort.learnt]):
            # If this learning rule is PES
            l_type = conns[0].learning_rule_type
            if isinstance(l_type, nengo.PES):
                # If there is a modulatory connection
                # associated with the learning rule
                l_rule = conns[0].learning_rule
                if l_rule in incoming_modulatory:
                    e = incoming_modulatory[l_rule]

                    # Cache the index of the input filter containing
                    # the error signal and the offset into the decoder
                    # where the learning rule should operate
                    error_filter_index = len(mod_filters)
                    decoder_offset = decoders.shape[1]

                    # Get new decoders and output keys for learnt connection
                    learnt_decoders, learnt_output_keys = \
                        get_decoders_and_keys(model, {sig: conns}, False)

                    # If there are no existing decodes, hstacking doesn't
                    # work so set decoders to new learnt decoder matrix
                    if decoder_offset == 0:
                        decoders = learnt_decoders
                    # Otherwise, stack learnt decoders beneath existing matrix
                    else:
                        decoders = np.hstack((decoders, learnt_decoders))

                    # Also add output keys to list
                    output_keys.extend(learnt_output_keys)

                    # Add error connection to lists
                    # of modulatory filters and routes
                    mod_filters, mod_keyspace_routes = add_filters(
                        mod_filters, mod_keyspace_routes, e, minimise=False)

                    # Either add a new filter to the filtered activity
                    # region or get the index of the existing one
                    activity_filter_index = \
                        self.filtered_activity_region.add_get_filter(
                            l_type.pre_tau)

                    # Add a new learning rule to the PES region
                    # **NOTE** divide learning rate by dt
                    # to account for activity scaling
                    self.pes_region.learning_rules.append(
                        PESLearningRule(
                            learning_rate=l_type.learning_rate / model.dt,
                            error_filter_index=error_filter_index,
                            decoder_offset=decoder_offset,
                            activity_filter_index=activity_filter_index))
                # Otherwise raise an error - PES requires a modulatory signal
                else:
                    raise ValueError(
                        "Ensemble %s has outgoing connection with PES "
                        "learning, but no corresponding modulatory "
                        "connection" % self.ensemble.label
                    )
            else:
                raise NotImplementedError(
                    "SpiNNaker does not support %s learning rule." % l_type
                )

        # Create, initially empty, Voja region, passing in scaling factor
        # used, with gain, to scale activities to match encoders
        self.voja_region = VojaRegion(1.0 / self.ensemble.radius)

        # Loop through incoming learnt connections
        for sig, conns in iteritems(incoming[EnsembleInputPort.learnt]):
            # If this learning rule is Voja
            l_type = conns[0].learning_rule_type
            if isinstance(l_type, nengo.Voja):
                # If there is a modulatory connection associated
                # with this connection's learning rule
                l_rule = conns[0].learning_rule

                # If there is a modulatory connection
                # associated with the learning rule
                l_rule = conns[0].learning_rule
                if l_rule in incoming_modulatory:
                    l = incoming_modulatory[l_rule]

                    # Cache the index of the input filter
                    # containing the learning signal
                    learn_sig_filter_index = len(mod_filters)

                    # Add learning connection to lists
                    # of modulatory filters and routes
                    mod_filters, mod_keyspace_routes = add_filters(
                        mod_filters, mod_keyspace_routes, l, minimise=False)
                # Otherwise, learning is always on so
                # invalidate learning signal index
                else:
                    learn_sig_filter_index = -1

                # Cache the index of the input filter containing
                # the input signal and the offset into the encoder
                # where the learning rule should operate
                decoded_input_filter_index = len(learnt_encoder_filters)
                encoder_offset = encoders_with_gain.shape[1]

                # Create a duplicate copy of the original size_in columns of
                # the encoder matrix for modification by this learning rule
                base_encoders = encoders_with_gain[:, :self.ensemble.size_in]
                encoders_with_gain = np.hstack((encoders_with_gain,
                                                base_encoders))

                # Add learnt connection to list of filters
                # and routes with learnt encoders
                learnt_encoder_filters, learnt_encoder_routes = add_filters(
                    learnt_encoder_filters, learnt_encoder_routes,
                    {sig: conns}, minimise=False)

                # Either add a new filter to the filtered activity
                # region or get the index of the existing one
                activity_filter_index = \
                    self.filtered_activity_region.add_get_filter(
                        l_type.post_tau)

                # Add a new learning rule to the Voja region
                # **NOTE** divide learning rate by dt
                # to account for activity scaling
                self.voja_region.learning_rules.append(
                    VojaLearningRule(
                        learning_rate=l_type.learning_rate,
                        learning_signal_filter_index=learn_sig_filter_index,
                        encoder_offset=encoder_offset,
                        decoded_input_filter_index=decoded_input_filter_index,
                        activity_filter_index=activity_filter_index))
            else:
                raise NotImplementedError(
                    "SpiNNaker does not support %s learning rule." % l_type
                )

        # Convert the encoders combined with the gain to S1615 before creating
        # the region.
        self.encoders_region = regions.MatrixRegion(
            tp.np_to_fix(encoders_with_gain),
            sliced_dimension=regions.MatrixPartitioning.rows
        )

        # Combine the direct input with the bias before converting to S1615 and
        # creating the region.
        tiled_direct_input = np.tile(
            self.direct_input,
            encoders_with_gain.shape[1] // self.ensemble.size_in)
        bias_with_di = params.bias + np.dot(encoders_with_gain,
                                            tiled_direct_input)

        assert bias_with_di.ndim == 1
        self.bias_region = regions.MatrixRegion(
            tp.np_to_fix(bias_with_di),
            sliced_dimension=regions.MatrixPartitioning.rows
        )

        # Create modulatory filter and routing regions
        self.mod_filters = FilterRegion(mod_filters, model.dt)
        self.mod_filter_routing = FilterRoutingRegion(
            mod_keyspace_routes, model.keyspaces.filter_routing_tag)

        # Create learnt encoder filter and routing regions
        self.learnt_encoder_filters = FilterRegion(learnt_encoder_filters,
                                                   model.dt)
        self.learnt_encoder_routing = FilterRoutingRegion(
            learnt_encoder_routes, model.keyspaces.filter_routing_tag)

        # Now decoder is fully built, extract size
        size_out = decoders.shape[1]

        self.decoders_region = regions.MatrixRegion(
            tp.np_to_fix(decoders / model.dt),
            sliced_dimension=regions.MatrixPartitioning.rows
        )
        self.output_keys_region = regions.KeyspacesRegion(
            output_keys, fields=[regions.KeyField({'cluster': 'cluster'})]
        )

        # Initially disable all recording
        self.spike_recording_region = None
        self.encoder_recording_region = None
        self.probe_spikes = False
        self.probe_encoders = False

        # Loop through all local probes
        for p in self.local_probes:
            # If this is an encoder probe and encoder probing
            # isn't already enabled, create region and set flag
            if p.attr == "scaled_encoders" and not self.probe_encoders:
                self.encoder_recording_region = EncoderRecordingRegion(
                    n_steps,
                    encoders_with_gain.shape[1] - self.ensemble.size_in
                )
                self.probe_encoders = True
            # If this is a spike probe and spike probing isn't
            # already enabled, create region and set flag
            elif p.attr in ("output", "spikes") and not self.probe_spikes:
                self.spike_recording_region = SpikeRecordingRegion(n_steps)
                self.probe_spikes = True
            else:
                raise NotImplementedError(
                    "SpiNNaker does not support %s probe type." % p.attr
                )

        # Create the regions list
        self.regions = [
            SystemRegion(self.ensemble.size_in,
                         encoders_with_gain.shape[1],
                         size_out,
                         model.machine_timestep,
                         self.ensemble.neuron_type.tau_ref,
                         self.ensemble.neuron_type.tau_rc,
                         model.dt,
                         self.probe_spikes,
                         self.probe_encoders
                         ),
            self.bias_region,
            self.encoders_region,
            self.decoders_region,
            self.output_keys_region,
            self.gain_region,
            self.input_filters,
            self.input_filter_routing,
            self.inhib_filters,
            self.inhib_filter_routing,
            self.mod_filters,
            self.mod_filter_routing,
            self.learnt_encoder_filters,
            self.learnt_encoder_routing,
            self.pes_region,
            self.voja_region,
            self.filtered_activity_region,
            self.spike_recording_region,
            self.encoder_recording_region,
        ]

        # Partition the ensemble and get a list of vertices to load to the
        # machine.  We can expect to be DTCM or CPU bound, so the SDRAM bound
        # can be quite lax to allow for lots of data probing.
        # TODO: Include other DTCM usage
        def cpu_usage(sl):
            """Calculate the CPU usage (in cycles) based on the number of
            neurons and the size_in and size_out of the ensemble.

            The equation and coefficients are taken from: "An Efficient
            SpiNNaker Implementation of the NEF", Mundy, Knight, Stewart and
            Furber [IJCNN 2015]
            """
            n_neurons = (sl.stop - sl.start)
            return (245 + 43*self.ensemble.size_in + 100 + 702*size_out +
                    188 + 69*n_neurons + 13*n_neurons*self.ensemble.size_in)

        self.vertices = list()
        sdram_constraint = partition.Constraint(8*2**20)  # Max 8MiB
        dtcm_constraint = partition.Constraint(64*2**10, .75)  # 75% of 64KiB
        cpu_constraint = partition.Constraint(200000, .8)  # 80% of 200k cycles
        constraints = {
            sdram_constraint: lambda s: regions.utils.sizeof_regions(
                self.regions, s),
            dtcm_constraint: lambda s: regions.utils.sizeof_regions(
                self.regions[:-2], s) + 5*(s.stop - s.start),  # **HACK** no recording, +5 bytes per neuron
            cpu_constraint: cpu_usage,
        }

        for sl in partition.partition(slice(0, self.ensemble.n_neurons),
                                      constraints):
            resources = {
                Cores: 1,
                SDRAM: regions.utils.sizeof_regions(self.regions, sl),
            }
            vsl = VertexSlice(sl, get_application("ensemble"), resources)
            self.vertices.append(vsl)

        # Return the vertices and callback methods
        return netlistspec(self.vertices, self.load_to_machine,
                           after_simulation_function=self.after_simulation)

    def load_to_machine(self, netlist, controller):
        """Load the ensemble data into memory."""
        # For each slice
        self.spike_recording_mem = dict()
        self.encoder_recording_mem = dict()
        for vertex in self.vertices:
            # Layout the slice of SDRAM we have been given
            region_memory = regions.utils.create_app_ptr_and_region_files(
                netlist.vertices_memory[vertex], self.regions, vertex.slice)

            # Write in each region
            for region, mem in zip(self.regions, region_memory):
                if region is None:
                    pass
                elif region is self.output_keys_region:
                    self.output_keys_region.write_subregion_to_file(
                        mem, vertex.slice, cluster=vertex.cluster)
                elif (region is self.spike_recording_region
                      and self.probe_spikes):
                    self.spike_recording_mem[vertex] = mem
                elif (region is self.encoder_recording_region
                      and self.probe_encoders):
                    self.encoder_recording_mem[vertex] = mem
                else:
                    region.write_subregion_to_file(mem, vertex.slice)

    def before_simulation(self, netlist, controller, simulator, n_steps):
        """Load data for a specific number of steps to the machine."""
        # TODO When supported by executables
        raise NotImplementedError

    def after_simulation(self, netlist, simulator, n_steps):
        """Retrieve data from a simulation and ensure."""
        # If we have probed the spikes then retrieve the spike data and store
        # it in the simulator data.
        if self.probe_spikes:
            probed_spikes = np.zeros((n_steps, self.ensemble.n_neurons))

            for vertex in self.vertices:
                mem = self.spike_recording_mem[vertex]
                mem.seek(0)

                spike_data = bitarray(endian="little")
                spike_data.frombytes(mem.read())
                n_neurons = vertex.slice.stop - vertex.slice.start

                bpf = self.spike_recording_region.bytes_per_frame(
                    vertex.slice) * 8
                spikes = (spike_data[n*bpf:n*bpf + n_neurons] for n in
                          range(n_steps))

                probed_spikes[:, vertex.slice] = \
                    np.array([[1./simulator.dt if s else 0.0 for s in ss]
                              for ss in spikes])

        # If we've probed (learnt) encoders
        if self.probe_encoders:
            # Create empty matrix to hold probed data
            probed_encoders = np.empty((
                n_steps,
                self.ensemble.n_neurons,
                self.encoder_recording_region.n_dimensions))

            for vertex in self.vertices:
                mem = self.encoder_recording_mem[vertex]
                mem.seek(0)

                # Read in neuron slice of fixed point
                # values and convert to float
                fp = np.fromstring(mem.read(), dtype=np.int32)
                slice_encoders = tp.fix_to_np(fp)

                # Reshape and insert into probed encoders
                n_neurons = vertex.slice.stop - vertex.slice.start
                slice_encoders = np.reshape(
                    slice_encoders,
                    (
                        n_steps,
                        n_neurons,
                        self.encoder_recording_region.n_dimensions
                    )
                )
                probed_encoders[:, vertex.slice.start:vertex.slice.stop, :] = \
                    slice_encoders

        # Store the data associated with every probe, applying the sampling
        # and slicing specified for the probe.
        for p in self.local_probes:
            if p.sample_every is None:
                sample_every = 1
            else:
                sample_every = p.sample_every / simulator.dt

            if not isinstance(p.target, ObjView):
                neuron_slice = slice(None)
            else:
                neuron_slice = p.target.slice

            # If this is an encoder probe
            if p.attr == "scaled_encoders":
                # **TODO** rejoin learning rule with encoder offset
                # Copy sliced data out of probed encoders
                simulator.data[p] = \
                    probed_encoders[::sample_every, neuron_slice, :]
            # Otherwise, if it's a spike probe,
            # copy sliced data out of probed spikes
            elif p.attr in ("output", "spikes"):
                simulator.data[p] = \
                    probed_spikes[::sample_every, neuron_slice]


class SystemRegion(collections.namedtuple(
    "SystemRegion", "n_input_dimensions, encoder_width, n_output_dimensions, "
                    "machine_timestep, t_ref, t_rc, dt, "
                    "probe_spikes, probe_encoders")):
    """Region of memory describing the general parameters of a LIF ensemble."""

    def sizeof(self, vertex_slice=slice(None)):
        """Get the number of bytes necessary to represent this region of
        memory.
        """
        return 10 * 4  # 10 words

    sizeof_padded = sizeof

    def write_subregion_to_file(self, fp, vertex_slice):
        """Write the system region for a specific vertex slice to a file-like
        object.
        """
        # The value -e^(-dt / tau_rc) is precomputed and is scaled down ever so
        # slightly to account for the effects of fixed point.  The result is
        # that the tuning curves of SpiNNaker neurons are usually within 5Hz of
        # the ideal curve and the tuning curve of reference Nengo neurons.
        # The fudge factor applied (i.e. 1.0*2^-11) was determined by running
        # the tuning curve test in "regression-tests/test_tuning_curve.py",
        # plotting the results and stopping when the ideal tuning curve was
        # very closely matched by the SpiNNaker tuning curve - further
        # improvement of this factor may be possible.

        n_neurons = vertex_slice.stop - vertex_slice.start
        data = struct.pack(
            "<10I",
            self.n_input_dimensions,
            self.encoder_width,
            self.n_output_dimensions,
            n_neurons,
            self.machine_timestep,
            int(self.t_ref // self.dt),  # tau_ref expressed as in ticks
            tp.value_to_fix(-np.expm1(-self.dt / self.t_rc) * (1.0 - 2**-11)),
            (0x1 if self.probe_spikes else 0x0),
            (0x1 if self.probe_encoders else 0x0),
            1
        )
        fp.write(data)

PESLearningRule = collections.namedtuple(
    "PESLearningRule",
    "learning_rate, error_filter_index, decoder_offset, activity_filter_index")


class PESRegion(regions.Region):
    """Region representing parameters for PES learning rules.
    """
    def __init__(self):
        self.learning_rules = []

    def sizeof(self, *args):
        return 4 + (len(self.learning_rules) * 16)

    def write_subregion_to_file(self, fp, vertex_slice):
        # Get length of slice for scaling learning rate
        n_neurons = float(vertex_slice.stop - vertex_slice.start)

        # Write number of learning rules
        fp.write(struct.pack("<I", len(self.learning_rules)))

        # Write learning rules
        for l in self.learning_rules:
            data = struct.pack(
                "<3Ii",
                tp.value_to_fix(l.learning_rate / n_neurons),
                l.error_filter_index,
                l.decoder_offset,
                l.activity_filter_index
            )
            fp.write(data)


VojaLearningRule = collections.namedtuple(
    "VojaLearningRule",
    "learning_rate, learning_signal_filter_index, encoder_offset, "
    "decoded_input_filter_index, activity_filter_index")


class VojaRegion(regions.Region):
    """Region representing parameters for PES learning rules.
    """
    def __init__(self, one_over_radius):
        self.learning_rules = []
        self.one_over_radius = one_over_radius

    def sizeof(self, vertex_slice):
        return 8 + (len(self.learning_rules) * 20)

    def write_subregion_to_file(self, fp, vertex_slice):
        # Write number of learning rules and scaling factor
        fp.write(struct.pack(
            "<2I",
            len(self.learning_rules),
            tp.value_to_fix(self.one_over_radius)
        ))

        # Write learning rules
        for l in self.learning_rules:
            data = struct.pack(
                "<Ii2Ii",
                tp.value_to_fix(l.learning_rate),
                l.learning_signal_filter_index,
                l.encoder_offset,
                l.decoded_input_filter_index,
                l.activity_filter_index
            )
            fp.write(data)


def get_decoders_and_keys(model, signals_connections, minimise=False):
    """Get a combined decoder matrix and a list of keys to use to transmit
    elements decoded using the decoders.
    """
    decoders = list()
    keys = list()

    # For each signal with a single connection we save the decoder and generate
    # appropriate keys
    for signal, connections in iteritems(signals_connections):
        assert len(connections) == 1
        decoder = model.params[connections[0]].decoders
        transform = model.params[connections[0]].transform
        decoder = np.dot(transform, decoder.T).T

        if not minimise:
            keep = np.array([True for _ in range(decoder.shape[1])])
        else:
            # We can reduce the number of packets sent and the memory
            # requirements by removing columns from the decoder matrix which
            # will always result in packets containing zeroes.
            keep = np.any(decoder != 0, axis=0)

        decoders.append(decoder[:, keep])
        for i, k in zip(range(decoder.shape[1]), keep):
            if k:
                keys.append(signal.keyspace(index=i))

    # Stack the decoders
    if len(decoders) > 0:
        decoders = np.hstack(decoders)
    else:
        decoders = np.array([[]])

    return decoders, keys


class SpikeRecordingRegion(regions.Region):
    """Region used to record spikes."""
    def __init__(self, n_steps):
        self.n_steps = n_steps

    def sizeof(self, vertex_slice):
        # Get the number of words per frame
        return self.bytes_per_frame(vertex_slice) * self.n_steps

    def bytes_per_frame(self, vertex_slice):
        n_neurons = vertex_slice.stop - vertex_slice.start
        words_per_frame = n_neurons//32 + (1 if n_neurons % 32 else 0)
        return 4 * words_per_frame

    def write_subregion_to_file(self, *args, **kwargs):  # pragma: no cover
        pass  # Nothing to do


class EncoderRecordingRegion(regions.Region):
    """Region used to record spikes."""
    def __init__(self, n_steps, n_dimensions):
        self.n_steps = n_steps
        self.n_dimensions = n_dimensions

    def sizeof(self, vertex_slice):
        # Get the number of words per frame
        s = self.bytes_per_frame(vertex_slice) * self.n_steps
        return s

    def bytes_per_frame(self, vertex_slice):
        n_neurons = vertex_slice.stop - vertex_slice.start
        words_per_frame = n_neurons * self.n_dimensions
        return 4 * words_per_frame

    def write_subregion_to_file(self, *args, **kwargs):  # pragma: no cover
        pass  # Nothing to do


class FilteredActivityRegion(regions.Region):
    def __init__(self, dt):
        self.filter_propogators = []
        self.dt = dt

    def add_get_filter(self, time_constant):
        # If time constant is none or less than dt,
        # a filter is not required so return -1
        if time_constant is None or time_constant < self.dt:
            return -1
        # Otherwise
        else:
            # Calculate propogator
            propogator = math.exp(-float(self.dt) / float(time_constant))

            # Convert to fixed-point
            propogator_fixed = tp.value_to_fix(propogator)

            # If there is already a filter with the same fixed-point
            # propogator in the list, return its index
            if propogator_fixed in self.filter_propogators:
                return self.filter_propogators.index(propogator_fixed)
            # Otherwise add propogator to list and return its index
            else:
                self.filter_propogators.append(propogator_fixed)
                return (len(self.filter_propogators) - 1)

    def sizeof(self, vertex_slice):
        return 4 + (8 * len(self.filter_propogators))

    def write_subregion_to_file(self, fp, vertex_slice):
        # Write number of learning rules
        fp.write(struct.pack("<I", len(self.filter_propogators)))

        # Write filters
        for f in self.filter_propogators:
            data = struct.pack(
                "<ii",
                f,
                tp.value_to_fix(1.0) - f,
            )
            fp.write(data)
