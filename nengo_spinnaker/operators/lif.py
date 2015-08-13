"""LIF Ensemble

Takes an intermediate representation of a LIF ensemble and returns a vertex and
appropriate callbacks to load and prepare the ensemble for simulation on
SpiNNaker.  The build method also manages the partitioning of the ensemble into
appropriate sized slices.
"""

from bitarray import bitarray
import collections
from nengo.base import ObjView
import numpy as np
from rig.machine import Cores, SDRAM
from six import iteritems
import struct

from nengo_spinnaker.builder.builder import InputPort, netlistspec, OutputPort
from nengo_spinnaker.builder.ports import EnsembleInputPort
from nengo_spinnaker.regions.filters import make_filter_regions
from .. import regions
from nengo_spinnaker.netlist import VertexSlice
from nengo_spinnaker import partition_and_cluster as partition
from nengo_spinnaker.utils.application import get_application
from nengo_spinnaker.utils.config import getconfig
from nengo_spinnaker.utils import type_casts as tp


class EnsembleLIF(object):
    # Tag names, corresponding to those defined in ensemble_profiler.h
    profiler_tag_names = {
        0:  "Timer tick",
        1:  "Input filter",
        2:  "Output filter",
        3:  "Neuron update",
        4:  "Transmit output",
    }

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

        # Convert the encoders combined with the gain to S1615 before creating
        # the region.
        encoders_with_gain = params.scaled_encoders
        self.encoders_region = regions.MatrixRegion(
            tp.np_to_fix(encoders_with_gain),
            sliced_dimension=regions.MatrixPartitioning.rows
        )

        # Combine the direct input with the bias before converting to S1615 and
        # creating the region.
        bias_with_di = params.bias + np.dot(encoders_with_gain,
                                            self.direct_input)
        assert bias_with_di.ndim == 1
        self.bias_region = regions.MatrixRegion(
            tp.np_to_fix(bias_with_di),
            sliced_dimension=regions.MatrixPartitioning.rows
        )

        # Convert the gains to S1615 before creating the region
        self.gain_region = regions.MatrixRegion(
            tp.np_to_fix(params.gain),
            sliced_dimension=regions.MatrixPartitioning.rows
        )

        # Extract all the filters from the incoming connections
        incoming = model.get_signals_connections_to_object(self)

        self.input_filters, self.input_filter_routing = make_filter_regions(
            incoming[InputPort.standard], model.dt, True,
            model.keyspaces.filter_routing_tag, width=self.ensemble.size_in
        )
        self.inhib_filters, self.inhib_filter_routing = make_filter_regions(
            incoming[EnsembleInputPort.global_inhibition], model.dt, True,
            model.keyspaces.filter_routing_tag, width=1
        )
        self.mod_filters, self.mod_filter_routing = make_filter_regions(
            {}, model.dt, True, model.keyspaces.filter_routing_tag
        )

        # Extract all the decoders for the outgoing connections and build the
        # regions for the decoders and the regions for the output keys.
        outgoing = model.get_signals_connections_from_object(self)
        decoders, output_keys = \
            get_decoders_and_keys(model, outgoing[OutputPort.standard], True)
        size_out = decoders.shape[1]

        # TODO: Include learnt decoders
        self.pes_region = PESRegion()

        self.decoders_region = regions.MatrixRegion(
            tp.np_to_fix(decoders / model.dt),
            sliced_dimension=regions.MatrixPartitioning.rows
        )
        self.output_keys_region = regions.KeyspacesRegion(
            output_keys, fields=[regions.KeyField({'cluster': 'cluster'})]
        )

        # Create the recording regions for locally situated probes
        self.spike_region = None
        self.probe_spikes = False
        self.voltage_region = None
        self.probe_voltages = False

        for probe in self.local_probes:
            # For each probe determine which regions and flags should be set
            if probe.attr in ("output", "spikes"):
                # If spikes are being probed then ensure that the flag is set
                # and a region exists.
                if not self.probe_spikes:
                    self.spike_region = SpikeRegion(n_steps)
                    self.probe_spikes = True
            elif probe.attr in ("voltage"):
                # If voltages are being probed then ensure that the flag is set
                # and a region exists.
                if not self.probe_voltages:
                    self.voltage_region = VoltageRegion(n_steps)
                    self.probe_voltages = True

        # If profiling is enabled
        num_profiler_samples = 0
        if getconfig(model.config, self.ensemble, "profile", False):
            # Try and get number of samples from config
            num_profiler_samples = getconfig(model.config, self.ensemble,
                                             "profile_num_samples")

            # If it's not specified, calculate sensible default
            if num_profiler_samples is None:
                num_profiler_samples =\
                    len(EnsembleLIF.profiler_tag_names) * n_steps * 2

        # Create profiler region
        self.profiler_region = regions.Profiler(num_profiler_samples)

        # Create the regions list
        self.regions = [
            SystemRegion(self.ensemble.size_in,
                         size_out,
                         model.machine_timestep,
                         self.ensemble.neuron_type.tau_ref,
                         self.ensemble.neuron_type.tau_rc,
                         model.dt,
                         self.probe_spikes,
                         self.probe_voltages,
                         num_profiler_samples
                         ),
            self.bias_region,
            self.encoders_region,
            self.decoders_region,
            self.output_keys_region,
            self.input_filters,
            self.input_filter_routing,
            self.inhib_filters,
            self.inhib_filter_routing,
            self.gain_region,
            self.mod_filters,
            self.mod_filter_routing,
            self.pes_region,
            self.profiler_region,
            self.spike_region,
            self.voltage_region,
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
            # **HACK** don't include last three regions in DTCM estimate
            # (profiler and spike recording)
            dtcm_constraint: lambda s: regions.utils.sizeof_regions(
                self.regions[:-3], s) + 5*(s.stop - s.start),
            cpu_constraint: cpu_usage,
        }
        app_name = (
            "ensemble_profiled" if num_profiler_samples > 0
            else "ensemble"
        )
        for sl in partition.partition(slice(0, self.ensemble.n_neurons),
                                      constraints):
            resources = {
                Cores: 1,
                SDRAM: regions.utils.sizeof_regions(self.regions, sl),
            }
            vsl = VertexSlice(sl, get_application(app_name), resources)
            self.vertices.append(vsl)

        # Return the vertices and callback methods
        return netlistspec(self.vertices, self.load_to_machine,
                           after_simulation_function=self.after_simulation)

    def load_to_machine(self, netlist, controller):
        """Load the ensemble data into memory."""
        # For each slice
        self.spike_mem = dict()
        self.voltage_mem = dict()
        self.profiler_output_mem = dict()

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
                elif region is self.spike_region and self.probe_spikes:
                    self.spike_mem[vertex] = mem
                elif (region is self.profiler_region
                      and self.profiler_region.n_samples > 0):
                    self.profiler_output_mem[vertex] = mem
                elif region is self.voltage_region and self.probe_voltages:
                    self.voltage_mem[vertex] = mem
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
                mem = self.spike_mem[vertex]
                mem.seek(0)

                spike_data = bitarray(endian="little")
                spike_data.frombytes(mem.read())
                n_neurons = vertex.slice.stop - vertex.slice.start

                bpf = self.spike_region.bytes_per_frame(n_neurons)*8
                spikes = (spike_data[n*bpf:n*bpf + n_neurons] for n in
                          range(n_steps))

                probed_spikes[:, vertex.slice] = \
                    np.array([[1./simulator.dt if s else 0.0 for s in ss]
                              for ss in spikes])

            # Store the data associated with every probe, applying the sampling
            # and slicing specified for the probe.
            for p in self.local_probes:
                if p.attr not in ("output", "spikes"):
                    continue

                if p.sample_every is None:
                    sample_every = 1
                else:
                    sample_every = p.sample_every / simulator.dt

                if not isinstance(p.target, ObjView):
                    neuron_slice = slice(None)
                else:
                    neuron_slice = p.target.slice

                simulator.data[p] = probed_spikes[::sample_every, neuron_slice]

        # If we have probed the voltages then retrieve the probed data and
        # store it in the simulator data.
        if self.probe_voltages:
            # Prepare a buffer to store the probed voltages
            probed_voltages = np.zeros((n_steps, self.ensemble.n_neurons))

            # Read back the voltage for each vertex
            for vertex in self.vertices:
                # Get the region of memory to read for this vertex
                mem = self.voltage_mem[vertex]
                mem.seek(0)

                # Determine how many bytes to read back
                n_neurons = vertex.slice.stop - vertex.slice.start
                n_bytes = (self.voltage_region.bytes_per_frame(n_neurons) *
                           n_steps)

                # Read back the voltage data, unlike the majority of values the
                # voltages are stored as U1.15 which means we read them back as
                # UINT16s before converting to floats.
                voltage_data = np.frombuffer(mem.read(n_bytes),
                                             dtype=np.uint16)
                voltages = tp.fix_to_np(voltage_data)
                voltages.shape = (n_steps, -1)

                # Store the voltage data into the probed voltages
                probed_voltages[:, vertex.slice] = voltages[:, vertex.slice]

            # Store the data associated with every probe, applying the sampling
            # and slicing specified for the probe.
            for p in self.local_probes:
                if p.attr != "voltage":
                    continue

                if p.sample_every is None:
                    sample_every = 1
                else:
                    sample_every = p.sample_every / simulator.dt

                if not isinstance(p.target, ObjView):
                    neuron_slice = slice(None)
                else:
                    neuron_slice = p.target.slice

                probe_data = probed_voltages[::sample_every, neuron_slice]
                if p not in simulator.data:
                    simulator.data[p] = probe_data
                else:
                    simulator.data[p] = np.hstack((simulator.data[p],
                                                   probe_data))

        # If profiling is enabled
        if self.profiler_region.n_samples > 0:
            # Loop through vertices
            for vertex in self.vertices:
                # Get profiler output memory block
                mem = self.profiler_output_mem[vertex]
                mem.seek(0)

                # Read profiler data from memory and put somewhere accessible
                simulator.profiler_data[self.ensemble] =\
                    self.profiler_region.read_from_mem(
                        mem, EnsembleLIF.profiler_tag_names)


class SystemRegion(collections.namedtuple(
    "SystemRegion", "n_input_dimensions, n_output_dimensions, "
                    "machine_timestep, t_ref, t_rc, dt, probe_spikes, "
                    "probe_voltages, num_profiler_samples")):
    """Region of memory describing the general parameters of a LIF ensemble."""

    def sizeof(self, vertex_slice=slice(None)):
        """Get the number of bytes necessary to represent this region of
        memory.
        """
        return 9 * 4  # 9 words

    sizeof_padded = sizeof

    def write_subregion_to_file(self, fp, vertex_slice):
        """Write the system region for a specific vertex slice to a file-like
        object.
        """
        # Prepare the flags, these indicate any additional tasks to be
        # performed by the executable.
        flags = 0x0

        if self.probe_spikes:
            flags |= 1 << 0

        if self.probe_voltages:
            flags |= 1 << 1

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
            "<9I",
            self.n_input_dimensions,
            self.n_output_dimensions,
            n_neurons,
            self.machine_timestep,
            int(self.t_ref // self.dt),  # tau_ref expressed as in ticks
            tp.value_to_fix(-np.expm1(-self.dt / self.t_rc) * (1.0 - 2**-11)),
            flags,
            1,
            self.num_profiler_samples
        )
        fp.write(data)


class PESRegion(regions.Region):
    """Region representing parameters for PES learning rules.
    """
    # TODO Implement PES

    def sizeof(self, *args):
        return 4

    def write_subregion_to_file(self, fp, vertex_slice):
        # Write out a zero, indicating no PES data
        fp.write(b"\x00" * 4)


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


class RecordingRegion(regions.Region):
    def __init__(self, n_steps):
        self.n_steps = n_steps

    def sizeof(self, vertex_slice):
        # Get the number of words per frame
        n_neurons = vertex_slice.stop - vertex_slice.start
        return self.bytes_per_frame(n_neurons) * self.n_steps

    def write_subregion_to_file(self, *args, **kwargs):  # pragma: no cover
        pass  # Nothing to do


class SpikeRegion(RecordingRegion):
    """Region used to record spikes."""
    def bytes_per_frame(self, n_neurons):
        words_per_frame = n_neurons//32 + (1 if n_neurons % 32 else 0)
        return 4 * words_per_frame


class VoltageRegion(RecordingRegion):
    """Region used to record neuron input voltages."""
    def bytes_per_frame(self, n_neurons):
        words_per_frame = n_neurons // 2 + n_neurons % 2
        return 4 * words_per_frame
