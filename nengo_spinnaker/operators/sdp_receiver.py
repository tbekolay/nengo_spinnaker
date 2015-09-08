from rig.machine import Cores, SDRAM
import six
import struct

from nengo_spinnaker.builder.builder import netlistspec
from nengo_spinnaker.builder.model import OutputPort
from nengo_spinnaker.netlist import Vertex
from nengo_spinnaker.regions import KeyspacesRegion, KeyField, Region
from nengo_spinnaker.regions import utils as region_utils
from nengo_spinnaker.utils.application import get_application


class SDPReceiver(object):
    """An operator which receives SDP packets and transmits the contained data
    as a stream of multicast packets.
    """
    def __init__(self):
        # Create a mapping of which connection is broadcast by which vertex
        self.connection_vertices = dict()
        self._sys_regions = dict()
        self._key_regions = dict()

    def make_vertices(self, model, *args, **kwargs):
        """Create vertices that will simulate the SDPReceiver."""
        # NOTE This approach will result in more routes being created than are
        # actually necessary; the way to avoid this is to modify how the
        # builder deals with signals when creating netlists.

        # Get all outgoing signals and their associated connections (this
        # SHOULD be a 1:1 mapping)
        out = model.get_signals_connections_from_object(self)
        for signal, connections in six.iteritems(out[OutputPort.standard]):
            assert len(connections) == 1, "Expecting a 1:1 mapping"
            conn = connections[0]

            # Get the transform, and from this the keys
            transform = model.params[conn].transform
            keys = [signal.keyspace(index=i) for i in
                    range(transform.shape[0])]

            # Create a vertex for this connection (assuming its size out <= 64)
            if len(keys) > 64:
                raise NotImplementedError(
                    "Connection {!s} is too wide to transmit to SpiNNaker. "
                    "Consider breaking the connection up or making the "
                    "originating node a function of time Node.".format(conn)
                )

            # Create the regions for the system
            sys_region = SystemRegion(model.machine_timestep, len(keys))
            keys_region = KeyspacesRegion(keys,
                                          [KeyField({"cluster": "cluster"})])

            # Get the resources
            resources = {
                Cores: 1,
                SDRAM: region_utils.sizeof_regions([sys_region, keys_region],
                                                   None)
            }

            # Create the vertex
            v = self.connection_vertices[conn] = Vertex(get_application("rx"),
                                                        resources)
            self._sys_regions[v] = sys_region
            self._key_regions[v] = keys_region

        # Return the netlist specification
        return netlistspec(list(self.connection_vertices.values()),
                           load_function=self.load_to_machine)

    def load_to_machine(self, netlist, controller):
        """Load data to the machine."""
        # Write each vertex region to memory
        for vx in six.itervalues(self.connection_vertices):
            sys_mem, key_mem = region_utils.create_app_ptr_and_region_files(
                netlist.vertices_memory[vx],
                [self._sys_regions[vx], self._key_regions[vx]],
                None
            )

            self._sys_regions[vx].write_region_to_file(sys_mem)
            self._key_regions[vx].write_subregion_to_file(
                key_mem, cluster=vx.cluster)


class SystemRegion(Region):
    """System region for an SDP Rx vertex."""
    def __init__(self, machine_timestep, size_out):
        self.machine_timestep = machine_timestep
        self.size_out = size_out

    def sizeof(self, *args, **kwargs):
        return 8

    def write_region_to_file(self, fp, *args, **kwargs):
        """Write the region to file."""
        fp.write(struct.pack("<2I", self.machine_timestep, self.size_out))
