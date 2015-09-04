import nengo
from nengo.utils.builder import full_transform

from .builder import (
    BuiltConnection, InputPort, Model, ObjectPort, OutputPort, spec
)
from .model import TransmissionParameters, ReceptionParameters


@Model.source_getters.register(nengo.base.NengoObject)
def generic_source_getter(model, conn):
    obj = model.object_operators[conn.pre_obj]
    return spec(ObjectPort(obj, OutputPort.standard))


@Model.sink_getters.register(nengo.base.NengoObject)
def generic_sink_getter(model, conn):
    obj = model.object_operators[conn.post_obj]
    return spec(ObjectPort(obj, InputPort.standard))


@Model.connection_parameter_builders.register(nengo.base.NengoObject)
def build_generic_connection_params(model, conn):
    return BuiltConnection(
        decoders=None,
        transform=full_transform(conn, slice_pre=False, allow_scalars=False),
        eval_points=None,
        solver_info=None
    )


def build_generic_transmission_params(model, conn):
    """Build parameters necessary for transmitting packets to simulate this
    connection.
    """
    # We return the full transform in a form that guarantees that it is a
    # matrix and we include the pre_slice separately.
    return TransmissionParameters(
        conn.pre_slice,
        full_transform(conn, slice_pre=False, allow_scalars=False)
    )


def build_generic_reception_params(model, conn):
    """Build parameters necessary for receiving packets that simulate this
    connection.
    """
    # Just extract the synapse from the connection.
    return ReceptionParameters(conn.synapse)
