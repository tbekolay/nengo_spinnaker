import nengo
from nengo.utils.builder import full_transform

from .builder import (
    BuiltConnection, InputPort, Model, ObjectPort, OutputPort, spec
)


@Model.source_getters.register(nengo.base.NengoObject)
def generic_source_getter(model, conn):
    obj = model.object_intermediates[conn.pre_obj]
    return spec(ObjectPort(obj, OutputPort.standard))


@Model.sink_getters.register(nengo.base.NengoObject)
def generic_sink_getter(model, conn):
    obj = model.object_intermediates[conn.post_obj]
    return spec(ObjectPort(obj, InputPort.standard))


@Model.connection_parameter_builders.register(nengo.base.NengoObject)
def build_generic_connection_params(model, conn):
    transform = full_transform(conn)
    return BuiltConnection(
        decoders=None,
        transform=transform,
        eval_points=None,
        solver_info=None
    )
