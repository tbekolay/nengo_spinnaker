from nengo.base import ObjView


def probe_target(probe):
    """Get the target object of a probe."""
    if isinstance(probe.target, ObjView):
        # If the target is an object view then return the underlying object
        return probe.target.obj
    else:
        # Otherwise return the target
        return probe.target
