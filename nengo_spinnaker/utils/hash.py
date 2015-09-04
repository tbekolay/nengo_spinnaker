def hash_slice(sl):
    """Create a unique hash of a slice object."""
    if isinstance(sl, slice):
        return hash((slice, sl.start, sl.stop, sl.step))
    elif sl is None:
        return hash(None)
    else:
        print(sl)
        return hash(tuple(sl))
