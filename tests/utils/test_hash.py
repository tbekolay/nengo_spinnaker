from nengo_spinnaker.utils.hash import hash_slice


def test_hash_slice():
    """Test hashing of slices."""
    sl_a = slice(10, 15, 3)
    sl_b = slice(10, 15, 3)
    slices = [slice(0, 15, 3), slice(10, 12, 3), slice(10, 15),
              None, [0, 1, 2]]

    assert hash_slice(sl_a) == hash_slice(sl_b)
    for sl in slices:
        assert hash_slice(sl_a) != hash_slice(sl)
