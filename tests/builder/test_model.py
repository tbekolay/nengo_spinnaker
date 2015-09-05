import numpy as np

from nengo_spinnaker.builder import model


class TestTransmissionParameters(object):
    """TransmissionParameters should support __hash__ and __eq__ on their
    contents.
    """
    def test_eq_and_hash(self):
        # Create several TransmissionParameters and ensure that they only hash
        # and report equal when they are actually equal. Create a derived class
        # and ensure that this is also taken into account.
        tp0 = model.TransmissionParameters(slice(0, 3), np.eye(5))
        tp1 = model.TransmissionParameters(slice(0, 3), np.eye(5))
        tp2 = model.TransmissionParameters(slice(0, 3), np.ones((3, 3)))
        tp3 = model.TransmissionParameters(slice(0, 3), np.ones((5, 5)))
        tp4 = model.TransmissionParameters(slice(0, 4), np.eye(5))

        class MyTransmissionParameters(model.TransmissionParameters):
            pass

        tp5 = MyTransmissionParameters(slice(0, 3), np.array([1]))

        # Test equality
        assert tp0 == tp1
        for tp in [tp2, tp3, tp4, tp5]:
            assert tp0 != tp
            assert tp1 != tp

        # Test hashing
        assert hash(tp0) == hash(tp1)
        for tp in [tp2, tp3, tp4, tp5]:
            assert hash(tp0) != hash(tp)
            assert hash(tp1) != hash(tp)
