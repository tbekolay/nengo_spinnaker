import itertools
import nengo
import numpy as np
import pytest
import struct
import tempfile

from nengo_spinnaker.operators import lif
from nengo_spinnaker.utils import type_casts as tp


class TestEnsembleLIF(object):
    @pytest.mark.parametrize("size_in", [1, 4, 5])
    def test_init(self, size_in):
        """Test that creating an Ensemble LIF creates an empty list of local
        probes and an empty input vector.
        """
        # Create an ensemble
        ens = nengo.Ensemble(100, size_in, add_to_container=False)

        op = lif.EnsembleLIF(ens)
        assert op.ensemble is ens
        assert np.all(op.direct_input == np.zeros(size_in))
        assert op.local_probes == list()


class TestSystemRegion(object):
    """Test system regions for Ensembles."""
    def test_sizeof(self):
        region = lif.SystemRegion(1, 2, 5, 1000, 0.01, 0.02, 0.001, False, True)
        assert region.sizeof() == 10 * 4  # 10 words
        assert region.sizeof_padded(slice(None)) == region.sizeof(slice(None))

    @pytest.mark.parametrize(
        "vertex_slice, vertex_neurons",
        [(slice(0, 100), 100),
         (slice(100, 120), 20),
         ]
    )
    @pytest.mark.parametrize(
        "machine_timestep, dt, size_in, encoder_width, tau_ref, tau_rc, "
        "size_out, probe_spikes, probe_encoders",
        [(1000, 0.001, 5, 5, 0.0, 0.002, 7, True, False),
         (10000, 0.01, 1, 2, 0.001, 0.02, 3, False, True),
         ]
    )
    def test_write_subregion_to_file(self, size_in, encoder_width, size_out,
                                     machine_timestep, tau_ref, tau_rc, dt,
                                     probe_spikes, probe_encoders,
                                     vertex_slice, vertex_neurons):
        # Check that the region is correctly written to file
        region = lif.SystemRegion(
            size_in, encoder_width, size_out, machine_timestep,
            tau_ref, tau_rc, dt, probe_spikes, probe_encoders
        )

        # Create the file
        fp = tempfile.TemporaryFile()

        # Write to it
        region.write_subregion_to_file(fp, vertex_slice)

        # Read back and check that the values are sane
        fp.seek(0)
        values = fp.read()
        assert len(values) == region.sizeof()

        (n_in, n_encoder, n_out, n_n, m_t, t_ref, dt_over_t_rc, rec_spikes, rec_encoders, i_dims) = \
            struct.unpack_from("<10I", values)
        assert n_in == size_in
        assert n_encoder == encoder_width
        assert n_out == size_out
        assert n_n == vertex_neurons
        assert m_t == machine_timestep
        assert t_ref == int(tau_ref // dt)
        assert (tp.value_to_fix(-np.expm1(-dt / tau_rc)) * 0.9 < dt_over_t_rc <
                tp.value_to_fix(-np.expm1(-dt / tau_rc)) * 1.1)
        assert ((probe_spikes and rec_spikes != 0) or
                (not probe_spikes and rec_spikes == 0))
        assert ((probe_encoders and rec_encoders != 0) or
                (not probe_encoders and rec_encoders == 0))
        assert i_dims == 1


class TestPESRegion(object):
    @pytest.mark.parametrize(
        "n_learning_rules",
        [0, 1, 2]
    )
    def test_sizeof(self, n_learning_rules):
        region = lif.PESRegion()

        learning_rule = lif.PESLearningRule(1e-4, 1, 2, -1)
        region.learning_rules.extend(
            itertools.repeat(learning_rule, n_learning_rules))

        assert region.sizeof() == 4 + (n_learning_rules * 16)

class TestSpikeRecordingRegion(object):
    """Spike regions use 1 bit per neuron per timestep but pad each frame to a
    multiple of words.
    """
    @pytest.mark.parametrize(
        "n_steps, vertex_slice, words_per_frame",
        [(1, slice(0, 2), 1),
         (100, slice(0, 32), 1),
         (1000, slice(0, 33), 2),
         ]
    )
    def test_sizeof(self, n_steps, vertex_slice, words_per_frame):
        # Create the region
        sr = lif.SpikeRecordingRegion(n_steps)

        # Check that the size is reported correctly
        assert sr.sizeof(vertex_slice) == 4 * words_per_frame * n_steps
