/*
 * Authors:
 *   - Andrew Mundy <mundya@cs.man.ac.uk>
 *   - Terry Stewart
 * 
 * Copyright:
 *   - Advanced Processor Technologies, School of Computer Science,
 *      University of Manchester
 *   - Computational Neuroscience Research Group, Centre for
 *      Theoretical Neuroscience, University of Waterloo
 */

#include "ensemble.h"
#include "ensemble_filtered_activity.h"
#include "ensemble_output.h"
#include "ensemble_pes.h"
#include "ensemble_voja.h"

static uint32_t lfsr = 1;                   //!< LFSR for spike perturbation

void ensemble_update(uint ticks, uint arg1)
{
  use(arg1);
  if (simulation_ticks != UINT32_MAX && ticks >= simulation_ticks)
  {
    spin1_exit(0);
  }

  // Filter inputs, updating accumulator for excitatory and inhibitory inputs
  input_filter_step(&g_input, true);
  input_filter_step(&g_input_inhibitory, true);
  input_filter_step(&g_input_modulatory, false);
  input_filter_step(&g_input_learnt_encoder, false);

  // Filter activity
  filtered_activity_step();

  // Compute the inhibition
  value_t inhibitory_input = 0;
  for (uint d = 0; d < g_ensemble.n_inhib_dims; d++)
  {
    inhibitory_input += g_input_inhibitory.input[d];
  }

  // Perform neuron updates
  for( uint n = 0; n < g_ensemble.n_neurons; n++ )
  {
    // If this neuron is refractory then skip any further processing
    if( neuron_refractory( n ) != 0 )
    {
      decrement_neuron_refractory( n );
      continue;
    }

    // Include neuron bias
    current_t i_membrane = (g_ensemble.i_bias[n] +
      inhibitory_input * g_ensemble.inhib_gain[n]);

    // Extract this neurons encoder vector
    const value_t *encoder_vector = neuron_encoder_vector(n);

    // Encode the input and add to the membrane current
    for(uint d = 0; d < g_input.n_dimensions; d++)
    {
      i_membrane += encoder_vector[d] * g_ensemble.input[d];
    }

    // Loop through learnt input signals and encoder slices
    uint f = 0;
    uint e = g_input.n_dimensions;
    for(; f < g_input_learnt_encoder.n_filters; f++, e += g_input.n_dimensions)
    {
      // Extract input signal from learnt encoder filter
      const filtered_input_buffer_t *filtered_input = g_input_learnt_encoder.filters[f];
      const value_t *filtered_input_signal = filtered_input->filtered;

      // Get encoder vector for this neuron offset for correct learnt encoder
      const value_t *learnt_encoder_vector = encoder_vector + e;

      // Record learnt encoders
      // **NOTE** idea here is that by interspersing these between encoding
      // operations, write buffer should get a chance to be flushed
      record_learnt_encoders(&g_ensemble.record_learnt_encoders,
        g_input.n_dimensions, learnt_encoder_vector);

      // Loop through filter dimensions
      for(uint d = 0; d < filtered_input->d_in; d++)
      {
        // Encode filtered input signal
        i_membrane += learnt_encoder_vector[d] * filtered_input_signal[d];
      }
    }

    voltage_t v_voltage = neuron_voltage(n);
    voltage_t v_delta = ( i_membrane - v_voltage ) * g_ensemble.dt_over_t_rc;
    /* io_printf( IO_STD, "n = %d, J = %k, V = %k, dV = %k\n",
                  n, i_membrane, v_voltage, v_delta );
    */

    v_voltage += v_delta;

    // Voltages can't go below 0.0
    if( v_voltage < 0.0k )
    {
      v_voltage = 0.0k;
    }

    // Save state
    set_neuron_voltage( n, v_voltage );

    // If this neuron has fired then process
    if( v_voltage > 1.0k )
    {
      //io_printf( IO_STD, "[Ensemble] Neuron %d spiked.", n );

      // Zero the voltage, set the refractory time
      set_neuron_refractory( n );
      set_neuron_voltage(n, 0.0k);

      /* Randomly perturb the refractory period to account for inter-tick
         spiking.*/
      if(kbits(lfsr & 0x00007fff) * v_delta < v_voltage - 1.0k)
      {
        decrement_neuron_refractory( n );
      }
      lfsr = (lfsr >> 1) ^ ((-(lfsr & 0x1)) & 0xB400);

      // Update the output values
      const value_t *decoder_vector = neuron_decoder_vector(n);
      for( uint d = 0; d < g_n_output_dimensions; d++ )
      {
        /* io_printf( IO_STD, "[%d] = %.3k (0x%08x)",
          d, neuron_decoder(n,d), neuron_decoder(n,d) ); */
        g_ensemble.output[d] += decoder_vector[d];
      }
      //io_printf( IO_STD, "\n" );

      // Record that the spike occurred
      record_spike(&g_ensemble.record_spikes, n);

      // Apply effect of neuron spiking to filtered activities
      filtered_activity_neuron_spiked(n);

      // Update non-filtered learning rules
      pes_neuron_spiked(n);
      voja_neuron_spiked(n);
    }
  }

  // Update filtered learning rules
  pes_step();
  voja_step();

  // Transmit decoded Ensemble representation
  for (uint output_index = 0; output_index < g_n_output_dimensions;
       output_index++)
  {
    while(!spin1_send_mc_packet(gp_output_keys[output_index],
                                bitsk(gp_output_values[output_index]),
                                WITH_PAYLOAD))
    {
      spin1_delay_us(1);
    }
    gp_output_values[output_index] = 0;
  }

  // Flush the spike recording buffer
  record_spike_buffer_flush(&g_ensemble.record_spikes);
}
