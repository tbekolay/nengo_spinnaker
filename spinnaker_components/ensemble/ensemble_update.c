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
#include "ensemble_profiler.h"
#include "ensemble_voja.h"

void ensemble_update(uint ticks, uint arg1)
{
  use(arg1);
  
  if (simulation_ticks != UINT32_MAX && ticks >= simulation_ticks)
  {
    profiler_finalise();
    spin1_exit(0);
  }

  profiler_write_entry(PROFILER_ENTER | PROFILER_TIMER);

  // Filter inputs, updating accumulator for excitatory and inhibitory inputs
  profiler_write_entry(PROFILER_ENTER | PROFILER_TIMER_INPUT_FILTER);
  input_filtering_step(&g_input);
  input_filtering_step(&g_input_inhibitory);
  input_filtering_step_no_accumulate(&g_input_modulatory);
  input_filtering_step_no_accumulate(&g_input_learnt_encoder);
  profiler_write_entry(PROFILER_EXIT | PROFILER_TIMER_INPUT_FILTER);

  // Filter activity
  filtered_activity_step();

  // Compute the inhibition
  value_t inhibitory_input = 0;
  for (uint d = 0; d < g_ensemble.n_inhib_dims; d++)
  {
    inhibitory_input += g_input_inhibitory.output[d];
  }

  profiler_write_entry(PROFILER_ENTER | PROFILER_TIMER_NEURON);

  // Perform neuron updates
  for( uint n = 0; n < g_ensemble.n_neurons; n++ )
  {
    // Is this neuron in its refractory period
    bool in_refractory_period = (neuron_refractory(n) != 0);

    // Include neuron bias
    current_t i_membrane = (g_ensemble.i_bias[n] +
      inhibitory_input * g_ensemble.gain[n]);

    // Extract this neurons encoder vector
    const value_t *encoder_vector = neuron_encoder_vector(n);

    // Loop through learnt input signals and encoder slices
    uint f = 0;
    uint e = g_input.output_size;
    for(; f < g_input_learnt_encoder.n_filters; f++, e += g_input.output_size)
    {
      // Extract input signal from learnt encoder filter
      const if_filter_t *filtered_input = &g_input_learnt_encoder.filters[f];

      // Get encoder vector for this neuron offset for correct learnt encoder
      const value_t *learnt_encoder_vector = encoder_vector + e;

      // Record learnt encoders
      // **NOTE** idea here is that by interspersing these between encoding
      // operations, write buffer should have time to be written out
      record_learnt_encoders(&g_ensemble.record_learnt_encoders,
        g_input.output_size, learnt_encoder_vector);

      // If neuron's not in refractory period, loop through filter
      // dimensions and apply input encoded by learnt encoders
      if(!in_refractory_period)
      {
        for(uint d = 0; d < filtered_input->size; d++)
        {
          i_membrane += learnt_encoder_vector[d] * filtered_input->output[d];
        }
      }
    }

    // If this neuron is refractory then skip any further processing
    if(in_refractory_period)
    {
      decrement_neuron_refractory( n );
      continue;
    }

    // Encode the input and add to the membrane current
    for(uint d = 0; d < g_input.output_size; d++)
    {
      i_membrane += encoder_vector[d] * g_ensemble.input[d];
    }

    value_t v_voltage = neuron_voltage(n);
    value_t v_delta = (i_membrane - v_voltage) * g_ensemble.exp_dt_over_t_rc;

    // Voltages can't go below 0.0
    v_voltage += v_delta;
    if(v_voltage < 0.0k)
    {
      v_voltage = 0.0k;
    }

    // Save state
    set_neuron_voltage(n, v_voltage);

    // If this neuron has fired then process
    if( v_voltage > 1.0k )
    {
      // Set the voltage to be the overshoot, set the refractory time
      set_neuron_refractory(n);
      set_neuron_voltage(n, v_voltage - 1.0k);

      // Decrement the refractory time in the case that the overshoot was
      // sufficiently significant.
      if(v_voltage > 2.0k)
      {
        decrement_neuron_refractory(n);
        set_neuron_voltage(n, v_voltage - 1.0k - v_delta);
      }

      // Update the output values
      const value_t *decoder_vector = neuron_decoder_vector(n);
      for( uint d = 0; d < g_n_output_dimensions; d++ )
      {
        g_ensemble.output[d] += decoder_vector[d];
      }

      // Record that the spike occurred
      record_spike(&g_ensemble.record_spikes, n);

      // Apply effect of neuron spiking to filtered activities
      filtered_activity_neuron_spiked(n);

      // Update non-filtered learning rules
      pes_neuron_spiked(n);
      voja_neuron_spiked(n);
    }
  }

  profiler_write_entry(PROFILER_EXIT | PROFILER_TIMER_NEURON);

  // Update filtered learning rules
  pes_step();
  voja_step();

  profiler_write_entry(PROFILER_ENTER | PROFILER_TIMER_OUTPUT);

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

  profiler_write_entry(PROFILER_EXIT | PROFILER_TIMER_OUTPUT);

  // Flush the spike recording buffer
  record_buffer_flush(&g_ensemble.record_spikes);

  profiler_write_entry(PROFILER_EXIT | PROFILER_TIMER);
}
