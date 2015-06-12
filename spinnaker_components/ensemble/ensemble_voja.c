/*
 * Ensemble - Voja
 *
 * Authors:
 *   - James Knight <knightj@cs.man.ac.uk>
 * 
 * Copyright:
 *   - Advanced Processor Technologies, School of Computer Science,
 *      University of Manchester
 *   - Computational Neuroscience Research Group, Centre for
 *      Theoretical Neuroscience, University of Waterloo
 * 
 */

#include "ensemble_voja.h"
#include "ensemble_filtered_activity.h"

#include <string.h>


//----------------------------------
// Structs
//----------------------------------
// Structure containing parameters and state required for Voja learning
typedef struct voja_parameters_t
{
  // Scalar learning rate used in Voja encoder delta calculation
  value_t learning_rate;

  // Index of the input signal filter that contains
  // learning signal. -1 if there is no learning signal
  int32_t learning_signal_filter_index;

  // Offset into encoder to apply Voja
  uint32_t encoder_offset;

  // Index of the input signal filter than contains
  // the decoded input from the pre-synaptic ensemble
  uint32_t decoded_input_filter_index;

  // Index of the activity filter to extract input from
  uint32_t activity_filter_index;
} voja_parameters_t;

//-----------------------------------------------------------------------------
// Global variables
//-----------------------------------------------------------------------------
static uint32_t g_num_voja_learning_rules = 0;
static voja_parameters_t *g_voja_learning_rules = NULL;

//-----------------------------------------------------------------------------
// Global functions
//-----------------------------------------------------------------------------
bool get_voja(address_t address)
{
  // Read number of Voja learning rules that are configured
  g_num_voja_learning_rules = address[0];
  
  io_printf(IO_BUF, "Voja learning: Num rules:%u\n", g_num_voja_learning_rules);
  
  if(g_num_voja_learning_rules > 0)
  {
    // Allocate memory
    MALLOC_FAIL_FALSE(g_voja_learning_rules,
                      g_num_voja_learning_rules * sizeof(voja_parameters_t));
    
    // Copy learning rules from region into new array
    memcpy(g_voja_learning_rules, &address[1], g_num_voja_learning_rules * sizeof(voja_parameters_t));
    
    // Display debug
    for(uint32_t l = 0; l < g_num_pes_learning_rules; l++)
    {
      const voja_parameters_t *parameters = &g_voja_learning_rules[l];
      //io_printf(IO_BUF, "\tRule %u, Learning rate:%k, Error signal filter index:%u, Decoder output offset:%u, Activity filter index:%u\n",
      //         l, parameters->learning_rate, parameters->error_signal_filter_index, parameters->decoder_output_offset, parameters->activity_filter_index);
    }
  }
  return true;
}
//-----------------------------------------------------------------------------
void voja_step()
{
  // Loop through all the learning rules
  for(uint32_t l = 0; l < g_num_voja_learning_rules; l++)
  {
    // Extract filtered error signal vector indexed by learning rule
    const voja_parameters_t *parameters = &g_voja_learning_rules[l];

    // Extract learning rate from parameters and, if specified, multiply by current learning signal value
    value_t learning_rate = parameters->learning_rate;
    if(parameters->learning_signal_filter_index != -1)
    {
      learning_rate *= g_input_modulatory.filters[parameters->learning_signal_filter_index];
    }

    // Extract decoded input signal from filter
    const filtered_input_buffer_t *decoded_input = g_input.filters[parameters->decoded_input_filter_index];
    const value_t *decoded_input_signal = decoded_input->filtered;

    // Extract filtered activity vector indexed by learning rule
    const value_t *filtered_activity = g_filtered_activities[parameters->activity_filter_index];

    // Loop through neurons
    for(uint n = 0; n < g_ensemble.n_neurons; n++)
    {
      // Get this neuron's encoder vector, offset by the encoder offset
      value_t *encoder_vector = neuron_encoder_vector(n) + parameters->encoder_offset;

      // Loop through input dimensions
      for(uint d = 0; d < decoded_input->d_in; d++)
      {
        encoder_vector[d] += learning_rate * filtered_activity[n] * (decoded_input_signal[d] - encoder_vector[d]))
      }
    }
  }
}