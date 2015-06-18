/*!
 * \brief Spike recording
 *
 * \author Andrew Mundy <mundya@cs.man.ac.uk>
 *
 * \copyright Advanced Processor Technologies, School of Computer Science,
 *   University of Manchester
 * \copyright Computational Neuroscience Research Group, Centre for
 *   Theoretical Neuroscience, University of Waterloo
 */

#ifndef __RECORD_SPIKES_H__
#define __RECORD_SPIKES_H__

#include <stdbool.h>
#include "spin1_api.h"
#include "common-typedefs.h"
#include "nengo-common.h"

typedef struct spike_recording_buffer_t
{
  uint *buffer;         //!< The buffer to write to
  uint block_length;    //!< Size of 1 block of the buffer (in words)
  uint n_blocks;        //!< Length of the buffer in blocks (= n_ticks)

  bool record;          //!< Whether or not to record the data in the buffer

  uint *_sdram_start;   //!< Start of the buffer in SDRAM
  uint *_sdram_current; //!< Current location in the SDRAM buffer
} spike_recording_buffer_t;

/*!\brief Initialise a new recording buffer.
 */
bool record_spike_buffer_initialise(spike_recording_buffer_t *buffer,
  address_t region, uint n_blocks, uint n_neurons);

/*!\brief Reset the recording region for a new period of simulation.
 */
void record_spike_buffer_reset();

/*!\brief Flush the current buffer.
 *
 * The contents of the buffer will be appended to the recording region in
 * SDRAM, but only if recording is in use.
 */
static inline void record_spike_buffer_flush(spike_recording_buffer_t *buffer)
{
  // Copy the current buffer into SDRAM
  if (buffer->record) {
    spin1_memcpy(buffer->_sdram_current, buffer->buffer,
                 buffer->block_length * sizeof(uint));
  }

  // Empty the buffer
  for (uint i = 0; i < buffer->block_length; i++) {
    buffer->buffer[i] = 0;
  }

  // Progress the pointer
  buffer->_sdram_current += buffer->block_length;
}

/*!\brief Record a spike for the given neuron.
 */
static inline void record_spike(spike_recording_buffer_t *buffer, uint n_neuron)
{
  // Get the offset within the current buffer, and the specific bit to set
  // We write to the buffer regardless of whether recording is desired or not
  // in order to reduce branching.
  buffer->buffer[n_neuron >> 5] |= 1 << (n_neuron & 0x1f);
}

#endif  // __RECORD_SPIKES_H__
