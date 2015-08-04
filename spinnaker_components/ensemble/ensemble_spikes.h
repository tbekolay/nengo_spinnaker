/* Spiking Neural Input
 * --------------------
 *
 * Ensembles of neurons can receive spikes as input.
 *
 * Synaptic weight matrices are stored as a single dense matrix in SDRAM. On
 * receiving a spike the key used in the packet is used to determine which row
 * of this matrix to process to simulate receiving the spike.
 *
 * The row(s) activated by a single spike are determined by looking through an
 * array of `synapse_index_t`s. As multiple rows may be activated each row is
 * queued separately.
 */

#include <stdint.h>
#include "spin1_api.h"
#include "input_filtering.h"

#ifndef __ENSEMBLE_SPIKES_H__
#define __ENSEMBLE_SPIKES_H__

extern if_collection_t *synapse_filters;   // Collection of synaptic filters

extern bool _spike_transmit_spikes;     // Whether spikes should be transmitted
extern uint32_t _spike_population_key;  // Base key used to transmit all spikes
// NOTE: The base key will also include the offset of the first neuron in this
// core, so to form a key one should add the neuron ID to the base key.

/* A pseudo routing table entry which can be used to determine which synaptic
 * row should be retrieved for a given received spike packet. The actual row
 * should be determined by a combination of the block offset and the neuron ID.
 *
 *   row_index = block_offset + (key & neuron_mask);
 */
typedef struct _synapse_index_t
{
  uint32_t key;   // Key against which to compare the received packet
  uint32_t mask;  // Mask against which to compare the received packet

  uint32_t block_offset;  // Row offset for this region of the weight matrix
  uint32_t neuron_mask;   // Mask to get the neuron ID from the key
} synapse_index_t;

/* A pseudo routing table containing `synapse_index_t` elements.
 */
typedef struct _synapse_row_table_t
{
  uint32_t n_entries;      // Number of entries
  synapse_index_t *table;  // Array of entries
} synapse_row_table_t;


/* Prepare for receiving spikes through the network.
 *
 * NOTE: Will register some callback handlers for MC packets and DMA
 * completion.
 */
void spikes_prepare_rx(
  bool transmit_spikes, // Whether to send spikes or not
  uint32_t population_key,  // Base key used to transmit all spikes
  uint32_t *filter_data,  // Standard filter data for the synapses
  uint32_t *synaptic_rows_address,  // Address of the weight matrix
  uint32_t *row_data  // 1 word length + array of `synapse_index_t`
);

/* Transmit the spike for a neuron, stalling processing until it can be sent.
 */
static inline void transmit_spike(uint32_t neuron)
{
  // If we should transmit a spike to indicate the firing of the neuron then
  // repeatedly try to inject the spike into the communication fabric.
  if (_spike_transmit_spikes)
  {
    // Compute the key then send a packet with no payload, delay until the
    // packet can be sent.
    uint32_t key = _spike_population_key + neuron;

    while(!spin1_send_mc_packet(key, 0x0, NO_PAYLOAD))
    {
      spin1_delay_us(1);
    }
  }
}

/* Update all synaptic filters.
 */
static inline void spikes_update_synaptic_filters(void)
{
  input_filtering_step(synapse_filters);
}

/* Get the contribution to neuron input from spikes.
 */
static inline value_t get_neuron_spike_input(uint32_t neuron)
{
  // Return the input from the nth filter output
  return synapse_filters->output[neuron];
}

#endif  // __ENSEMBLE_SPIKES_H__
