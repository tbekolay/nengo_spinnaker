#include <stdbool.h>
#include "ensemble.h"
#include "ensemble_spikes.h"
#include "common-impl.h"
#include "input_filtering.h"
#include "spin1_api.h"

// Settings
#define SPIKE_QUEUE_LENGTH 256  // Queue of spikes to process
#define ROW_QUEUE_LENGTH 8      // Queue of rows to process

// DMA tags
enum {
  READ_SYNAPTIC_ROW
} DmaTag;

// Globals
bool spike_pipeline = false;  // Indicates that spikes are(n't) being processed

bool _spike_transmit_spikes = false;  // Whether spikes should be transmitted
uint32_t _spike_population_key = 0;   // Base key used to transmit all spikes
// NOTE: The base key will also include the offset of the first neuron in this
// core, so to form a key one should add the neuron ID to the base key.

synapse_row_table_t row_table;  // Routing information for spikes

uint32_t *synaptic_rows;      // Start address of weight matrices in memory
uint32_t next_buffer = 0;     // Index of next synaptic buffer
uint32_t **synaptic_buffers;  // Buffer for row of weight matrix

if_collection_t *synapse_filters;   // Collection of synaptic filters

// ----------------------------------------------------------------------------
/* Generic stacks of uint32_ts -- used to store spikes and rows which need
 * processing, may be switched out for circular buffers at some later stage.
 */
typedef struct _uint32_queue_t
{
  uint32_t head;       // Current head of the stack
  uint32_t max_length; // Maximum length of the stack
  uint32_t *stack;     // The actual storage
} uint32_queue_t;

// Clear a queue
static inline void _uint32_queue_clear(uint32_queue_t *queue)
{
  queue->head = 0;  // Empty the stack
}

// Append to a queue
// Return value indicates if the append was successful.
static inline bool _uint32_queue_append(uint32_queue_t *queue, uint32_t value)
{
  // If we can add an item to the stack
  if (queue->head < queue->max_length)
  {
    // Add the item to the stack and indicate success
    queue->stack[queue->head++] = value;
    return true;
  }
  // Otherwise return false to indicate that we couldn't append to the queue
  return false;
}

// Pop an item from a queue
// Return value indicates if the pop was successful.
static inline bool _uint32_queue_pop(uint32_queue_t *queue, uint32_t *dest)
{
  // If the stack contains anything
  if (queue->head)
  {
    *dest = queue->stack[queue->head--];
    return true;
  }
  // Otherwise indicate that the stack was empty
  return false;
}

// Check if the queue is empty
static inline bool _uint32_queue_empty(uint32_queue_t *queue)
{
  return (queue->head == 0);
}

// ----------------------------------------------------------------------------
/* Specific queues and failure counters for spikes and rows */
uint32_queue_t _spike_queue;
uint32_t _spike_queue_data[SPIKE_QUEUE_LENGTH];
uint32_t _spike_queue_unqueued = 0, _spike_queue_unprocessed = 0;

void _spike_queue_prep(void)
{
  // Link everything together and initialise
  _spike_queue.head = 0;
  _spike_queue.max_length = SPIKE_QUEUE_LENGTH;
  _spike_queue.stack = _spike_queue_data;
}

// Queue a spike for processing
static inline void _queue_spike(uint32_t spike)
{
  // If we can't queue the spike then record the failure
  if (!_uint32_queue_append(&_spike_queue, spike))
  {
    _spike_queue_unqueued++;
  }
}

// Retrieve a spike from the queue
static inline bool _get_spike(uint32_t *spike)
{
  // Get a spike from the queue
  return _uint32_queue_pop(&_spike_queue, spike);
}

// Clear the spike queue
static inline void _clear_spikes(void)
{
  _spike_queue_unprocessed += _spike_queue.head;
  _uint32_queue_clear(&_spike_queue);
}

// Determine if there are any spikes to process
static inline bool _no_spikes(void)
{
  return _uint32_queue_empty(&_spike_queue);
}

uint32_queue_t _row_queue;
uint32_t _row_queue_data[ROW_QUEUE_LENGTH];
uint32_t _row_queue_unqueued = 0, _row_queue_unprocessed = 0;

void _row_queue_prep(void)
{
  // Link everything together and initialise
  _row_queue.head = 0;
  _row_queue.max_length = SPIKE_QUEUE_LENGTH;
  _row_queue.stack = _row_queue_data;
}

// Queue a row for processing
static inline void _queue_row(uint32_t row)
{
  // If we can't queue the row then record the failure
  if (!_uint32_queue_append(&_row_queue, row))
  {
    _row_queue_unqueued++;
  }
}

// Retrieve a row from the queue
static inline bool _get_row(uint32_t *row)
{
  // Get a row from the queue
  return _uint32_queue_pop(&_row_queue, row);
}

// Clear the row queue
static inline void _clear_rows(void)
{
  _row_queue_unprocessed = _row_queue.head;
  _uint32_queue_clear(&_row_queue);
}

// Determine if there are any rows to process
static inline bool _no_rows(void)
{
  return _uint32_queue_empty(&_row_queue);
}

// ----------------------------------------------------------------------------
/* Process a spike
 * This will get the next spike from the queue of spikes to process, will
 * unpack all of the rows this spike activates and adds those the the queue of
 * rows to process.
 */
static inline void process_spike(synapse_row_table_t *rows)
{
  // Try and get a spike to process
  uint32_t spike;
  if (_get_spike(&spike))
  {
    // If there was a spike then we look up the rows associated with it in the
    // synapse index table.
    for (uint32_t n = 0; n < rows->n_entries; n++)
    {
      // If this entry matches the spike then add the row information to the
      // queue of rows to process.
      synapse_index_t *entry = &rows->table[n];
      if ((spike & entry->mask) == entry->key)
      {
        _queue_row(entry->block_offset + (spike & entry->neuron_mask));
      }
    }
  }
}

// ----------------------------------------------------------------------------
/* Start processing spikes and retrieving synaptic rows from SDRAM.
 */
void start_spike_pipeline(uint arg0, uint arg1)
{
  // Indicate that spike processing is taking place
  spike_pipeline = true;

  // We use neither of the arguments
  use(arg0);
  use(arg1);

  // If there are no rows to process then we process the first spike
  if (_no_rows())
  {
    // Unpack a spike to determine which rows need to be processed
    process_spike(&row_table);
  }

  // Get a row to process and schedule a DMA event to allow to be processed.
  uint32_t row_index;
  if (_get_row(&row_index))
  {
    // If there was a row to process then use the index of the row to launch a
    // DMA request.
    uint32_t *row_address = \
      synaptic_rows + g_ensemble.n_neurons*row_index*sizeof(uint32_t);

    // Schedule the DMA
    spin1_dma_transfer(
      READ_SYNAPTIC_ROW, synaptic_buffers[next_buffer], row_address, DMA_READ,
      (g_ensemble.n_neurons + 1) * sizeof(uint32_t)
    );
  }
  else
  {
    // Indicate that no spike processing is taking place
    spike_pipeline = false;
  }
}

// ----------------------------------------------------------------------------
/* Receive a spike
 * This will queue a spike for processing, the processing will unpack a spike
 * into the rows of the synaptic matrix which need reading back and including
 * in input to synaptic filters.
 */
void receive_spike(uint key, uint arg1)
{
  // No payload, so ignore `arg1`
  use(arg1);

  // Add the spike to the list of spikes to unpack and queue
  _queue_spike((uint32_t) key);

  // If there isn't a running spike pipeline ensure we start one.
  if (!spike_pipeline)
  {
    spin1_schedule_callback(start_spike_pipeline, 0, 0, 1);
  }
}

// ----------------------------------------------------------------------------
/* Read a synaptic row
 * This will read a synaptic row which has been returned by DMA from SDRAM. The
 * first word in the row will indicate which synaptic filter should be used for
 * the values in the row.  There is one word (a `value_t`) for each neuron in
 * the local population.
 */
void read_synaptic_row(uint transfer_id, uint tag)
{
  // We don't use the transfer_id and we just exit if the tag isn't a read from
  // DMA.
  use(transfer_id);
  if (tag != READ_SYNAPTIC_ROW)
  {
    return;
  }

  // Store the current buffer and update to use the next buffer for the next
  // transfer.
  uint32_t *synaptic_buffer = synaptic_buffers[next_buffer];
  next_buffer ^= 0x1;

  // Start the next transfer, ideally that transfer should be hidden behind the
  // following processing.
  spin1_schedule_callback(start_spike_pipeline, 0, 0, 1);

  // Use the first word to get the filter index, then accumulate all the other
  // contributions in the accumulator for that filter.
  if_filter_t filter = synapse_filters->filters[synaptic_buffer[0]];
  value_t *data = (value_t *) &synaptic_buffer[1];
  for (uint32_t n = 0; n < g_ensemble.n_neurons; n++)
  {
    // The input for the synapse feeding neuron n is "dimension" n of the
    // filter.
    _if_filter_input(&filter, n, data[n]);
  }
}

// ----------------------------------------------------------------------------
/* Load in synaptic routing data
 */
void prepare_synaptic_routing(uint32_t *data, synapse_row_table_t *table)
{
  // The number of routes should be the first, this will allow use to allocate
  // sufficient memory for the rest of the data.
  table->n_entries = data[0];
  MALLOC_OR_DIE(table->table, sizeof(synapse_index_t)*table->n_entries);

  // Copy the entries into the table
  spin1_memcpy(table->table, &data[1],
               sizeof(synapse_index_t)*table->n_entries);
}

// ----------------------------------------------------------------------------
/* Prepare for receiving spikes.
 */
void spikes_prepare_rx(
  bool transmit_spikes,
  uint32_t population_key,
  uint32_t *filter_data,
  uint32_t *synaptic_rows_address,
  uint32_t *row_data
)
{
  // Store basic parameters
  _spike_transmit_spikes = transmit_spikes;
  _spike_population_key = population_key;

  // Prepare queues for receiving spikes
  _spike_queue_prep();
  _row_queue_prep();

  // Read in synaptic filters
  input_filtering_get_filters(synapse_filters, filter_data);
  input_filtering_initialise_output(synapse_filters, g_ensemble.n_neurons);

  // Store the location of the synaptic rows in memory
  synaptic_rows = synaptic_rows_address;

  // Load in the synaptic routing information
  prepare_synaptic_routing(row_data, &row_table);

  // Register listeners for DMA events and MC (no payload) events
  spin1_callback_on(MC_PACKET_RECEIVED, receive_spike, 0);
  spin1_callback_on(DMA_TRANSFER_DONE, read_synaptic_row, 1);
}
