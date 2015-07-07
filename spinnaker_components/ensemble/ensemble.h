/**
 * \addtogroup Ensemble
 * \brief An implementation of the Nengo LIF neuron with multidimensional
 *        input capabilities.
 *
 * The Ensemble component implements a LIF neuron model which accepts and
 * transmits multidimensional values.  As in the NEF each neuron in the
 * Ensemble has an *Encoder* which is provided by the Nengo framework running
 * on the host. On each time step the encoders are used to convert the real
 * value presented to the ensemble into currents applied to input of each
 * simulated neuron. Spikes are accumulated and converted into real values
 * using *decoders* (again provided by the host). Decoded values are output
 * in an interleaved fashion during the neuron update loop.
 *
 * \author Andrew Mundy <mundya@cs.man.ac.uk>
 * \author Terry Stewart
 * \author James Knight <knightj@cs.man.ac.uk>
 * 
 * \copyright Advanced Processor Technologies, School of Computer Science,
 *   University of Manchester
 * \copyright Computational Neuroscience Research Group, Centre for
 *   Theoretical Neuroscience, University of Waterloo
 * @{
 */

#ifndef __ENSEMBLE_H__
#define __ENSEMBLE_H__

#include "spin1_api.h"
#include "stdfix-full-iso.h"
#include "common-impl.h"

#include "nengo_typedefs.h"
#include "nengo-common.h"

#include "dimensional-io.h"
#include "record_spikes.h"
#include "record_learnt_encoders.h"
#include "input_filter.h"

/* Structs ******************************************************************/
/** \brief Representation of system region. See ::data_system. */
// **NOTE** it's important to used sized types here!
typedef struct region_system 
{
  uint32_t n_input_dimensions;
  uint32_t encoder_width;
  uint32_t n_output_dimensions;
  uint32_t n_neurons;
  uint32_t machine_timestep;
  uint32_t t_ref;
  value_t exp_dt_over_t_rc;
  uint32_t record_spikes;
  uint32_t record_learnt_encoders;
  uint32_t n_inhibitory_dimensions;
} region_system_t;

/** \brief Shared ensemble parameters.
  */
typedef struct ensemble_parameters
{
  //! Number of neurons \f$N\f$
  uint n_neurons;

  //! Machine time step  / useconds
  uint machine_timestep;

  //! Refractory period \f$\tau_{ref} - 1\f$ / steps
  uint t_ref;
  value_t exp_dt_over_t_rc;    //!< \f$-\exp{\frac{dt}{\tau_{rc}}}\$

  current_t *i_bias;                      //!< Population biases \f$1 \times N\f$
  value_t *neuron_voltage;                //!< Neuron voltages
  uint8_t *neuron_refractory;             //!< Refractory states

  uint n_inhib_dims;                      //!< Number of dimensions in inhibitory connection
  value_t *gain;                          //!< Per-neuron gain (value of transform)

  uint encoder_width;
  value_t *encoders;                      //!< Encoder values \f$N \times D_{in}\f$ (including gains)
  value_t *decoders;                      //!< Decoder values \f$N \times\sum D_{outs}\f$

  //! Input buffer
  value_t *input;

  //! Output buffer
  value_t *output;

  //! Spike recording buffer
  spike_recording_buffer_t record_spikes;

  //! Learnt encoder recording buffer
  encoder_recording_buffer_t record_learnt_encoders;
} ensemble_parameters_t;


/* Parameters and Buffers ***************************************************/
extern ensemble_parameters_t g_ensemble;  //!< Global parameters
extern uint g_output_period;              //!< Delay in transmitting decoded output

extern uint g_n_output_dimensions;

// Input filters and buffers for general and inhibitory inputs. Their outputs
// are summed into accumulators which are used to drive the standard neural input
extern input_filter_t g_input;
extern input_filter_t g_input_inhibitory;

// Input filters and buffers for modulatory signals. Their
// outputs are left seperate for use by learning rules
extern input_filter_t g_input_modulatory;

// Input filters and buffers for signals to be encoded by learnt encoders.
// Each output is encoded by a seperate encoder so these are also left seperate
extern input_filter_t g_input_learnt_encoder;

/* Functions ****************************************************************/
/**
 * \brief Initialise the ensemble.
 */
bool initialise_ensemble(
  region_system_t *pars  //!< Pointer to formatted system region
);

/**
 * \brief Filter input values, perform neuron update and transmit any output
 *        packets.
 * \param arg0 Unused parameter
 * \param arg1 Unused parameter
 *
 * Neurons are then simulated using Euler's Method as in most implementations
 * of the NEF.  When a neuron spikes it is immediately decoded and its
 * contribution to the output of the Ensemble added to ::output_values.
 */
void ensemble_update( uint arg0, uint arg1 );

/* Static inline access functions ********************************************/
// -- Encoder(s) and decoder(s)
//! Get the encoder value for the given neuron and dimension
static inline value_t neuron_encoder(uint n, uint d)
{
  return g_ensemble.encoders[n * g_ensemble.encoder_width + d];
}

static inline value_t *neuron_encoder_vector(uint n)
{
  return &g_ensemble.encoders[n * g_ensemble.encoder_width];
}

static inline value_t neuron_decoder(uint n, uint d)
{
  return g_ensemble.decoders[n * g_n_output_dimensions + d];
}

static inline value_t *neuron_decoder_vector(uint n)
{
  return &g_ensemble.decoders[n * g_n_output_dimensions];
}

// -- Voltages and refractory periods
//! Get the membrane voltage for the given neuron
static inline voltage_t neuron_voltage(uint n)
{
  return g_ensemble.neuron_voltage[n];
}

//! Set the membrane voltage for the given neuron
static inline void set_neuron_voltage(uint n, voltage_t v)
{
  g_ensemble.neuron_voltage[n] = v;
}

//! Get the refractory status of a given neuron
static inline uint8_t neuron_refractory(uint n)
{
  return g_ensemble.neuron_refractory[n];
}

//! Put the given neuron in a refractory state (zero voltage, set timer)
static inline void set_neuron_refractory(uint n)
{
  g_ensemble.neuron_refractory[n] = g_ensemble.t_ref;
}

//! Decrement the refractory time for the given neuron
static inline void decrement_neuron_refractory(uint n)
{
  g_ensemble.neuron_refractory[n]--;
}

#endif

/** @} */
