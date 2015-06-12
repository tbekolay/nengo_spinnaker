/**
 * Ensemble - Voja
 * -----------------
 * Functions to perform Voja encoder learning
 * 
 * Authors:
 *   - James Knight <knightj@cs.man.ac.uk>
 * 
 * Copyright:
 *   - Advanced Processor Technologies, School of Computer Science,
 *      University of Manchester
 * 
 * \addtogroup ensemble
 * @{
 */


#ifndef __ENSEMBLE_VOJA_H__
#define __ENSEMBLE_VOJA_H__

#include "ensemble.h"

//----------------------------------
// Functions
//----------------------------------
/**
* \brief Copy in data controlling the Voja learning
* rule from the Voja region of the Ensemble.
*/
bool get_voja(address_t address);

/**
* \brief Apply voja learning to encoders
*/
void voja_step();

/** @} */

#endif  // __ENSEMBLE_VOJA_H__