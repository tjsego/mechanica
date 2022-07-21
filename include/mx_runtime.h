/*
 * ca_runtime.h
 *
 *  Created on: Jun 30, 2015
 *      Author: andy
 */

#ifndef _INCLUDE_CA_RUNTIME_H_
#define _INCLUDE_CA_RUNTIME_H_

#include <stdio.h>

#ifndef MDCORE_SINGLE
#define MDCORE_SINGLE
#endif

#include <MxSimulator.h>
#include <MxBind.h>
#include <MxUtil.h>
#include <MxCluster.hpp>
#include <Flux.hpp>
#include <event/MxParticleEventSingle.h>
#include <event/MxParticleTimeEvent.h>
#include <rendering/MxClipPlane.hpp>
#include <rendering/MxKeyEvent.hpp>
#include <rendering/MxColorMapper.hpp>
#include <rendering/MxStyle.hpp>
#include <state/MxSpeciesValue.h>

/**
 * Initialize the entire runtime.
 */
CAPI_FUNC(HRESULT) Mx_Initialize(int);


#endif /* _INCLUDE_CA_RUNTIME_H_ */
