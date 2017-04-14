#ifndef OPENMM_H_
#define OPENMM_H_

/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2009-2016 Stanford University and the Authors.      *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "openmm/AndersenThermostat.h"
#include "openmm/BrownianIntegrator.h"
#include "openmm/CMAPTorsionForce.h"
#include "openmm/CMMotionRemover.h"
#include "openmm/CompoundIntegrator.h"
#include "openmm/CustomBondForce.h"
#include "openmm/CustomCentroidBondForce.h"
#include "openmm/CustomCompoundBondForce.h"
#include "openmm/CustomAngleForce.h"
#include "openmm/CustomTorsionForce.h"
#include "openmm/CustomExternalForce.h"
#include "openmm/CustomGBForce.h"
#include "openmm/CustomHbondForce.h"
#include "openmm/CustomIntegrator.h"
#include "openmm/CustomManyParticleForce.h"
#include "openmm/CustomNonbondedForce.h"
#include "openmm/Force.h"
#include "openmm/GayBerneForce.h"
#include "openmm/GBSAOBCForce.h"
#include "openmm/HarmonicAngleForce.h"
#include "openmm/HarmonicBondForce.h"
#include "openmm/Integrator.h"
#include "openmm/LangevinIntegrator.h"
#include "openmm/LocalEnergyMinimizer.h"
#include "openmm/MonteCarloAnisotropicBarostat.h"
#include "openmm/MonteCarloBarostat.h"
#include "openmm/MonteCarloMembraneBarostat.h"
#include "openmm/NonbondedForce.h"
#include "openmm/Context.h"
#include "openmm/OpenMMException.h"
#include "openmm/PeriodicTorsionForce.h"
#include "openmm/RBTorsionForce.h"
#include "openmm/State.h"
#include "openmm/System.h"
#include "openmm/TabulatedFunction.h"
#include "openmm/Units.h"
#include "openmm/VariableLangevinIntegrator.h"
#include "openmm/VariableVerletIntegrator.h"
#include "openmm/Vec3.h"
#include "openmm/VerletIntegrator.h"
#include "openmm/VirtualSite.h"
#include "openmm/Platform.h"
#include "openmm/serialization/XmlSerializer.h"

#endif /*OPENMM_H_*/