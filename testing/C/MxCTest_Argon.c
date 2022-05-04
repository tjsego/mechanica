#include "MxCTest.h"

#include <MxCPotential.h>
#include <MxCBind.h>
#include <MxCParticle.h>


int main(int argc, char** argv) {
    //  dimensions of universe
    float dim[] = {10.0, 10.0, 10.0};

    int cells[] = {5, 5, 5};
    double cutoff = 1.0;

    // new simulator
    struct MxSimulator_ConfigHandle config;
    struct MxUniverseConfigHandle uconfig;
    MXCTEST_CHECK(MxCSimulator_Config_init(&config));
    MXCTEST_CHECK(MxCSimulator_Config_getUniverseConfig(&config, &uconfig));
    MXCTEST_CHECK(MxCSimulator_Config_setWindowless(&config, 1));
    MXCTEST_CHECK(MxCUniverseConfig_setDim(&uconfig, dim));
    MXCTEST_CHECK(MxCUniverseConfig_setCells(&uconfig, cells));
    MXCTEST_CHECK(MxCUniverseConfig_setCutoff(&uconfig, cutoff));
    MXCTEST_CHECK(MxCSimulator_initC(&config, NULL, 0));

    // create a potential representing a 12-6 Lennard-Jones potential
    struct MxPotentialHandle pot;
    double pottol = 1.0e-3;
    MXCTEST_CHECK(MxCPotential_create_lennard_jones_12_6(&pot, 0.275, 1.0, 9.5075e-06, 6.1545e-03, &pottol));

    // create a particle type
    struct MxParticleTypeHandle Argon;
    MXCTEST_CHECK(MxCParticleType_init(&Argon));
    MXCTEST_CHECK(MxCParticleType_setName(&Argon, "ArgonType"));
    MXCTEST_CHECK(MxCParticleType_setRadius(&Argon, 0.1));
    MXCTEST_CHECK(MxCParticleType_setMass(&Argon, 39.4));
    MXCTEST_CHECK(MxCParticleType_registerType(&Argon));

    // bind the potential with the *TYPES* of the particles
    MXCTEST_CHECK(MxCBind_types(&pot, &Argon, &Argon, 0));

    // uniform cube
    MXCTEST_CHECK(MxCParticleType_factory(&Argon, NULL, 2500, NULL, NULL));

    // run the simulator
    MXCTEST_CHECK(MxCTest_runQuiet(100));

    return 0;
}