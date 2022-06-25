#include "MxCTest.h"

#include <MxCParticle.h>
#include <MxCPotential.h>
#include <MxCBind.h>
#include <MxCForce.h>
#include <MxCBond.h>


int main(int argc, char** argv) {
    double cutoff = 8.0;
    float dim[] = {20.0, 20.0, 20.0};

    struct MxSimulator_ConfigHandle config;
    struct MxUniverseConfigHandle uconfig;

    MXCTEST_CHECK(MxCSimulator_Config_init(&config));
    MXCTEST_CHECK(MxCSimulator_Config_setWindowless(&config, 1));
    MXCTEST_CHECK(MxCSimulator_Config_getUniverseConfig(&config, &uconfig));
    MXCTEST_CHECK(MxCUniverseConfig_setDim(&uconfig, dim));
    MXCTEST_CHECK(MxCUniverseConfig_setCutoff(&uconfig, cutoff));

    MXCTEST_CHECK(MxCSimulator_initC(&config, NULL, 0));

    double dt;
    MXCTEST_CHECK(MxCUniverse_getDt(&dt));

    struct MxParticleDynamicsEnumHandle dynEnums;
    MXCTEST_CHECK(MxCParticleDynamics_init(&dynEnums));

    struct MxParticleTypeHandle BeadType;
    MXCTEST_CHECK(MxCParticleType_init(&BeadType));
    MXCTEST_CHECK(MxCParticleType_setMass(&BeadType, 0.4));
    MXCTEST_CHECK(MxCParticleType_setRadius(&BeadType, 0.2));
    MXCTEST_CHECK(MxCParticleType_setDynamics(&BeadType, dynEnums.PARTICLE_OVERDAMPED));
    MXCTEST_CHECK(MxCParticleType_registerType(&BeadType));

    struct MxPotentialHandle pot_bb, pot_bond, pot_ang;

    double bb_min = 0.1, bb_max = 1.0;
    MXCTEST_CHECK(MxCPotential_create_coulomb(&pot_bb, 0.1, &bb_min, &bb_max, NULL, NULL));
    MXCTEST_CHECK(MxCBind_types(&pot_bb, &BeadType, &BeadType, 0));

    double bond_min=0.0, bond_max = 2.0;
    MXCTEST_CHECK(MxCPotential_create_harmonic(&pot_bond, 0.4, 0.2, &bond_min, &bond_max, NULL));

    double ang_tol = 0.01;
    MXCTEST_CHECK(MxCPotential_create_harmonic_angle(&pot_ang, 0.2, 0.85 * M_PI, NULL, NULL, &ang_tol));

    struct GaussianHandle force_rnd;
    struct MxForceHandle force_rnd_base;
    MXCTEST_CHECK(MxCGaussian_init(&force_rnd, 0.1, 0.0, dt));
    MXCTEST_CHECK(MxCGaussian_toBase(&force_rnd, &force_rnd_base));
    MXCTEST_CHECK(MxCBind_force(&force_rnd_base, &BeadType));

    unsigned int numBeads = 80;
    float xx[numBeads];
    xx[0] = 4.0;
    for(unsigned int i = 1; i < numBeads; i++) {
        xx[i] = xx[i - 1] + 0.15;
    }

    float pos0[] = {xx[0], 10.0, 10.0};
    struct MxParticleHandleHandle bead, p, n;
    int beadid, pid, nid;
    struct MxAngleHandleHandle angle;
    MXCTEST_CHECK(MxCParticleType_createParticle(&BeadType, &beadid, pos0, NULL));
    MXCTEST_CHECK(MxCParticleHandle_init(&bead, beadid));
    for(unsigned int i = 1; i < numBeads; i++) {
        pos0[0] = xx[i];
        MXCTEST_CHECK(MxCParticleType_createParticle(&BeadType, &nid, pos0, NULL));
        MXCTEST_CHECK(MxCParticleHandle_init(&n, nid));
        if(i > 1) {
            MXCTEST_CHECK(MxCAngleHandle_create(&angle, &pot_ang, &p, &bead, &n));
        }
        p = bead;
        bead = n;
    }

    MXCTEST_CHECK(MxCTest_runQuiet(100));

    return 0;
}