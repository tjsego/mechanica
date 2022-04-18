#include "MxCTest.h"

#include <MxCUniverse.h>
#include <MxCParticle.h>
#include <MxCPotential.h>
#include <MxCForce.h>
#include <MxCUtil.h>
#include <MxCBind.h>


int main(int argc, char** argv) {

    int A_count = 5000;
    int B_count = 5000;
    float dim[] = {20.0, 20.0, 20.0};
    float cutoff = 3.0;

    struct MxSimulator_ConfigHandle config;
    MXCTEST_CHECK(MxCSimulator_Config_init(&config));
    MXCTEST_CHECK(MxCSimulator_Config_setWindowless(&config, 1));
    struct MxUniverseConfigHandle uconfig;
    MXCTEST_CHECK(MxCSimulator_Config_getUniverseConfig(&config, &uconfig));
    MXCTEST_CHECK(MxCUniverseConfig_setDim(&uconfig, dim));
    MXCTEST_CHECK(MxCUniverseConfig_setCutoff(&uconfig, cutoff));
    MXCTEST_CHECK(MxCSimulator_initC(&config, NULL, 0));

    double dt;
    MXCTEST_CHECK(MxCUniverse_getDt(&dt));

    struct MxParticleDynamicsEnumHandle dynEnums;
    MXCTEST_CHECK(MxCParticleDynamics_init(&dynEnums));

    struct MxStyleHandle AStyle, BStyle;
    MXCTEST_CHECK(MxCStyle_init(&AStyle));
    MXCTEST_CHECK(MxCStyle_setColor(&AStyle, "red"));
    
    MXCTEST_CHECK(MxCStyle_init(&BStyle));
    MXCTEST_CHECK(MxCStyle_setColor(&BStyle, "blue"));

    struct MxParticleTypeHandle AType, BType;

    MXCTEST_CHECK(MxCParticleType_init(&AType));
    MXCTEST_CHECK(MxCParticleType_setMass(&AType, 40.0));
    MXCTEST_CHECK(MxCParticleType_setRadius(&AType, 0.4));
    MXCTEST_CHECK(MxCParticleType_setDynamics(&AType, dynEnums.PARTICLE_OVERDAMPED));
    MXCTEST_CHECK(MxCParticleType_setStyle(&AType, &AStyle));

    MXCTEST_CHECK(MxCParticleType_init(&BType));
    MXCTEST_CHECK(MxCParticleType_setMass(&BType, 40.0));
    MXCTEST_CHECK(MxCParticleType_setRadius(&BType, 0.4));
    MXCTEST_CHECK(MxCParticleType_setDynamics(&BType, dynEnums.PARTICLE_OVERDAMPED));
    MXCTEST_CHECK(MxCParticleType_setStyle(&BType, &BStyle));

    MXCTEST_CHECK(MxCParticleType_registerType(&AType));
    MXCTEST_CHECK(MxCParticleType_registerType(&BType));

    struct MxPotentialHandle pot_aa, pot_bb, pot_ab;

    double aa_d = 3.0, aa_a = 5.0, aa_max = 3.0;
    double bb_d = 3.0, bb_a = 5.0, bb_max = 3.0;
    double ab_d = 0.3, ab_a = 5.0, ab_max = 3.0;

    MXCTEST_CHECK(MxCPotential_create_morse(&pot_aa, &aa_d, &aa_a, NULL, NULL, &aa_max, NULL));
    MXCTEST_CHECK(MxCPotential_create_morse(&pot_bb, &bb_d, &bb_a, NULL, NULL, &bb_max, NULL));
    MXCTEST_CHECK(MxCPotential_create_morse(&pot_ab, &ab_d, &ab_a, NULL, NULL, &ab_max, NULL));

    MXCTEST_CHECK(MxCBind_types(&pot_aa, &AType, &AType, 0));
    MXCTEST_CHECK(MxCBind_types(&pot_bb, &BType, &BType, 0));
    MXCTEST_CHECK(MxCBind_types(&pot_ab, &AType, &BType, 0));

    struct GaussianHandle force_rnd;
    struct MxForceHandle force_rnd_base;
    MXCTEST_CHECK(MxCGaussian_init(&force_rnd, 50.0, 0.0, dt));
    MXCTEST_CHECK(MxCGaussian_toBase(&force_rnd, &force_rnd_base));
    MXCTEST_CHECK(MxCBind_force(&force_rnd_base, &AType));
    MXCTEST_CHECK(MxCBind_force(&force_rnd_base, &BType));

    struct MxPointsTypeHandle ptTypeEnums;
    MXCTEST_CHECK(MxCPointsType_init(&ptTypeEnums));
    float *ptsA, *ptsB;
    MXCTEST_CHECK(MxCRandomPoints(ptTypeEnums.SolidCube, A_count, 0.0, 0.0, 0.0, &ptsA));
    MXCTEST_CHECK(MxCRandomPoints(ptTypeEnums.SolidCube, B_count, 0.0, 0.0, 0.0, &ptsB));

    float *center = (float*)malloc(3 * sizeof(float));
    MXCTEST_CHECK(MxCUniverse_getCenter(&center));

    int pid;
    float ptp[3];
    for(unsigned int i = 0; i < A_count; i++) {
        for(unsigned int j = 0; j < 3; j++) 
            ptp[j] = center[j] + ptsA[3 * i + j] * 14.5;
        MXCTEST_CHECK(MxCParticleType_createParticle(&AType, &pid, ptp, NULL));
    }
    for(unsigned int i = 0; i < B_count; i++) {
        for(unsigned int j = 0; j < 3; j++) 
            ptp[j] = center[j] + ptsB[3 * i + j] * 14.5;
        MXCTEST_CHECK(MxCParticleType_createParticle(&BType, &pid, ptp, NULL));
    }

    MXCTEST_CHECK(MxCTest_runQuiet(10));

    return 0;
}