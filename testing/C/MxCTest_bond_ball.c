#include "MxCTest.h"

#include <MxCParticle.h>
#include <MxCForce.h>
#include <MxCPotential.h>
#include <MxCBind.h>
#include <MxCStyle.h>
#include <MxCUtil.h>


int main(int argc, char** argv) {
    //  dimensions of universe
    float dim[] = {30.0, 30.0, 30.0};

    double cutoff = 5.0;
    double dt = 0.0005;

    // new simulator
    struct MxSimulator_ConfigHandle config;
    struct MxUniverseConfigHandle uconfig;
    MXCTEST_CHECK(MxCSimulator_Config_init(&config));
    MXCTEST_CHECK(MxCSimulator_Config_setWindowless(&config, 1));
    MXCTEST_CHECK(MxCSimulator_Config_getUniverseConfig(&config, &uconfig));
    MXCTEST_CHECK(MxCUniverseConfig_setDim(&uconfig, dim));
    MXCTEST_CHECK(MxCUniverseConfig_setCutoff(&uconfig, cutoff));
    MXCTEST_CHECK(MxCUniverseConfig_setDt(&uconfig, dt));
    MXCTEST_CHECK(MxCSimulator_initC(&config, NULL, 0));

    struct MxStyleHandle AStyle, BStyle, CStyle;
    MXCTEST_CHECK(MxCStyle_init(&AStyle));
    MXCTEST_CHECK(MxCStyle_init(&BStyle));
    MXCTEST_CHECK(MxCStyle_init(&CStyle));
    MXCTEST_CHECK(MxCStyle_setColor(&AStyle, "MediumSeaGreen"));
    MXCTEST_CHECK(MxCStyle_setColor(&BStyle, "skyblue"));
    MXCTEST_CHECK(MxCStyle_setColor(&CStyle, "orange"));

    float Aradius = 0.5;
    float Bradius = 0.2;
    float Cradius = 10.0;

    struct MxParticleTypeHandle AType, BType, CType;
    struct MxParticleDynamicsEnumHandle dynEnums;
    MXCTEST_CHECK(MxCParticleType_init(&AType));
    MXCTEST_CHECK(MxCParticleType_init(&BType));
    MXCTEST_CHECK(MxCParticleType_init(&CType));
    MXCTEST_CHECK(MxCParticleDynamics_init(&dynEnums));

    MXCTEST_CHECK(MxCParticleType_setName(&AType, "AType"));
    MXCTEST_CHECK(MxCParticleType_setRadius(&AType, Aradius));
    MXCTEST_CHECK(MxCParticleType_setDynamics(&AType, dynEnums.PARTICLE_OVERDAMPED));
    MXCTEST_CHECK(MxCParticleType_setMass(&AType, 5.0));
    MXCTEST_CHECK(MxCParticleType_setStyle(&AType, &AStyle));

    MXCTEST_CHECK(MxCParticleType_setName(&BType, "BType"));
    MXCTEST_CHECK(MxCParticleType_setRadius(&BType, Bradius));
    MXCTEST_CHECK(MxCParticleType_setDynamics(&BType, dynEnums.PARTICLE_OVERDAMPED));
    MXCTEST_CHECK(MxCParticleType_setMass(&BType, 1.0));
    MXCTEST_CHECK(MxCParticleType_setStyle(&BType, &BStyle));

    MXCTEST_CHECK(MxCParticleType_setName(&CType, "CType"));
    MXCTEST_CHECK(MxCParticleType_setRadius(&CType, Cradius));
    MXCTEST_CHECK(MxCParticleType_setFrozen(&CType, 1));
    MXCTEST_CHECK(MxCParticleType_setStyle(&CType, &CStyle));

    MXCTEST_CHECK(MxCParticleType_registerType(&AType));
    MXCTEST_CHECK(MxCParticleType_registerType(&BType));
    MXCTEST_CHECK(MxCParticleType_registerType(&CType));

    float *center = (float*)malloc(3 * sizeof(float));
    MXCTEST_CHECK(MxCUniverse_getCenter(&center));

    int pid;
    MXCTEST_CHECK(MxCParticleType_createParticle(&CType, &pid, center, NULL));

    struct MxPointsTypeHandle pointTypesEnum;
    MXCTEST_CHECK(MxCPointsType_init(&pointTypesEnum));

    float *ringPoints;
    int numRingPoints = 100;
    MXCTEST_CHECK(MxCPoints(pointTypesEnum.Ring, numRingPoints, &ringPoints));

    float pt[3];
    float offset[] = {0.0, 0.0, -1.0};
    
    for(unsigned int i = 0; i < numRingPoints; i++) {
        for(unsigned int j = 0; j < 3; j++) {
            pt[j] = ringPoints[3 * i + j] * (Bradius + Cradius) + center[j] + offset[j];
        }
        MXCTEST_CHECK(MxCParticleType_createParticle(&BType, &pid, pt, NULL));
    }

    struct MxPotentialHandle pc, pa, pb, pab, ph;
    double pc_m = 2.0;
    double pc_max = 5.0;
    MXCTEST_CHECK(MxCPotential_create_glj(&pc, 30.0, &pc_m, NULL, NULL, NULL, NULL, &pc_max, NULL, NULL));
    double pa_m = 2.5;
    double pa_max = 3.0;
    MXCTEST_CHECK(MxCPotential_create_glj(&pa, 3.0, &pa_m, NULL, NULL, NULL, NULL, &pa_max, NULL, NULL));
    double pb_m = 4.0;
    double pb_max = 1.0;
    MXCTEST_CHECK(MxCPotential_create_glj(&pb, 1.0, &pb_m, NULL, NULL, NULL, NULL, &pb_max, NULL, NULL));
    double pab_m = 2.0;
    double pab_max = 1.0;
    MXCTEST_CHECK(MxCPotential_create_glj(&pab, 1.0, &pab_m, NULL, NULL, NULL, NULL, &pab_max, NULL, NULL));
    double ph_r0 = 0.001;
    double ph_k = 200.0;
    MXCTEST_CHECK(MxCPotential_create_harmonic(&ph, 200.0, 0.001, NULL, NULL, NULL));

    MXCTEST_CHECK(MxCBind_types(&pc, &AType, &CType, 0));
    MXCTEST_CHECK(MxCBind_types(&pc, &BType, &CType, 0));
    MXCTEST_CHECK(MxCBind_types(&pa, &AType, &AType, 0));
    MXCTEST_CHECK(MxCBind_types(&pab, &AType, &BType, 0));

    struct MxForceHandle forceBase;
    struct GaussianHandle force;
    MXCTEST_CHECK(MxCGaussian_init(&force, 5.0, 0.0, dt));
    MXCTEST_CHECK(MxCGaussian_toBase(&force, &forceBase));
    MXCTEST_CHECK(MxCBind_force(&forceBase, &AType));
    MXCTEST_CHECK(MxCBind_force(&forceBase, &BType));

    struct MxParticleListHandle plist;
    MXCTEST_CHECK(MxCParticleList_init(&plist));
    int numBParts;
    unsigned int pindex;
    struct MxParticleHandleHandle part;
    MXCTEST_CHECK(MxCParticleType_getNumParticles(&BType, &numBParts));
    for(unsigned int i = 0; i < numBParts; i++) {
        MXCTEST_CHECK(MxCParticleType_getParticle(&BType, i, &part));
        MXCTEST_CHECK(MxCParticleList_insertP(&plist, &part, &pindex));
    }
    MXCTEST_CHECK(MxCBind_bonds(&ph, &plist, 1.0, NULL, NULL, 0, NULL, NULL, NULL, NULL));

    MXCTEST_CHECK(MxCTest_runQuiet(100));

    return 0;
}