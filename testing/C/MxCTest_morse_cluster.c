#include "MxCTest.h"

#include <MxCParticle.h>
#include <MxCCluster.h>
#include <MxCBind.h>


int main(int argc, char** argv) {
    float dim[] = {30.0, 30.0, 30.0};
    double cutoff = 10.0;
    double dt = 0.0005;

    struct MxSimulator_ConfigHandle config;
    struct MxUniverseConfigHandle uconfig;

    MXCTEST_CHECK(MxCSimulator_Config_init(&config));
    MXCTEST_CHECK(MxCSimulator_Config_getUniverseConfig(&config, &uconfig));
    MXCTEST_CHECK(MxCUniverseConfig_setDim(&uconfig, dim));
    MXCTEST_CHECK(MxCUniverseConfig_setCutoff(&uconfig, cutoff));
    MXCTEST_CHECK(MxCUniverseConfig_setDt(&uconfig, dt));
    MXCTEST_CHECK(MxCSimulator_initC(&config, NULL, 0));

    struct MxCParticleType ATypeDef = MxCParticleTypeDef_init();
    struct MxCParticleType BTypeDef = MxCParticleTypeDef_init();
    struct MxCParticleTypeStyle ATypeStyleDef = MxCParticleTypeStyleDef_init();
    struct MxCParticleTypeStyle BTypeStyleDef = MxCParticleTypeStyleDef_init();

    ATypeDef.radius = 0.5;
    ATypeDef.dynamics = 1;
    ATypeDef.mass = 10.0;
    ATypeStyleDef.color = "MediumSeaGreen";
    ATypeDef.style = &ATypeStyleDef;

    BTypeDef.radius = 0.5;
    BTypeDef.dynamics = 1;
    BTypeDef.mass = 10.0;
    BTypeStyleDef.color = "skyblue";
    BTypeDef.style = &BTypeStyleDef;

    struct MxParticleTypeHandle AType, BType;
    MXCTEST_CHECK(MxCParticleType_initD(&AType, ATypeDef));
    MXCTEST_CHECK(MxCParticleType_initD(&BType, BTypeDef));
    MXCTEST_CHECK(MxCParticleType_registerType(&AType));
    MXCTEST_CHECK(MxCParticleType_registerType(&BType));

    struct MxCClusterType CTypeDef = MxCClusterTypeDef_init();
    CTypeDef.numTypes = 2;
    CTypeDef.types = (struct MxParticleTypeHandle**)malloc(CTypeDef.numTypes * sizeof(struct MxParticleTypeHandle*));
    CTypeDef.types[0] = &AType;
    CTypeDef.types[1] = &BType;

    struct MxClusterParticleTypeHandle CType;
    MXCTEST_CHECK(MxCClusterParticleType_initD(&CType, CTypeDef));
    MXCTEST_CHECK(MxCClusterParticleType_registerType(&CType));

    struct MxPotentialHandle p1, p2;
    double p1_d = 0.5;
    double p1_a = 5.0;
    double p1_max = 3.0;
    double p2_d = 0.5;
    double p2_a = 2.5;
    double p2_max = 3.0;
    MXCTEST_CHECK(MxCPotential_create_morse(&p1, &p1_d, &p1_a, NULL, NULL, &p1_max, NULL));
    MXCTEST_CHECK(MxCPotential_create_morse(&p2, &p2_d, &p2_a, NULL, NULL, &p2_max, NULL));
    MXCTEST_CHECK(MxCBind_types(&p1, &AType, &AType, 1));
    MXCTEST_CHECK(MxCBind_types(&p2, &BType, &BType, 1));

    struct GaussianHandle force;
    struct MxForceHandle force_base;
    MXCTEST_CHECK(MxCGaussian_init(&force, 10.0, 0.0, dt));
    MXCTEST_CHECK(MxCGaussian_toBase(&force, &force_base));
    MXCTEST_CHECK(MxCBind_force(&force_base, &AType));
    MXCTEST_CHECK(MxCBind_force(&force_base, &BType));

    float *center = (float*)malloc(3 * sizeof(float));
    MXCTEST_CHECK(MxCUniverse_getCenter(&center));

    struct MxClusterParticleHandleHandle c1, c2;
    float pos[] = {center[0] - 3.0, center[1], center[2]};
    int cid1, cid2;
    MXCTEST_CHECK(MxCClusterParticleType_createParticle(&CType, &cid1, pos, NULL));
    MXCTEST_CHECK(MxCClusterParticleHandle_init(&c1, cid1));

    pos[0] = center[0] + 7.0;
    MXCTEST_CHECK(MxCClusterParticleType_createParticle(&CType, &cid2, pos, NULL));
    MXCTEST_CHECK(MxCClusterParticleHandle_init(&c2, cid2));

    int pid;
    for(unsigned int i = 0; i < 2000; i++) {
        MXCTEST_CHECK(MxCClusterParticleHandle_createParticle(&c1, &AType, &pid, NULL, NULL));
        MXCTEST_CHECK(MxCClusterParticleHandle_createParticle(&c2, &BType, &pid, NULL, NULL));
    }

    MXCTEST_CHECK(MxCTest_runQuiet(20));

    return 0;
}