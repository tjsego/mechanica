#include "MxCTest.h"

#include <MxCParticle.h>
#include <MxCBoundaryConditions.h>
#include <MxCPotential.h>
#include <MxCBind.h>


int main(int argc, char** argv) {
    float dim[] = {30.0, 30.0, 30.0};
    int cells[] = {3, 3, 3};
    double dist = 3.9;
    double offset = 6.0;
    double dt = 0.01;

    struct MxSimulator_ConfigHandle config;
    struct MxUniverseConfigHandle uconfig;

    MXCTEST_CHECK(MxCSimulator_Config_init(&config));
    MXCTEST_CHECK(MxCSimulator_Config_getUniverseConfig(&config, &uconfig));

    MXCTEST_CHECK(MxCUniverseConfig_setDim(&uconfig, dim));
    MXCTEST_CHECK(MxCUniverseConfig_setCutoff(&uconfig, 7.0));
    MXCTEST_CHECK(MxCUniverseConfig_setCells(&uconfig, cells));
    MXCTEST_CHECK(MxCUniverseConfig_setDt(&uconfig, dt));

    MXCTEST_CHECK(MxCSimulator_initC(&config, NULL, 0));

    struct MxCParticleType ATypeDef = MxCParticleTypeDef_init();
    struct MxCParticleType SphereTypeDef = MxCParticleTypeDef_init();
    struct MxCParticleType TestTypeDef = MxCParticleTypeDef_init();
    struct MxCParticleTypeStyle AStyleDef = MxCParticleTypeStyleDef_init();
    struct MxCParticleTypeStyle SphereStyleDef = MxCParticleTypeStyleDef_init();
    struct MxCParticleTypeStyle TestStyleDef = MxCParticleTypeStyleDef_init();

    AStyleDef.color = "MediumSeaGreen";
    ATypeDef.mass = 2.5;
    ATypeDef.style = &AStyleDef;

    SphereStyleDef.color = "orange";
    SphereTypeDef.radius = 3.0;
    SphereTypeDef.frozen = 1;
    SphereTypeDef.style = &SphereStyleDef;

    TestStyleDef.color = "orange";
    TestTypeDef.radius = 0.0;
    TestTypeDef.frozen = 1;
    TestTypeDef.style = &TestStyleDef;

    struct MxParticleTypeHandle AType, SphereType, TestType;
    MXCTEST_CHECK(MxCParticleType_initD(&AType, ATypeDef));
    MXCTEST_CHECK(MxCParticleType_initD(&SphereType, SphereTypeDef));
    MXCTEST_CHECK(MxCParticleType_initD(&TestType, TestTypeDef));

    MXCTEST_CHECK(MxCParticleType_registerType(&AType));
    MXCTEST_CHECK(MxCParticleType_registerType(&SphereType));
    MXCTEST_CHECK(MxCParticleType_registerType(&TestType));

    struct MxPotentialHandle p;
    double p_m = 2.0;
    double p_max = 5.0;
    MXCTEST_CHECK(MxCPotential_create_glj(&p, 50.0, &p_m, NULL, NULL, NULL, NULL, &p_max, NULL, NULL));

    MXCTEST_CHECK(MxCBind_types(&p, &AType, &SphereType, 0));
    MXCTEST_CHECK(MxCBind_types(&p, &AType, &TestType, 0));

    struct MxBoundaryConditionsHandle bcs;
    MXCTEST_CHECK(MxCUniverse_getBoundaryConditions(&bcs));
    MXCTEST_CHECK(MxCBoundaryConditions_setPotential(&bcs, &AType, &p));

    float *center = (float*)malloc(3 * sizeof(float));
    MXCTEST_CHECK(MxCUniverse_getCenter(&center));
    float pos[] = {5.0 + center[0], center[1], center[2]};
    int pid;
    MXCTEST_CHECK(MxCParticleType_createParticle(&SphereType, &pid, pos, NULL));
    
    pos[2] += SphereTypeDef.radius + dist;
    MXCTEST_CHECK(MxCParticleType_createParticle(&AType, &pid, pos, NULL));

    pos[2] = dist;
    MXCTEST_CHECK(MxCParticleType_createParticle(&AType, &pid, pos, NULL));

    pos[2] = dim[2] - dist;
    MXCTEST_CHECK(MxCParticleType_createParticle(&AType, &pid, pos, NULL));

    pos[0] = dist;
    pos[1] = center[1] - offset;
    pos[2] = center[2];
    MXCTEST_CHECK(MxCParticleType_createParticle(&AType, &pid, pos, NULL));

    pos[0] = dim[0] - dist;
    pos[1] = center[1] + offset;
    pos[2] = center[2];
    MXCTEST_CHECK(MxCParticleType_createParticle(&AType, &pid, pos, NULL));

    pos[0] = center[0];
    pos[1] = dist;
    pos[2] = center[2];
    MXCTEST_CHECK(MxCParticleType_createParticle(&AType, &pid, pos, NULL));

    pos[1] = dim[1] - dist;
    MXCTEST_CHECK(MxCParticleType_createParticle(&AType, &pid, pos, NULL));

    MXCTEST_CHECK(MxCTest_runQuiet(100));

    return 0;
}