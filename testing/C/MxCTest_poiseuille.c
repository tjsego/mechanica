#include "MxCTest.h"

#include <MxCParticle.h>
#include <MxCBind.h>
#include <MxCBoundaryConditions.h>
#include <MxCForce.h>


void forceFunction(struct MxConstantForceHandle *fh, float *f) {
    f[0] = 0.1;
    f[1] = 0.0;
    f[2] = 0.0;
}


int main(int argc, char** argv) {
    double dt = 0.1;
    float *dim = (float*)malloc(3 * sizeof(float));
    dim[0] = 15.0; dim[1] = 12.0; dim[2] = 10.0;
    int *cells = (int*)malloc(3 * sizeof(int));
    cells[0] = 7; cells[1] = 6; cells[2] = 5;
    double cutoff = 0.5;

    struct MxBoundaryConditionsArgsContainerHandle bargs;
    float *bvel = (float*)malloc(3 * sizeof(float));
    for(unsigned int i = 0; i < 3; i++) bvel[i] = 0.0;
    MXCTEST_CHECK(MxCBoundaryConditionsArgsContainer_init(&bargs));
    MXCTEST_CHECK(MxCBoundaryConditionsArgsContainer_setVelocity(&bargs, "top", bvel));
    MXCTEST_CHECK(MxCBoundaryConditionsArgsContainer_setVelocity(&bargs, "bottom", bvel));

    struct MxSimulator_ConfigHandle config;
    struct MxUniverseConfigHandle uconfig;
    MXCTEST_CHECK(MxCSimulator_Config_init(&config));
    MXCTEST_CHECK(MxCSimulator_Config_setWindowless(&config, 1));
    MXCTEST_CHECK(MxCSimulator_Config_getUniverseConfig(&config, &uconfig));
    MXCTEST_CHECK(MxCUniverseConfig_setDt(&uconfig, dt));
    MXCTEST_CHECK(MxCUniverseConfig_setDim(&uconfig, dim));
    MXCTEST_CHECK(MxCUniverseConfig_setCells(&uconfig, cells));
    MXCTEST_CHECK(MxCUniverseConfig_setCutoff(&uconfig, cutoff));
    MXCTEST_CHECK(MxCUniverseConfig_setBoundaryConditions(&uconfig, &bargs));
    MXCTEST_CHECK(MxCSimulator_initC(&config, NULL, 0));

    struct MxCParticleTypeStyle ATypeStyleDef = MxCParticleTypeStyleDef_init();
    struct MxCParticleType ATypeDef = MxCParticleTypeDef_init();

    ATypeStyleDef.color = "seagreen";
    ATypeDef.radius = 0.05;
    ATypeDef.mass = 10.0;
    ATypeDef.style = &ATypeStyleDef;

    struct MxParticleTypeHandle AType;
    MXCTEST_CHECK(MxCParticleType_initD(&AType, ATypeDef));
    MXCTEST_CHECK(MxCParticleType_registerType(&AType));

    struct MxPotentialHandle dpd;
    double dpd_alpha = 10.0;
    double dpd_sigma = 1.0;
    MXCTEST_CHECK(MxCPotential_create_dpd(&dpd, &dpd_alpha, NULL, &dpd_sigma, NULL, NULL));
    MXCTEST_CHECK(MxCBind_types(&dpd, &AType, &AType, 0));

    struct MxConstantForceHandle pressure;
    struct MxForceHandle pressure_base;
    struct MxUserForceFuncTypeHandle forceFunc;
    MxUserForceFuncTypeHandleFcn _forceFunction = (MxUserForceFuncTypeHandleFcn)forceFunction;
    MXCTEST_CHECK(MxCForce_EvalFcn_init(&forceFunc, &_forceFunction));
    MXCTEST_CHECK(MxCConstantForce_init(&pressure, &forceFunc, 0.0));

    MXCTEST_CHECK(MxCConstantForce_toBase(&pressure, &pressure_base));
    MXCTEST_CHECK(MxCBind_force(&pressure_base, &AType));

    int pid;
    for(unsigned int i = 0; i < 5000; i++) 
        MXCTEST_CHECK(MxCParticleType_createParticle(&AType, &pid, NULL, NULL));

    MXCTEST_CHECK(MxCTest_runQuiet(20));

    return 0;
}