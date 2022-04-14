#include "MxCTest.h"

#include <MxCParticle.h>
#include <MxCPotential.h>
#include <MxCForce.h>
#include <MxCBind.h>
#include <MxCFlux.h>


int main(int argc, char** argv) {
    float dim[] = {15.0, 6.0, 6.0};
    int cells[] = {9, 3, 3};
    double cutoff = 3.0;
    
    struct BoundaryConditionKindHandle bcKindEnum;
    MXCTEST_CHECK(MxCBoundaryConditionKind_init(&bcKindEnum));

    struct MxBoundaryConditionsArgsContainerHandle bargs;
    MXCTEST_CHECK(MxCBoundaryConditionsArgsContainer_init(&bargs));
    MXCTEST_CHECK(MxCBoundaryConditionsArgsContainer_setValue(&bargs, "x", bcKindEnum.BOUNDARY_RESETTING));

    struct MxSimulator_ConfigHandle config;
    struct MxUniverseConfigHandle uconfig;

    MXCTEST_CHECK(MxCSimulator_Config_init(&config));
    MXCTEST_CHECK(MxCSimulator_Config_getUniverseConfig(&config, &uconfig));
    MXCTEST_CHECK(MxCUniverseConfig_setDim(&uconfig, dim));
    MXCTEST_CHECK(MxCUniverseConfig_setDt(&uconfig, 0.1));
    MXCTEST_CHECK(MxCUniverseConfig_setCutoff(&uconfig, 3.0));
    MXCTEST_CHECK(MxCUniverseConfig_setBoundaryConditions(&uconfig, &bargs));
    MXCTEST_CHECK(MxCSimulator_initC(&config, NULL, 0));

    struct MxCParticleTypeStyle ATypeStyleDef = MxCParticleTypeStyleDef_init();
    struct MxCParticleType ATypeDef = MxCParticleTypeDef_init();

    ATypeStyleDef.speciesName = "S1";
    ATypeDef.numSpecies = 3;
    ATypeDef.species = (char**)malloc(3 * sizeof(char*));
    ATypeDef.species[0] = "S1";
    ATypeDef.species[1] = "S2";
    ATypeDef.species[2] = "S3";
    ATypeDef.style = &ATypeStyleDef;

    struct MxParticleTypeHandle AType;
    MXCTEST_CHECK(MxCParticleType_initD(&AType, ATypeDef));
    MXCTEST_CHECK(MxCParticleType_registerType(&AType));

    struct MxFluxesHandle flux;
    MXCTEST_CHECK(MxCFluxes_fluxFick(&flux, &AType, &AType, "S1", 2.0, 0.0));

    float *center = (float*)malloc(3 * sizeof(float));
    MXCTEST_CHECK(MxCUniverse_getCenter(&center));

    float pos[] = {center[0], center[1] - 1.0, center[2]};
    int pid0, pid1;
    MXCTEST_CHECK(MxCParticleType_createParticle(&AType, &pid0, pos, NULL));
    pos[0] = center[0] - 5.0;
    pos[1] = center[1] + 1.0;
    float velocity[] = {0.5, 0.0, 0.0};
    MXCTEST_CHECK(MxCParticleType_createParticle(&AType, &pid1, pos, velocity));

    struct MxParticleHandleHandle part0, part1;
    MXCTEST_CHECK(MxCParticleHandle_init(&part0, pid0));
    MXCTEST_CHECK(MxCParticleHandle_init(&part1, pid1));

    struct MxStateVectorHandle svec0, svec1;
    MXCTEST_CHECK(MxCParticleHandle_getSpecies(&part0, &svec0));
    MXCTEST_CHECK(MxCParticleHandle_getSpecies(&part1, &svec1));

    MXCTEST_CHECK(MxCStateVector_setItem(&svec0, 0, 1.0));
    MXCTEST_CHECK(MxCStateVector_setItem(&svec1, 0, 0.0));

    MXCTEST_CHECK(MxCTest_runQuiet(100));

    return 0;
}