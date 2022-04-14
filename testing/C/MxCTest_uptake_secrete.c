#include "MxCTest.h"

#include <MxCFlux.h>
#include <MxCBoundaryConditions.h>


int main(int argc, char** argv) {
    float dim[] = {6.5, 6.5, 6.5};

    struct BoundaryConditionSpaceKindHandle bcEnums;
    MXCTEST_CHECK(MxCBoundaryConditionSpaceKind_init(&bcEnums));

    struct MxBoundaryConditionsArgsContainerHandle bargs;
    MXCTEST_CHECK(MxCBoundaryConditionsArgsContainer_init(&bargs));
    MXCTEST_CHECK(MxCBoundaryConditionsArgsContainer_setValueAll(&bargs, bcEnums.SPACE_FREESLIP_FULL));

    struct MxSimulator_ConfigHandle config;
    struct MxUniverseConfigHandle uconfig;
    MXCTEST_CHECK(MxCSimulator_Config_init(&config));
    MXCTEST_CHECK(MxCSimulator_Config_getUniverseConfig(&config, &uconfig));
    MXCTEST_CHECK(MxCUniverseConfig_setDim(&uconfig, dim));
    MXCTEST_CHECK(MxCUniverseConfig_setBoundaryConditions(&uconfig, &bargs));
    MXCTEST_CHECK(MxCSimulator_initC(&config, NULL, 0));

    struct MxCParticleTypeStyle ATypeStyleDef = MxCParticleTypeStyleDef_init();
    struct MxCParticleTypeStyle ProducerTypeStyleDef = MxCParticleTypeStyleDef_init();
    struct MxCParticleTypeStyle ConsumerTypeStyleDef = MxCParticleTypeStyleDef_init();

    struct MxCParticleType ATypeDef = MxCParticleTypeDef_init();
    struct MxCParticleType ProducerTypeDef = MxCParticleTypeDef_init();
    struct MxCParticleType ConsumerTypeDef = MxCParticleTypeDef_init();

    ATypeDef.radius = 0.1;
    ATypeDef.numSpecies = 3;
    ATypeDef.species = (char**)malloc(3 * sizeof(char*));
    ATypeDef.species[0] = "S1";
    ATypeDef.species[1] = "S2";
    ATypeDef.species[2] = "S3";
    ATypeStyleDef.speciesName = "S1";
    ATypeDef.style = &ATypeStyleDef;

    ProducerTypeDef.radius = 0.1;
    ProducerTypeDef.numSpecies = 3;
    ProducerTypeDef.species = (char**)malloc(3 * sizeof(char*));
    ProducerTypeDef.species[0] = "S1";
    ProducerTypeDef.species[1] = "S2";
    ProducerTypeDef.species[2] = "S3";
    ProducerTypeStyleDef.speciesName = "S1";
    ProducerTypeDef.style = &ProducerTypeStyleDef;

    ConsumerTypeDef.radius = 0.1;
    ConsumerTypeDef.numSpecies = 3;
    ConsumerTypeDef.species = (char**)malloc(3 * sizeof(char*));
    ConsumerTypeDef.species[0] = "S1";
    ConsumerTypeDef.species[1] = "S2";
    ConsumerTypeDef.species[2] = "S3";
    ConsumerTypeStyleDef.speciesName = "S1";
    ConsumerTypeDef.style = &ConsumerTypeStyleDef;

    struct MxParticleTypeHandle AType, ProducerType, ConsumerType;
    MXCTEST_CHECK(MxCParticleType_initD(&AType, ATypeDef));
    MXCTEST_CHECK(MxCParticleType_initD(&ProducerType, ProducerTypeDef));
    MXCTEST_CHECK(MxCParticleType_initD(&ConsumerType, ConsumerTypeDef));
    MXCTEST_CHECK(MxCParticleType_registerType(&AType));
    MXCTEST_CHECK(MxCParticleType_registerType(&ProducerType));
    MXCTEST_CHECK(MxCParticleType_registerType(&ConsumerType));

    struct MxFluxesHandle fluxAA, fluxPA, fluxAC;
    MXCTEST_CHECK(MxCFluxes_fluxFick(&fluxAA, &AType, &AType, "S1", 1.0, 0.0));
    MXCTEST_CHECK(MxCFluxes_fluxFick(&fluxPA, &ProducerType, &AType, "S1", 1.0, 0.0));
    MXCTEST_CHECK(MxCFluxes_fluxFick(&fluxAC, &AType, &ConsumerType, "S1", 2.0, 10.0));

    float *posP = (float*)malloc(3 * sizeof(float));
    float *posC = (float*)malloc(3 * sizeof(float));
    float offset = 1.0;
    float *center = (float*)malloc(3 * sizeof(float));
    MXCTEST_CHECK(MxCUniverse_getCenter(&center));
    posP[0] = offset; posP[1] = center[1]; posP[2] = center[2];
    posC[0] = dim[0] - offset; posC[1] = center[1]; posC[2] = center[2];

    int pid;
    MXCTEST_CHECK(MxCParticleType_createParticle(&ProducerType, &pid, posP, NULL));
    MXCTEST_CHECK(MxCParticleType_createParticle(&ConsumerType, &pid, posC, NULL));

    struct MxParticleHandleHandle partP;
    MXCTEST_CHECK(MxCParticleType_getParticle(&ProducerType, 0, &partP));
    struct MxStateVectorHandle svec;
    MXCTEST_CHECK(MxCParticleHandle_getSpecies(&partP, &svec));
    MXCTEST_CHECK(MxCStateVector_setItem(&svec, 0, 200.0));

    for(unsigned int i = 0; i < 1000; i++) 
        MXCTEST_CHECK(MxCParticleType_createParticle(&AType, &pid, NULL, NULL));

    MXCTEST_CHECK(MxCTest_runQuiet(100));

    return 0;
}