#include "MxCTest.h"

#include <MxCUniverse.h>
#include <MxCParticle.h>
#include <MxCForce.h>
#include <MxCBind.h>
#include <MxCFlux.h>


int main(int argc, char** argv) {
    float dim[] = {6.5, 6.5, 6.5};

    struct MxSimulator_ConfigHandle config;
    struct MxUniverseConfigHandle uconfig;
    struct MxBoundaryConditionsArgsContainerHandle bargs;
    struct BoundaryConditionSpaceKindHandle bcKindEnum;

    MXCTEST_CHECK(MxCBoundaryConditionSpaceKind_init(&bcKindEnum));
    MXCTEST_CHECK(MxCBoundaryConditionsArgsContainer_init(&bargs));
    MXCTEST_CHECK(MxCBoundaryConditionsArgsContainer_setValueAll(&bargs, bcKindEnum.SPACE_FREESLIP_FULL));

    MXCTEST_CHECK(MxCSimulator_Config_init(&config));
    MXCTEST_CHECK(MxCSimulator_Config_setWindowless(&config, 1));
    MXCTEST_CHECK(MxCSimulator_Config_getUniverseConfig(&config, &uconfig));
    MXCTEST_CHECK(MxCUniverseConfig_setBoundaryConditions(&uconfig, &bargs));
    MXCTEST_CHECK(MxCUniverseConfig_setDim(&uconfig, dim));
    MXCTEST_CHECK(MxCSimulator_initC(&config, NULL, 0));

    struct MxCParticleType ATypeDef = MxCParticleTypeDef_init();
    struct MxCParticleType BTypeDef = MxCParticleTypeDef_init();
    ATypeDef.radius = 0.1;
    ATypeDef.species = (char**)malloc(sizeof(char*));
    ATypeDef.species[0] = "S1";
    ATypeDef.numSpecies = 1;
    BTypeDef.species = (char**)malloc(sizeof(char*));
    BTypeDef.species[0] = "S1";
    BTypeDef.numSpecies = 1;

    struct MxParticleTypeHandle AType, BType;
    MXCTEST_CHECK(MxCParticleType_initD(&AType, ATypeDef));
    MXCTEST_CHECK(MxCParticleType_registerType(&AType));
    MXCTEST_CHECK(MxCParticleType_initD(&BType, BTypeDef));
    MXCTEST_CHECK(MxCParticleType_registerType(&BType));

    struct MxSpeciesListHandle slistB;
    MXCTEST_CHECK(MxCParticleType_getSpecies(&BType, &slistB));
    struct MxSpeciesHandle S1B;
    MXCTEST_CHECK(MxCSpeciesList_getItemS(&slistB, "S1", &S1B));
    MXCTEST_CHECK(MxCSpecies_setConstant(&S1B, 1));

    double dt;
    MXCTEST_CHECK(MxCUniverse_getDt(&dt));

    struct GaussianHandle force_rnd;
    struct MxForceHandle force_rnd_base;
    MXCTEST_CHECK(MxCGaussian_init(&force_rnd, 0.1, 1.0, dt));
    MXCTEST_CHECK(MxCGaussian_toBase(&force_rnd, &force_rnd_base));

    MXCTEST_CHECK(MxCBind_forceS(&force_rnd_base, &AType, "S1"));
    MXCTEST_CHECK(MxCBind_forceS(&force_rnd_base, &BType, "S1"));

    struct MxFluxesHandle fluxhAA, fluxhAB;
    MXCTEST_CHECK(MxCFluxes_fluxFick(&fluxhAA, &AType, &AType, "S1", 1.0, 0.0));
    MXCTEST_CHECK(MxCFluxes_fluxFick(&fluxhAB, &AType, &BType, "S1", 1.0, 0.0));

    unsigned int numParts = 500;
    int pid;
    for(unsigned int i = 0; i < numParts; i++) {
        MXCTEST_CHECK(MxCParticleType_createParticle(&AType, &pid, NULL, NULL));
    }

    struct MxParticleHandleHandle part;
    MXCTEST_CHECK(MxCParticleType_getParticle(&AType, 0, &part));
    MXCTEST_CHECK(MxCParticleHandle_become(&part, &BType));

    struct MxStateVectorHandle svec;
    MXCTEST_CHECK(MxCParticleHandle_getSpecies(&part, &svec));
    MXCTEST_CHECK(MxCStateVector_setItem(&svec, 0, 10.0));

    MXCTEST_CHECK(MxCTest_runQuiet(20));

    return 0;
}