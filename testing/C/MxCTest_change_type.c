#include "MxCTest.h"

#include <MxCParticle.h>
#include <MxCPotential.h>
#include <MxCBind.h>
#include <MxCUtil.h>


int main(int argc, char** argv) {
    struct MxSimulator_ConfigHandle config;
    struct MxUniverseConfigHandle uconfig;

    MXCTEST_CHECK(MxCSimulator_Config_init(&config));
    MXCTEST_CHECK(MxCSimulator_Config_getUniverseConfig(&config, &uconfig));
    MXCTEST_CHECK(MxCUniverseConfig_setCutoff(&uconfig, 3.0));
    MXCTEST_CHECK(MxCSimulator_initC(&config, NULL, 0));

    struct MxCParticleTypeStyle AStyleDef = MxCParticleTypeStyleDef_init();
    struct MxCParticleTypeStyle BStyleDef = MxCParticleTypeStyleDef_init();
    struct MxCParticleType ATypeDef = MxCParticleTypeDef_init();
    struct MxCParticleType BTypeDef = MxCParticleTypeDef_init();

    AStyleDef.color = "MediumSeaGreen";
    ATypeDef.radius = 0.1;
    ATypeDef.dynamics = 1;
    ATypeDef.style = &AStyleDef;

    BStyleDef.color = "skyblue";
    BTypeDef.radius = 0.1;
    BTypeDef.dynamics = 1;
    BTypeDef.style = &BStyleDef;

    struct MxParticleTypeHandle AType, BType;
    MXCTEST_CHECK(MxCParticleType_initD(&AType, ATypeDef));
    MXCTEST_CHECK(MxCParticleType_initD(&BType, BTypeDef));

    MXCTEST_CHECK(MxCParticleType_registerType(&AType));
    MXCTEST_CHECK(MxCParticleType_registerType(&BType));

    struct MxPotentialHandle p, q, r;
    double pot_min = 0.01;
    double pot_max = 3.0;
    MXCTEST_CHECK(MxCPotential_create_coulomb(&p, 0.5, &pot_min, &pot_max, NULL, NULL));
    MXCTEST_CHECK(MxCPotential_create_coulomb(&q, 0.5, &pot_min, &pot_max, NULL, NULL));
    MXCTEST_CHECK(MxCPotential_create_coulomb(&r, 2.0, &pot_min, &pot_max, NULL, NULL));

    MXCTEST_CHECK(MxCBind_types(&p, &AType, &AType, 0));
    MXCTEST_CHECK(MxCBind_types(&q, &BType, &BType, 0));
    MXCTEST_CHECK(MxCBind_types(&r, &AType, &BType, 0));

    struct MxPointsTypeHandle ptTypes;
    MXCTEST_CHECK(MxCPointsType_init(&ptTypes));

    float *center = (float*)malloc(3 * sizeof(float));
    MXCTEST_CHECK(MxCUniverse_getCenter(&center));

    float *pos;
    int pid;
    unsigned int numPos = 1000;
    MXCTEST_CHECK(MxCRandomPoints(ptTypes.SolidCube, numPos, 0, 0, 0, &pos));

    for(unsigned int i = 0; i < numPos; i++) {
        float partPos[3];
        for(unsigned int j = 0; j < 3; j++) 
            partPos[j] = pos[3 * i + j] * 10 + center[j];
        MXCTEST_CHECK(MxCParticleType_createParticle(&AType, &pid, partPos, NULL));
    }

    struct MxParticleHandleHandle *neighbors;
    int numNeighbors;
    struct MxParticleHandleHandle part;
    MXCTEST_CHECK(MxCParticleType_getParticle(&AType, 0, &part));
    MXCTEST_CHECK(MxCParticleHandle_neighborsD(&part, 5.0, &neighbors, &numNeighbors));
    for(unsigned int i = 0; i < numNeighbors; i++) {
        MXCTEST_CHECK(MxCParticleHandle_become(&neighbors[i], &BType));
    }

    MXCTEST_CHECK(MxCTest_runQuiet(100));

    return 0;
}