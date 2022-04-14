#include "MxCTest.h"

#include <MxCParticle.h>
#include <MxCPotential.h>
#include <MxCBind.h>
#include <MxCClipPlane.h>
#include <MxCEvent.h>
#include <MxCSystem.h>


struct MxClipPlaneHandle *cp3 = NULL;
float **simCenter = NULL;


HRESULT rotate_clip(struct MxTimeEventHandle *e) {
    double simtime;
    MxCUniverse_getTime(&simtime);
    float cf = 2.0 * M_PI * simtime / 20.0;
    float normal[3] = {0.0, cos(cf), sin(cf)};
    MxCClipPlane_setEquationPN(cp3, *simCenter, normal);
    return S_OK;
}

HRESULT predicate(struct MxTimeEventHandle *e) {
    return S_OK;
}


int main(int argc, char** argv) {
    struct MxSimulator_ConfigHandle config;
    struct MxUniverseConfigHandle uconfig;

    float cp1_point[] = {2.0, 2.0, 2.0};
    float cp1_normal[] = {1.0, 1.0, 0.5};
    float cp2_point[] = {5.0, 5.0, 5.0};
    float cp2_normal[] = {-1.0, 1.0, -1.0};
    float *cp1_eq = (float*)malloc(4 * sizeof(float));
    float *cp2_eq = (float*)malloc(4 * sizeof(float));
    float cp_eqs[8];
    MXCTEST_CHECK(MxCPlaneEquationFPN(cp1_point, cp1_normal, &cp1_eq));
    MXCTEST_CHECK(MxCPlaneEquationFPN(cp2_point, cp2_normal, &cp2_eq));
    for(unsigned int i = 0; i < 4; i++) {
        cp_eqs[i] = cp1_eq[i];
        cp_eqs[i + 4] = cp2_eq[i];
    }

    MXCTEST_CHECK(MxCSimulator_Config_init(&config));
    MXCTEST_CHECK(MxCSimulator_Config_setWindowSize(&config, 900, 900));
    MXCTEST_CHECK(MxCSimulator_Config_setClipPlanes(&config, cp_eqs, 2));
    MXCTEST_CHECK(MxCSimulator_Config_getUniverseConfig(&config, &uconfig));
    MXCTEST_CHECK(MxCSimulator_initC(&config, NULL, 0));

    struct MxCParticleType ArgonTypeDef = MxCParticleTypeDef_init();
    ArgonTypeDef.radius = 0.1;
    ArgonTypeDef.mass = 39.4;

    struct MxParticleTypeHandle ArgonType;
    MXCTEST_CHECK(MxCParticleType_initD(&ArgonType, ArgonTypeDef));
    MXCTEST_CHECK(MxCParticleType_registerType(&ArgonType));

    struct MxPotentialHandle pot;
    double pot_tol = 1.0e-3;
    MXCTEST_CHECK(MxCPotential_create_lennard_jones_12_6(&pot, 0.275, 3.0, 9.5075e-06, 6.1545e-03, &pot_tol));

    MXCTEST_CHECK(MxCBind_types(&pot, &ArgonType, &ArgonType, 0));

    int pid;
    for(unsigned int i = 0; i < 13000; i++) 
        MXCTEST_CHECK(MxCParticleType_createParticle(&ArgonType, &pid, NULL, NULL));

    float *center = (float*)malloc(3 * sizeof(float));
    MXCTEST_CHECK(MxCUniverse_getCenter(&center));
    simCenter = &center;
    float normal[3] = {0.0, 1.0, 0.0};
    struct MxClipPlaneHandle _cp3;
    MXCTEST_CHECK(MxCClipPlanes_createPN(&_cp3, center, normal));
    cp3 = &_cp3;

    struct MxTimeEventHandle timeEventHandle;
    double dt;
    MXCTEST_CHECK(MxCUniverse_getDt(&dt));
    struct MxTimeEventTimeSetterEnumHandle timeSetterEnums;
    MXCTEST_CHECK(MxCTimeEventTimeSetterEnum_init(&timeSetterEnums));

    MxTimeEventMethodHandleFcn invokeMethod = rotate_clip;
    MxTimeEventMethodHandleFcn predicateMethod = predicate;
    MXCTEST_CHECK(MxCOnTimeEvent(&timeEventHandle, dt, &invokeMethod, &predicateMethod, timeSetterEnums.DEFAULT, 0.0, -1.0));

    MXCTEST_CHECK(MxCSystem_cameraViewRight());

    MXCTEST_CHECK(MxCTest_runQuiet(20));

    return 0;
}