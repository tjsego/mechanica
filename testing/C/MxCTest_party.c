#include "MxCTest.h"

#include <MxCParticle.h>
#include <MxCBond.h>
#include <MxCBind.h>
#include <MxCEvent.h>
#include <MxCSystem.h>


HRESULT vary_colors(struct MxTimeEventHandle *e) {
    double utime;
    MXCTEST_CHECK(MxCUniverse_getTime(&utime));
    float rate = (float)2.0 * M_PI * utime;
    float sf = (sinf(rate) + 1.0) * 0.5;
    float sf2 = (sinf(2.0 * rate) + 1) * 0.5;
    float cf = (cosf(rate) + 1.0) * 0.5;

    float gridColor[] = {sf, 0.0, sf2};
    float sceneBoxColor[] = {sf2, sf, 0.0};
    float lightDirection[] = {3.0 * (2.0 * sf - 1.0), 3.0 * (2.0 * cf - 1.0), 2.0};
    float lightColor[] = {(sf + 1.0) * 0.5, (sf + 1.0) * 0.5, (sf + 1.0) * 0.5};
    float ambientColor[] = {sf, sf, sf};
    
    MXCTEST_CHECK(MxCSystem_setGridColor(gridColor));
    MXCTEST_CHECK(MxCSystem_setSceneBoxColor(sceneBoxColor));
    MXCTEST_CHECK(MxCSystem_setShininess(1000.0 * sf + 10.0));
    MXCTEST_CHECK(MxCSystem_setLightDirection(lightDirection));
    MXCTEST_CHECK(MxCSystem_setLightColor(lightColor));
    MXCTEST_CHECK(MxCSystem_setAmbientColor(ambientColor));
    
    return S_OK;
}

HRESULT passthrough(struct MxTimeEventHandle *e) {
    return S_OK;
}


int main(int argc, char** argv) {
    double ATypeRadius = 0.1;

    struct MxSimulator_ConfigHandle config;
    MXCTEST_CHECK(MxCSimulator_Config_init(&config));
    MXCTEST_CHECK(MxCSimulator_Config_setWindowless(&config, 1));
    MXCTEST_CHECK(MxCSimulator_initC(&config, NULL, 0));

    struct MxParticleTypeHandle AType;
    MXCTEST_CHECK(MxCParticleType_init(&AType));
    MXCTEST_CHECK(MxCParticleType_setRadius(&AType, ATypeRadius));
    MXCTEST_CHECK(MxCParticleType_registerType(&AType));

    struct MxPotentialHandle pot;
    MXCTEST_CHECK(MxCPotential_create_harmonic(&pot, 100.0, 0.3, NULL, NULL, NULL));
    MXCTEST_CHECK(MxCBind_types(&pot, &AType, &AType, 0));

    float *center = (float*)malloc(3 * sizeof(float));
    MXCTEST_CHECK(MxCUniverse_getCenter(&center));

    float disp[] = {ATypeRadius + 0.07, 0.0, 0.0};
    float pos0[3], pos1[3];
    for(unsigned int i = 0; i < 3; i++) {
        pos0[i] = center[i] - disp[i];
        pos1[i] = center[i] + disp[i];
    }

    struct MxParticleHandleHandle p0, p1;
    int pid0, pid1;
    MXCTEST_CHECK(MxCParticleType_createParticle(&AType, &pid0, pos0, NULL));
    MXCTEST_CHECK(MxCParticleType_createParticle(&AType, &pid1, pos1, NULL));
    MXCTEST_CHECK(MxCParticleHandle_init(&p0, pid0));
    MXCTEST_CHECK(MxCParticleHandle_init(&p1, pid1));

    struct MxBondHandleHandle bh;
    MXCTEST_CHECK(MxCBondHandle_create(&bh, &pot, &p0, &p1));

    double dt;
    MXCTEST_CHECK(MxCUniverse_getDt(&dt));

    struct MxTimeEventHandle e;
    MxTimeEventMethodHandleFcn invokeMethod = (MxTimeEventMethodHandleFcn)&vary_colors;
    MxTimeEventMethodHandleFcn predicateMethod = (MxTimeEventMethodHandleFcn)&passthrough;
    struct MxTimeEventTimeSetterEnumHandle timeSetterEnum;
    MXCTEST_CHECK(MxCTimeEventTimeSetterEnum_init(&timeSetterEnum));
    MXCTEST_CHECK(MxCOnTimeEvent(&e, dt, &invokeMethod, &predicateMethod, timeSetterEnum.DEFAULT, 0.0, -1.0));

    MXCTEST_CHECK(MxCTest_runQuiet(100));

    return 0;
}