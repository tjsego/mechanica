#include "MxCTest.h"

#include <MxCParticle.h>
#include <MxCPotential.h>
#include <MxCForce.h>
#include <MxCBind.h>
#include <MxCEvent.h>


HRESULT split(struct MxParticleTimeEventHandle *e) {
    struct MxParticleTypeHandle ptype;
    MXCTEST_CHECK(MxCParticleTimeEvent_getTargetType(e, &ptype));
    struct MxParticleHandleHandle targetParticle;
    if(MxCParticleTimeEvent_getTargetParticle(e, &targetParticle) != S_OK) {
        float *center = (float*)malloc(3 * sizeof(float));
        MXCTEST_CHECK(MxCUniverse_getCenter(&center));
        int pid;
        MXCTEST_CHECK(MxCParticleType_createParticle(&ptype, &pid, center, NULL));
    }
    else {
        double pradius;
        MXCTEST_CHECK(MxCParticleType_getRadius(&ptype, &pradius));
        struct MxParticleHandleHandle p;
        MXCTEST_CHECK(MxCParticleHandle_split(&targetParticle, &p));
        MXCTEST_CHECK(MxCParticleHandle_setRadius(&targetParticle, pradius));
        MXCTEST_CHECK(MxCParticleHandle_setRadius(&p, pradius));
    }
    return S_OK;
}

HRESULT passthrough(struct MxParticleTimeEventHandle *e) {
    return S_OK;
}


int main(int argc, char** argv) {
    MXCTEST_CHECK(MxCSimulator_init(NULL, 0));

    struct MxParticleTypeHandle MyCellType;
    MXCTEST_CHECK(MxCParticleType_init(&MyCellType));
    MXCTEST_CHECK(MxCParticleType_setMass(&MyCellType, 39.4));
    MXCTEST_CHECK(MxCParticleType_setRadius(&MyCellType, 0.2));
    MXCTEST_CHECK(MxCParticleType_setTargetTemperature(&MyCellType, 50.0));
    MXCTEST_CHECK(MxCParticleType_registerType(&MyCellType));

    struct MxPotentialHandle pot;
    double pot_tol = 1.0e-3;
    MXCTEST_CHECK(MxCPotential_create_lennard_jones_12_6(&pot, 0.275, 1.0, 9.5075e-06, 6.1545e-03, &pot_tol));
    MXCTEST_CHECK(MxCBind_types(&pot, &MyCellType, &MyCellType, 0));

    struct BerendsenHandle force;
    struct MxForceHandle force_base;
    MXCTEST_CHECK(MxCBerendsen_init(&force, 10.0));
    MXCTEST_CHECK(MxCBerendsen_toBase(&force, &force_base));
    MXCTEST_CHECK(MxCBind_force(&force_base, &MyCellType));
    
    struct MxParticleTimeEventTimeSetterEnumHandle setterEnum;
    MXCTEST_CHECK(MxCParticleTimeEventTimeSetterEnum_init(&setterEnum));

    struct MxParticleTimeEventParticleSelectorEnumHandle selectorEnum;
    MXCTEST_CHECK(MxCParticleTimeEventParticleSelectorEnum_init(&selectorEnum));

    MxParticleTimeEventMethodHandleFcn invokeMethod = (MxParticleTimeEventMethodHandleFcn)&split;
    MxParticleTimeEventMethodHandleFcn predicateMethod = (MxParticleTimeEventMethodHandleFcn)&passthrough;

    struct MxParticleTimeEventHandle pevent;
    MXCTEST_CHECK(MxCOnParticleTimeEvent(
        &pevent, &MyCellType, 0.05, &invokeMethod, &predicateMethod, setterEnum.EXPONENTIAL, 0.0, -1.0, selectorEnum.DEFAULT
    ));

    MXCTEST_CHECK(MxCTest_runQuiet(100));

    return 0;
}