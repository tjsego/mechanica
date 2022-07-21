#include "MxTest.h"
#include <Mechanica.h>


struct AType : MxParticleType {

    AType() : MxParticleType(true) {
        radius = 1.0;
        mass = 2.5;
        style->setColor("MediumSeaGreen");
        registerType();
    };

};

struct SphereType : MxParticleType {

    SphereType() : MxParticleType(true) {
        radius = 3.0;
        setFrozen(true);
        style->setColor("orange");
        registerType();
    };
    
};

#include <MxLogger.h>

int main(int argc, char const *argv[])
{
    // dimensions of universe
    MxVector3f dim(30.f);

    float dist = 3.9;
    float offset = 6.0;

    MxBoundaryConditionsArgsContainer *bcArgs = new MxBoundaryConditionsArgsContainer();
    bcArgs->setValue("x", BOUNDARY_POTENTIAL);
    bcArgs->setValue("y", BOUNDARY_POTENTIAL);
    bcArgs->setValue("z", BOUNDARY_POTENTIAL);

    MxSimulator_Config config;
    config.setWindowless(true);
    config.universeConfig.dim = dim;
    config.universeConfig.cutoff = 7.0;
    config.universeConfig.spaceGridSize = {3, 3, 3};
    config.universeConfig.dt = 0.01;
    config.universeConfig.setBoundaryConditions(bcArgs);
    MXTEST_CHECK(MxSimulator_initC(config));

    AType *A = new AType();
    SphereType *Sphere = new SphereType();
    A = (AType*)A->get();
    Sphere = (SphereType*)Sphere->get();

    double p_d = 100.0;
    double p_a = 1.0;
    double p_min = -3.0;
    double p_max = 4.0;
    MxPotential *p = MxPotential::morse(&p_d, &p_a, NULL, &p_min, &p_max);

    MXTEST_CHECK(MxBind::types(p, A, Sphere));

    MxBoundaryConditions *bcs = MxUniverse::get()->getBoundaryConditions();
    MXTEST_CHECK(MxBind::boundaryCondition(p, &bcs->bottom, A));
    MXTEST_CHECK(MxBind::boundaryCondition(p, &bcs->top, A));
    MXTEST_CHECK(MxBind::boundaryCondition(p, &bcs->left, A));
    MXTEST_CHECK(MxBind::boundaryCondition(p, &bcs->right, A));
    MXTEST_CHECK(MxBind::boundaryCondition(p, &bcs->front, A));
    MXTEST_CHECK(MxBind::boundaryCondition(p, &bcs->back, A));
    
    MxVector3f pos = MxUniverse::getCenter() + MxVector3f(5, 0, 0);
    (*Sphere)(&pos);

    // above the sphere
    pos = MxUniverse::getCenter() + MxVector3f(5, 0, Sphere->radius + dist);
    (*A)(&pos);

    // bottom of simulation
    pos = {MxUniverse::getCenter()[0], MxUniverse::getCenter()[1], dist};
    (*A)(&pos);

    // top of simulation
    pos = {MxUniverse::getCenter()[0], MxUniverse::getCenter()[1], dim[2] - dist};
    (*A)(&pos);

    // left of simulation
    pos = {dist, MxUniverse::getCenter()[1] - offset, MxUniverse::getCenter()[2]};
    (*A)(&pos);

    // right of simulation
    pos = {dim[0] - dist, MxUniverse::get()->getCenter()[1] + offset, MxUniverse::get()->getCenter()[2]};
    (*A)(&pos);

    // front of simulation
    pos = {MxUniverse::get()->getCenter()[0], dist, MxUniverse::get()->getCenter()[2]};
    (*A)(&pos);

    // back of simulation
    pos = {MxUniverse::get()->getCenter()[0], dim[1] - dist, MxUniverse::get()->getCenter()[2]};
    (*A)(&pos);

    // run the simulator
    MXTEST_CHECK(MxUniverse::step(MxUniverse::get()->getDt() * 100));

    return S_OK;
}
