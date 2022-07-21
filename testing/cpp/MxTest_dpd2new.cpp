#include "MxTest.h"
#include <Mechanica.h>


struct AType : MxParticleType {

    AType() : MxParticleType(true) {
        radius = 0.3;
        dynamics = PARTICLE_OVERDAMPED;
        mass = 10.0;
        style->setColor("seagreen");
        registerType();
    };

};


int main(int argc, char const *argv[])
{
    MxBoundaryConditionsArgsContainer bcArgs;
    bcArgs.setValue("x", BOUNDARY_PERIODIC);
    bcArgs.setValue("y", BOUNDARY_PERIODIC);
    bcArgs.setValue("z", BOUNDARY_NO_SLIP);

    MxSimulator_Config config;
    config.setWindowless(true);
    config.universeConfig.dt = 0.1;
    config.universeConfig.dim = {15, 12, 10};
    MXTEST_CHECK(MxSimulator_initC(config));

    AType *A = new AType();
    A = (AType*)A->get();

    double dpd_alpha = 0.3;
    double dpd_gamma = 1.0;
    double dpd_sigma = 1.0;
    double dpd_cutoff = 0.6;
    MxPotential *dpd = MxPotential::dpd(&dpd_alpha, &dpd_gamma, &dpd_sigma, &dpd_cutoff);

    double dpd_wall_alpha = 0.5;
    double dpd_wall_gamma = 10.0;
    double dpd_wall_sigma = 1.0;
    double dpd_wall_cutoff = 0.1;
    MxPotential *dpd_wall = MxPotential::dpd(&dpd_wall_alpha, &dpd_wall_gamma, &dpd_wall_sigma, &dpd_wall_cutoff);

    double dpd_left_alpha = 1.0;
    double dpd_left_gamma = 100.0;
    double dpd_left_sigma = 0.0;
    double dpd_left_cutoff = 0.5;
    MxPotential *dpd_left = MxPotential::dpd(&dpd_left_alpha, &dpd_left_gamma, &dpd_left_sigma, &dpd_left_cutoff);

    MXTEST_CHECK(MxBind::types(dpd, A, A));
    MXTEST_CHECK(MxBind::boundaryCondition(dpd_wall, &MxUniverse::get()->getBoundaryConditions()->top, A));
    MXTEST_CHECK(MxBind::boundaryCondition(dpd_left, &MxUniverse::get()->getBoundaryConditions()->left, A));

    A->factory(1000);

    // run the simulator
    MXTEST_CHECK(MxUniverse::step(MxUniverse::get()->getDt() * 100));

    return S_OK;
}
