#include "MxTest.h"
#include <Mechanica.h>


struct AType : MxParticleType {

    AType() : MxParticleType(true) {
        radius = 0.05;
        dynamics = PARTICLE_OVERDAMPED;
        mass = 10.0;
        style->setColor("seagreen");
        registerType();
    };

};


int main(int argc, char const *argv[])
{
    MxBoundaryConditionsArgsContainer *bcArgs = new MxBoundaryConditionsArgsContainer();
    bcArgs->setVelocity("top", {0, 0, 0});
    bcArgs->setVelocity("bottom", {0, 0, 0});
    
    MxSimulator_Config config;
    config.universeConfig.dt = 0.1;
    config.universeConfig.dim = {15, 12, 10};
    config.universeConfig.spaceGridSize = {7, 6, 5};
    config.universeConfig.cutoff = 0.5;
    config.universeConfig.setBoundaryConditions(bcArgs);
    config.setWindowless(true);
    MXTEST_CHECK(MxSimulator_initC(config));
    
    AType *A = new AType();
    A = (AType*)A->get();

    double dpd_alpha = 10.0;
    double dpd_sigma = 1.0;
    MxPotential *dpd = MxPotential::dpd(&dpd_alpha, NULL, &dpd_sigma);

    MXTEST_CHECK(MxBind::types(dpd, A, A));
    
    // Driving pressure
    MxConstantForce *pressure = new MxConstantForce({0.1, 0, 0});
    MXTEST_CHECK(MxBind::force(pressure, A));
    
    A->factory(10000);
    
    // run the simulator
    MXTEST_CHECK(MxUniverse::step(MxUniverse::get()->getDt() * 100));
    
    return S_OK;
}
