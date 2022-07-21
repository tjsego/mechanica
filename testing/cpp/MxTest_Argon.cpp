#include "MxTest.h"
#include <Mechanica.h>


// create a particle type

struct ArgonType : MxParticleType {

    ArgonType() : MxParticleType(true) {
        radius = 0.1;
        mass = 39.4;
        registerType();
    }

};


int main(int argc, char const *argv[])
{
    // dimensions of universe
    MxVector3f dim(10.);

    // new simulator
    MxSimulator_Config config;
    config.setWindowless(true);
    config.universeConfig.dim = dim;
    config.universeConfig.spaceGridSize = {5, 5, 5};
    config.universeConfig.cutoff = 1.;
    MXTEST_CHECK(MxSimulator_initC(config));

    // create a potential representing a 12-6 Lennard-Jones potential
    double pot_tol = 0.001;
    MxPotential *pot = MxPotential::lennard_jones_12_6(0.275, 1.0, 9.5075e-06, 6.1545e-03, &pot_tol);

    // Register and get the particle type; registration always only occurs once
    ArgonType *Argon = new ArgonType();
    Argon = (ArgonType*)Argon->get();

    // bind the potential with the *TYPES* of the particles
    MXTEST_CHECK(MxBind::types(pot, Argon, Argon));

    // random cube
    Argon->factory(2500);

    // run the simulator
    MXTEST_CHECK(MxUniverse::step(MxUniverse::get()->getDt() * 100));

    return S_OK;
}
