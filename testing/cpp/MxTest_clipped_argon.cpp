#include "MxTest.h"
#include <Mechanica.h>


// create a particle type
// all new Particle derived types are automatically
// registered with the universe

struct ArgonType : MxParticleType {

    ArgonType() : MxParticleType(true) {
        radius = 0.1;
        mass = 39.4;
        registerType();
    };
};


int main(int argc, char const *argv[])
{
    // new simulator
    MxSimulator_Config config;
    config.setWindowless(true);
    config.setWindowSize({900, 900});
    config.clipPlanes = {
        MxPlaneEquation({1, 1, 0.5}, {2, 2, 2}), 
        MxPlaneEquation({-1, 1, -1}, {5, 5, 5})
    };
    config.universeConfig.dim = {10, 10, 10};
    MXTEST_CHECK(MxSimulator_initC(config));

    // create a potential representing a 12-6 Lennard-Jones potential
    // A The first parameter of the Lennard-Jones potential.
    // B The second parameter of the Lennard-Jones potential.
    // cutoff
    double pot_tol = 0.001;
    MxPotential *pot = MxPotential::lennard_jones_12_6(0.275, 3.0, 9.5075e-06, 6.1545e-03, &pot_tol);

    ArgonType *Argon = new ArgonType();
    Argon = (ArgonType*)Argon->get();

    // bind the potential with the *TYPES* of the particles
    MXTEST_CHECK(MxBind::types(pot, Argon, Argon));

    // uniform random cube
    Argon->factory(13000);

    // run the simulator
    MXTEST_CHECK(MxUniverse::step(MxUniverse::get()->getDt() * 20));

    return S_OK;
}
