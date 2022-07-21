#include "MxTest.h"
#include <Mechanica.h>


// create a particle type

struct ArgonType : MxParticleType {

    ArgonType() : MxParticleType(true) {
        mass = 39.4;
        target_energy = 10000.0;
        registerType();
    }

};


int main(int argc, char const *argv[])
{
    // potential cutoff distance
    double cutoff = 1.0;

    // dimensions of universe
    MxVector3f dim(10.);

    // new simulator
    MxSimulator_Config config;
    config.setWindowless(true);
    config.universeConfig.dim = dim;
    config.universeConfig.cutoff = cutoff;
    MXTEST_CHECK(MxSimulator_initC(config));

    // create a potential representing a 12-6 Lennard-Jones potential
    double pot_tol = 0.001;
    MxPotential *pot = MxPotential::lennard_jones_12_6(0.275, cutoff, 9.5075e-06, 6.1545e-03, &pot_tol);

    // Register and get the particle type; registration always only occurs once
    ArgonType *Argon = new ArgonType();
    Argon = (ArgonType*)Argon->get();

    // bind the potential with the *TYPES* of the particles
    MXTEST_CHECK(MxBind::types(pot, Argon, Argon));

    // create a thermostat, coupling time constant determines how rapidly the
    // thermostat operates, smaller numbers mean thermostat acts more rapidly
    Berendsen *tstat = MxForce::berendsen_tstat(10.0);

    // bind it just like any other force
    MXTEST_CHECK(MxBind::force(tstat, Argon));

    int nr_parts = 100;
    std::vector<MxVector3f> velocities;
    velocities.reserve(nr_parts);
    for(int i = 0; i < nr_parts; i++) 
        velocities.push_back(MxRandomUnitVector() * 0.1);
    Argon->factory(nr_parts, NULL, &velocities);

    // run the simulator
    MXTEST_CHECK(MxUniverse::step(MxUniverse::get()->getDt() * 100));

    return S_OK;
}
