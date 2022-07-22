#include "MxTest.h"
#include <Mechanica.h>


struct BeadType : MxParticleType {

    BeadType() : MxParticleType(true) {
        mass = 1.0;
        radius = 0.1;
        dynamics = PARTICLE_OVERDAMPED;
        registerType();
    };

};


int main(int argc, char const *argv[])
{
    MxSimulator_Config config;
    config.universeConfig.dim = {20, 20, 20};
    config.universeConfig.cutoff = 8.0;
    config.setWindowless(true);
    MXTEST_CHECK(MxSimulator_initC(config));

    BeadType *Bead = new BeadType();
    Bead = (BeadType*)Bead->get();

    // simple harmonic potential to pull particles
    double pot_max = 3.0;
    MxPotential *pot = MxPotential::harmonic(1.0, 0.1, NULL, &pot_max);

    // make a ring of of 50 particles
    std::vector<MxVector3f> pts = MxPoints(MxPointsType::Ring, 50);

    // constuct a particle for each position, make
    // a list of particles
    MxParticleList beads;
    for(auto &p : pts) {
        MxVector3f pos = p * 5 + MxUniverse::getCenter();
        beads.insert((*Bead)(&pos));
    }

    // create an explicit bond for each pair in the
    // list of particles. The bind_pairwise method
    // searches for all possible pairs within a cutoff
    // distance and connects them with a bond.
    MXTEST_CHECK(MxBind::bonds(pot, &beads, 1));

    // run the simulator
    MXTEST_CHECK(MxUniverse::step(MxUniverse::get()->getDt() * 100));

    return S_OK;
}
