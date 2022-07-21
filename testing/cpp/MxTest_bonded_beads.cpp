#include "MxTest.h"
#include <Mechanica.h>


struct BeadType : MxParticleType {

    BeadType() : MxParticleType(true) {
        mass = 0.4;
        radius = 0.2;
        dynamics = PARTICLE_OVERDAMPED;
        registerType();
    };

};


int main(int argc, char const *argv[])
{
    // new simulator
    MxSimulator_Config config;
    config.universeConfig.dim = {20., 20., 20.};
    config.universeConfig.cutoff = 8.0;
    config.setWindowless(true);
    MXTEST_CHECK(MxSimulator_initC(config));

    BeadType *Bead = new BeadType();
    Bead = (BeadType*)Bead->get();

    double pot_bb_min = 0.1;
    double pot_bb_max = 1.0;
    MxPotential *pot_bb = MxPotential::coulomb(0.1, &pot_bb_min, &pot_bb_max);

    // hamonic bond between particles
    double pot_bond_max = 2.0;
    MxPotential *pot_bond = MxPotential::harmonic(0.4, 0.2, NULL, &pot_bond_max);

    // angle bond potential
    double pot_ang_tol = 0.01;
    MxPotential *pot_ang = MxPotential::harmonic_angle(0.01, 0.85 * M_PI, NULL, NULL, &pot_ang_tol);

    // bind the potential with the *TYPES* of the particles
    MXTEST_CHECK(MxBind::types(pot_bb, Bead, Bead));

    // create a random force. In overdamped dynamcis, we neeed a random force to
    // enable the objects to move around, otherwise they tend to get trapped
    // in a potential
    Gaussian *rforce = MxForce::random(0.1, 0);

    // bind it just like any other force
    MXTEST_CHECK(MxBind::force(rforce, Bead));

    // Place particles
    MxParticleHandle *p = NULL;     // previous bead
    MxParticleHandle *bead;         // current bead
    MxParticleHandle *n;            // new bead

    MxVector3f pos(4.0, 10.f, 10.f);
    bead = (*Bead)(&pos);

    while(pos[0] < 16.f) {
        pos[0] += 0.15;
        n = (*Bead)(&pos);
        MxBond::create(pot_bond, bead, n);
        if(p != NULL) 
            MxAngle::create(pot_ang, p, bead, n);
        p = bead;
        bead = n;
    }

    // run the simulator
    MXTEST_CHECK(MxUniverse::step(MxUniverse::get()->getDt() * 100));

    return S_OK;
}
