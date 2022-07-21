#include "MxTest.h"
#include <Mechanica.h>


HRESULT fission(const MxParticleTimeEvent &event) {
    MxParticleHandle *m = event.targetParticle;
    MxParticleHandle *d = m->fission();
    m->setRadius(event.targetType->radius);
    m->setMass(event.targetType->mass);
    d->setRadius(event.targetType->radius);
    d->setMass(event.targetType->mass);
    return S_OK;
};


struct CellType : MxParticleType {

    CellType() : MxParticleType(true) {
        mass = 20;
        target_energy = 0;
        radius = 0.5;
        dynamics = PARTICLE_OVERDAMPED;
        registerType();
    };

    void on_register() {
        MxOnParticleTimeEvent(this, 1.0, new MxParticleTimeEventMethod(&fission), NULL, (unsigned int)MxParticleTimeEventTimeSetterEnum::EXPONENTIAL, 0, -1);
    };

};


int main(int argc, char const *argv[])
{
    MxSimulator_Config config;
    config.universeConfig.dim = {20., 20., 20.};
    config.setWindowless(true);
    MXTEST_CHECK(MxSimulator_initC(config));

    double pot_d = 0.1;
    double pot_a = 6.0;
    double pot_min = -1.0;
    double pot_max = 1.0;
    MxPotential *pot = MxPotential::morse(&pot_d, &pot_a, NULL, &pot_min, &pot_max);

    CellType *Cell = new CellType();
    Cell = (CellType*)Cell->get();

    MXTEST_CHECK(MxBind::types(pot, Cell, Cell));

    Gaussian *rforce = MxForce::random(0.5, 0.0);
    MxBind::force(rforce, Cell);

    MxVector3f pos(10.0);
    (*Cell)(&pos);

    // run the simulator
    MXTEST_CHECK(MxUniverse::step(MxUniverse::get()->getDt() * 100));

    return S_OK;
}
