#include "MxTest.h"
#include <Mechanica.h>


struct CellType : MxParticleType {

    CellType() : MxParticleType(true) {
        radius = 0.5;
        target_energy = 0;
        radius = 0.5;
        registerType();
    };

};


HRESULT fission(const MxParticleTimeEvent &event) {
    event.targetParticle->fission();
    return S_OK;
};


int main(int argc, char const *argv[])
{
    MxSimulator_Config config;
    config.setWindowless(true);
    config.universeConfig.dim = {20, 20, 20};
    MXTEST_CHECK(MxSimulator_initC(config));

    double pot_min = 0.1;
    double pot_max = 1.0;
    MxPotential *pot = MxPotential::coulomb(10, &pot_min, &pot_max);

    CellType *Cell = new CellType();
    Cell = (CellType*)Cell->get();

    MXTEST_CHECK(MxBind::types(pot, Cell, Cell));

    MxVector3f pos(10.0);
    (*Cell)(&pos);

    MxOnParticleTimeEvent(Cell, 1.0, new MxParticleTimeEventMethod(&fission), NULL, 0, 0, -1, (unsigned int)MxTimeEventTimeSetterEnum::EXPONENTIAL);

    // run the simulator
    MXTEST_CHECK(MxUniverse::step(MxUniverse::get()->getDt() * 100));

    return S_OK;
}
