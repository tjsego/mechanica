#include "MxTest.h"
#include <Mechanica.h>


struct AType : MxParticleType {

    AType() : MxParticleType(true) {
        species = new MxSpeciesList();
        species->insert("S1");
        species->insert("S2");
        species->insert("S3");
        style->newColorMapper(this, "S1");
        registerType();
    };

};


int main(int argc, char const *argv[])
{
    MxBoundaryConditionsArgsContainer *bcArgs = new MxBoundaryConditionsArgsContainer();
    bcArgs->setValue("x", BOUNDARY_PERIODIC | BOUNDARY_RESETTING);

    MxSimulator_Config config;
    config.setWindowless(true);
    config.universeConfig.dt = 0.1;
    config.universeConfig.dim = {15, 6, 6};
    config.universeConfig.spaceGridSize = {9, 3, 3};
    config.universeConfig.cutoff = 3;
    config.universeConfig.setBoundaryConditions(bcArgs);
    MXTEST_CHECK(MxSimulator_initC(config));

    AType *A = new AType();
    A = (AType*)A->get();

    MxFluxes::flux(A, A, "S1", 2);

    MxParticleHandle *a1, *a2;
    MxVector3f pos;
    pos = MxUniverse::getCenter() - MxVector3f(0, 1, 0);
    a1 = (*A)(&pos);
    pos = MxUniverse::getCenter() + MxVector3f(-5, 1, 0);
    MxVector3f vel(0.5, 0, 0);
    a2 = (*A)(&pos, &vel);

    a1->getSpecies()->setItem(a1->getSpecies()->species->index_of("S1"), 3.0);
    a2->getSpecies()->setItem(a2->getSpecies()->species->index_of("S1"), 0.0);

    // run the simulator
    MXTEST_CHECK(MxUniverse::step(MxUniverse::get()->getDt() * 100));

    return S_OK;
}
