#include "MxTest.h"
#include <Mechanica.h>


static std::vector<std::string> speciesNames = {"S1", "S2", "S3"};


struct AType : MxParticleType {

    AType() : MxParticleType(true) {
        radius = 0.1;
        species = new MxSpeciesList();
        for(auto &s : speciesNames) species->insert(s);
        style->newColorMapper(this, "S1");
        registerType();
    };

};

struct BType : MxParticleType {

    BType() : MxParticleType(true) {
        radius = 0.1;
        species = new MxSpeciesList();
        for(auto &s : speciesNames) species->insert(s);
        style->newColorMapper(this, "S1");
        registerType();
    };
    
};


HRESULT spew(const MxParticleTimeEvent &event) {
    std::cout << "Spew" << std::endl;
    MxStateVector *sv = event.targetParticle->getSpecies();
    int32_t s_idx = sv->species->index_of("S1");
    sv->setItem(s_idx, 500);
    MxSpeciesValue(*(sv->item(s_idx)), sv, s_idx).secrete(250.0, 1.0);
    return S_OK;
}


int main(int argc, char const *argv[])
{
    MxBoundaryConditionsArgsContainer *bcArgs = new MxBoundaryConditionsArgsContainer();
    bcArgs->setValueAll(BOUNDARY_FREESLIP);

    MxSimulator_Config config;
    config.universeConfig.dim = {6.5, 6.5, 6.5};
    config.universeConfig.setBoundaryConditions(bcArgs);
    config.setWindowless(true);
    MXTEST_CHECK(MxSimulator_initC(config));

    AType *A = new AType();
    BType *B = new BType();
    A = (AType*)A->get();
    B = (BType*)B->get();

    MxFluxes::flux(A, A, "S1", 5, 0.005);

    A->factory(10000);

    // Grab a particle
    MxParticleHandle *o = A->parts.item(0);

    // Change type to B, since there is no flux rule between A and B
    o->become(B);

    MxParticleTimeEventMethod spew_e(spew);
    MxOnParticleTimeEvent(B, 0.3, &spew_e);

    // run the simulator
    MXTEST_CHECK(MxUniverse::step(0.35));

    return S_OK;
}
