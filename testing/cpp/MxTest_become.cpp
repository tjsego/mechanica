#include "MxTest.h"
#include <Mechanica.h>


static std::vector<std::string> speciesNames = {"S1", "S2", "S3"};


struct AType : MxParticleType {

    AType() : MxParticleType(true) {
        radius = 1.0;
        species = new MxSpeciesList();
        for(auto &s : speciesNames) 
            species->insert(s);
        style = new MxStyle();
        style->newColorMapper(this, "S2");

        registerType();
    };

};

struct BType : MxParticleType {

    BType() : MxParticleType(true) {
        radius = 4.0;
        species = new MxSpeciesList();
        for(auto &s : speciesNames) 
            species->insert(s);
        style = new MxStyle();
        style->newColorMapper(this, "S2");

        registerType();
    };

};


int main(int argc, char const *argv[])
{
    MxSimulator_Config config;
    config.setWindowless(true);
    MXTEST_CHECK(MxSimulator_initC(config));

    AType *A = new AType();
    A = (AType*)A->get();
    BType *B = new BType();
    B = (BType*)B->get();

    MxParticleHandle *o = (*A)();
    MxStateVector *ostate = o->getSpecies();
    ostate->setItem(ostate->species->index_of("S2"), 0.5);
    MXTEST_CHECK(o->become(B));

    // run the simulator
    MXTEST_CHECK(MxUniverse::step(MxUniverse::get()->getDt() * 100));

    return S_OK;
}
