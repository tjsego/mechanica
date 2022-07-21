#include "MxTest.h"
#include <Mechanica.h>


struct AType : MxParticleType {

    AType() : MxParticleType(true) {
        radius = 0.1;
        dynamics = PARTICLE_OVERDAMPED;
        style = new MxStyle("MediumSeaGreen");
        registerType();
    };

};

struct BType : MxParticleType {

    BType() : MxParticleType(true) {
        radius = 0.1;
        dynamics = PARTICLE_OVERDAMPED;
        style = new MxStyle("skyblue");
        registerType();
    };
    
};


int main(int argc, char const *argv[])
{
    MxSimulator_Config config;
    config.setWindowless(true);
    config.universeConfig.cutoff = 3.0;
    MXTEST_CHECK(MxSimulator_initC(config));

    AType *A = new AType();
    BType *B = new BType();
    A = (AType*)A->get();
    B = (BType*)B->get();

    double pot_min = 0.01;
    double pot_max = 3.0;
    MxPotential *p = MxPotential::coulomb(0.5, &pot_min, &pot_max);
    MxPotential *q = MxPotential::coulomb(0.5, &pot_min, &pot_max);
    MxPotential *r = MxPotential::coulomb(2.0, &pot_min, &pot_max);

    MXTEST_CHECK(MxBind::types(p, A, A));
    MXTEST_CHECK(MxBind::types(q, B, B));
    MXTEST_CHECK(MxBind::types(r, A, B));

    int nr_parts = 1000;
    std::vector<MxVector3f> pos;
    pos.reserve(nr_parts);
    for(auto &p_pos : MxRandomPoints(MxPointsType::SolidCube, 1000)) 
        pos.push_back(p_pos * 10 + MxUniverse::getCenter());

    A->factory(nr_parts, &pos);

    MxParticleHandle *a = A->items()->item(0);
    float n_dist = 5.0;
    MxParticleList *n_list = a->neighbors(&n_dist);
    for(int i = 0; i < n_list->nr_parts; i++) 
        n_list->item(i)->become(B);

    // run the simulator
    MXTEST_CHECK(MxUniverse::step(MxUniverse::get()->getDt() * 100));

    return S_OK;
}
