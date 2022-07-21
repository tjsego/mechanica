#include "MxTest.h"
#include <Mechanica.h>


struct AType : MxParticleType {

    AType() : MxParticleType(true) {
        radius = 0.5;
        mass = 5.0;
        dynamics = PARTICLE_OVERDAMPED;
        style->setColor("MediumSeaGreen");
        registerType();
    }

};

struct BType : MxParticleType {

    BType() : MxParticleType(true) {
        radius = 0.2;
        mass = 1.0;
        dynamics = PARTICLE_OVERDAMPED;
        style->setColor("skyblue");
        registerType();
    }

};

struct CType : MxParticleType {

    CType() : MxParticleType(true) {
        radius = 10.0;
        setFrozen(true);
        style->setColor("orange");
        registerType();
    }

};


HRESULT update(const MxTimeEvent &e) {
    BType B;
    MxParticleType *B_p = B.get();
    std::cout << e.times_fired << ", " << B_p->items()->getCenterOfMass() << std::endl;
    return S_OK;
};


int main(int argc, char const *argv[])
{
    MxVector3f dim(30.);

    MxSimulator_Config config;
    config.setWindowless(true);
    config.universeConfig.dim = dim;
    config.universeConfig.cutoff = 5.0;
    config.universeConfig.dt = 0.0005;
    MXTEST_CHECK(MxSimulator_initC(config));

    AType *A = new AType();
    BType *B = new BType();
    CType *C = new CType();
    A = (AType*)A->get();
    B = (BType*)B->get();
    C = (CType*)C->get();

    MxVector3f c_pos = MxUniverse::getCenter();
    (*C)(&c_pos);

    // make a ring of of 50 particles
    std::vector<MxVector3f> pts = MxPoints(MxPointsType::Ring, 100);
    for(auto &p : pts) {
        p = p * (B->radius + C->radius) + MxUniverse::getCenter() - MxVector3f(0, 0, 1);
        std::cout << p << std::endl;
    }
    B->factory(pts.size(), &pts);

    double pot_pc_m = 2.0;
    double pot_pc_max = 5.0;
    MxPotential *pc = MxPotential::glj(30.0, &pot_pc_m, NULL, NULL, NULL, NULL, &pot_pc_max);
    
    double pot_pa_m = 2.5;
    double pot_pa_max = 3.0;
    MxPotential *pa = MxPotential::glj(3.0, &pot_pa_m, NULL, NULL, NULL, NULL, &pot_pa_max);
    
    double pot_pb_m = 4.0;
    double pot_pb_max = 1.0;
    MxPotential *pb = MxPotential::glj(1.0, &pot_pb_m, NULL, NULL, NULL, NULL, &pot_pb_max);
    
    double pot_pab_m = 2.0;
    double pot_pab_max = 1.0;
    MxPotential *pab = MxPotential::glj(1.0, &pot_pab_m, NULL, NULL, NULL, NULL, &pot_pab_max);
    
    MxPotential *ph = MxPotential::harmonic(200.0, 0.001);

    MXTEST_CHECK(MxBind::types(pc, A, C));
    MXTEST_CHECK(MxBind::types(pc, B, C));
    MXTEST_CHECK(MxBind::types(pa, A, A));
    MXTEST_CHECK(MxBind::types(pab, A, B));

    Gaussian *r = MxForce::random(5, 0);

    MXTEST_CHECK(MxBind::force(r, A));
    MXTEST_CHECK(MxBind::force(r, B));

    MxBind::bonds(ph, B->items(), 1.0);

    // Implement the callback
    MxTimeEventMethod update_ev(update);
    MxOnTimeEvent(0.01, &update_ev);

    // run the simulator
    MXTEST_CHECK(MxUniverse::step(MxUniverse::get()->getDt() * 100));

    return S_OK;
}
