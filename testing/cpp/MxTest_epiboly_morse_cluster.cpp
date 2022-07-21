#include "MxTest.h"
#include <Mechanica.h>


struct BType : MxParticleType {

    BType() : MxParticleType(true) {
        radius = 0.25;
        dynamics = PARTICLE_OVERDAMPED;
        mass = 15.0;
        style->setColor("skyblue");
        registerType();
    };

};

struct CType : MxClusterParticleType {

    CType() : MxClusterParticleType(true) {
        radius = 5.0;
        registerType();
    };

};

struct YolkType : MxParticleType {

    YolkType() : MxParticleType(true) {
        radius = 10.0;
        mass = 1000000.0;
        dynamics = PARTICLE_OVERDAMPED;
        setFrozen(true);
        style->setColor("gold");
        registerType();
    };

};


static YolkType *Yolk = NULL;
static int yolk_id = -1;


static HRESULT split(const MxParticleTimeEvent &event) {
    MxClusterParticleHandle *particle = (MxClusterParticleHandle*)event.targetParticle;
    MxClusterParticleType *ptype = (MxClusterParticleType*)event.targetType;

    MxParticleHandle yolk(yolk_id, Yolk->id);
    MxVector3f axis = particle->getPosition() - yolk.getPosition();
    particle->split(&axis);
    return S_OK;
}


int main(int argc, char const *argv[])
{
    MxVector3f pos;

    MxSimulator_Config config;
    config.setWindowless(true);
    config.universeConfig.dim = {30, 30, 30};
    config.universeConfig.cutoff = 3;
    config.universeConfig.dt = 0.001;
    config.universeConfig.spaceGridSize = {6, 6, 6};
    MXTEST_CHECK(MxSimulator_initC(config));

    BType *B = new BType();
    CType *C = new CType();
    Yolk = new YolkType();
    B = (BType*)B->get();
    C = (CType*)C->get();
    Yolk = (YolkType*)Yolk->get();

    double total_height = 2.0 * Yolk->radius + 2.0 * C->radius;
    double yshift = 1.5 * (total_height / 2.0 - Yolk->radius);
    double cshift = total_height / 2.0 - C->radius - 1.0;

    pos = MxUniverse::getCenter() - MxVector3f(0, 0, -yshift);
    MxParticleHandle *yolk = (*Yolk)(&pos);
    yolk_id = yolk->id;

    pos = yolk->getPosition() + MxVector3f(0, 0, yolk->getRadius() + C->radius - 5.0);
    MxParticleHandle *c = (*C)(&pos);

    B->factory(8000);

    double pot_a = 6;

    double pb_d = 1.0;
    double pb_r0 = 0.5;
    double pb_min = 0.01;
    double pb_max = 3.0;
    bool pb_shifted = false;
    MxPotential *pb = MxPotential::morse(&pb_d, &pot_a, &pb_r0, &pb_min, &pb_max, NULL, &pb_shifted);

    double pub_d = 1.0;
    double pub_r0 = 0.5;
    double pub_min = 0.01;
    double pub_max = 3.0;
    bool pub_shifted = false;
    MxPotential *pub = MxPotential::morse(&pub_d, &pot_a, &pub_r0, &pub_min, &pub_max, NULL, &pub_shifted);

    double py_d = 0.1;
    double py_min = -5.0;
    double py_max = 1.0;
    MxPotential *py = MxPotential::morse(&py_d, &pot_a, NULL, &py_min, &py_max);

    Gaussian *rforce = MxForce::random(500.0, 0.0, 0.0001);

    MXTEST_CHECK(MxBind::force(rforce, B));
    MXTEST_CHECK(MxBind::types(pb, B, B, true));
    MXTEST_CHECK(MxBind::types(pub, B, B));
    MXTEST_CHECK(MxBind::types(py, Yolk, B));

    MxParticleTimeEventMethod invokeMethod(split);
    MxOnParticleTimeEvent(C, 1.0, &invokeMethod, NULL, (unsigned int)MxParticleTimeEventParticleSelectorEnum::LARGEST);

    // run the simulator
    MXTEST_CHECK(MxUniverse::step(MxUniverse::get()->getDt() * 100));

    return S_OK;
}
