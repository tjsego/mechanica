#include "MxTest.h"
#include <Mechanica.h>

#include <string>


static int colorCount;


struct BeadType : MxParticleType {

    int i;

    BeadType() : MxParticleType(true) {
        radius = 3.0;
        species = new MxSpeciesList();
        species->insert("S1");
        style->newColorMapper(this, "S1");
        registerType();
    };

};


HRESULT keypress(MxKeyEvent *e) {
    std::vector<std::string> mapperNames = MxColorMapper::getNames();

    if(std::strcmp(e->keyName().c_str(), "n")) {
        colorCount = (colorCount + 1) % mapperNames.size();
    } 
    else if(std::strcmp(e->keyName().c_str(), "p")) {
        colorCount = (colorCount - 1) % mapperNames.size();
    } 
    else {
        return S_OK;
    }

    MxParticleType *Bead = BeadType().get();
    Bead->style->setColorMap(mapperNames[colorCount]);
    return S_OK;
};


int main(int argc, char const *argv[])
{
    MxSimulator_Config config;
    config.setWindowless(true);
    MXTEST_CHECK(MxSimulator_initC(config));

    BeadType *Bead = new BeadType();
    Bead = (BeadType*)Bead->get();

    std::vector<MxVector3f> pts = MxRandomPoints(MxPointsType::Ring, 100);
    for(auto &p : pts) {
        MxVector3f pt = p * 4 + MxUniverse::getCenter();
        (*Bead)(&pt);
    }

    colorCount = 0;
    MxKeyEventHandlerType cb = &keypress;
    MxKeyEvent::addHandler(&cb);

    // run the simulator
    MXTEST_CHECK(MxUniverse::step(MxUniverse::get()->getDt() * 100));

    return S_OK;
}
