#include "MxTest.h"
#include <Mechanica.h>


std::vector<std::string> speciesNames = {"S1", "S2", "S3"};


struct AType : MxParticleType {

    AType() : MxParticleType(true) {
        radius = 0.1;
        species = new MxSpeciesList();
        for(auto &s : speciesNames) species->insert(s);
        style->newColorMapper(this, "S1");
        registerType();
    };

};

struct ProducerType : MxParticleType {

    ProducerType() : MxParticleType(true) {
        radius = 0.1;
        species = new MxSpeciesList();
        for(auto &s : speciesNames) species->insert(s);
        style->newColorMapper(this, "S1");
        registerType();
    };

};

struct ConsumerType : MxParticleType {

    ConsumerType() : MxParticleType(true) {
        radius = 0.1;
        species = new MxSpeciesList();
        for(auto &s : speciesNames) species->insert(s);
        style->newColorMapper(this, "S1");
        registerType();
    };

};


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
    ProducerType *Producer = new ProducerType();
    ConsumerType *Consumer = new ConsumerType();
    A = (AType*)A->get();
    Producer = (ProducerType*)Producer->get();
    Consumer = (ConsumerType*)Consumer->get();

    // define fluxes between objects types
    MxFluxes::flux(A, A, "S1", 5.0, 0.0);
    MxFluxes::secrete(Producer, A, "S1", 5.0, 0.0);
    MxFluxes::uptake(A, Consumer, "S1", 10.0, 500.0);

    // make a bunch of objects
    A->factory(10000);

    // Grab some objects
    MxParticleHandle *producer = A->parts.item(0);
    MxParticleHandle *consumer = A->parts.item(A->parts.nr_parts - 1);

    // Change types
    MXTEST_CHECK(producer->become(Producer));
    MXTEST_CHECK(consumer->become(Consumer));

    // Set initial condition
    producer->getSpecies()->setItem(producer->getSpecies()->species->index_of("S1"), 2000.0);

    // run the simulator
    MXTEST_CHECK(MxUniverse::step(MxUniverse::get()->getDt() * 100));

    return S_OK;
}
