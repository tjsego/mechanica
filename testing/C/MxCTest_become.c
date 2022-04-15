#include "MxCTest.h"

#include <MxCUniverse.h>
#include <MxCParticle.h>
#include <MxCSpecies.h>
#include <MxCStyle.h>
#include <MxCStateVector.h>


int main(int argc, char** argv) {
    
    struct MxSimulator_ConfigHandle config;
    MXCTEST_CHECK(MxCSimulator_Config_init(&config));
    MXCTEST_CHECK(MxCSimulator_Config_setWindowless(&config, 1));
    MXCTEST_CHECK(MxCSimulator_initC(&config, NULL, 0));

    struct MxSpeciesHandle S1, S2, S3;
    MXCTEST_CHECK(MxCSpecies_initS(&S1, "S1"));
    MXCTEST_CHECK(MxCSpecies_initS(&S2, "S2"));
    MXCTEST_CHECK(MxCSpecies_initS(&S3, "S3"));
    struct MxSpeciesListHandle slistA, slistB;
    MXCTEST_CHECK(MxCSpeciesList_init(&slistA));
    MXCTEST_CHECK(MxCSpeciesList_insert(&slistA, &S1));
    MXCTEST_CHECK(MxCSpeciesList_insert(&slistA, &S2));
    MXCTEST_CHECK(MxCSpeciesList_insert(&slistA, &S3));
    MXCTEST_CHECK(MxCSpeciesList_init(&slistB));
    MXCTEST_CHECK(MxCSpeciesList_insert(&slistB, &S1));
    MXCTEST_CHECK(MxCSpeciesList_insert(&slistB, &S2));
    MXCTEST_CHECK(MxCSpeciesList_insert(&slistB, &S3));

    struct MxParticleTypeHandle AType, BType;

    MXCTEST_CHECK(MxCParticleType_init(&AType));
    MXCTEST_CHECK(MxCParticleType_init(&BType));

    MXCTEST_CHECK(MxCParticleType_registerType(&AType));
    MXCTEST_CHECK(MxCParticleType_registerType(&BType));

    MXCTEST_CHECK(MxCParticleType_setName(&AType, "AType"));
    MXCTEST_CHECK(MxCParticleType_setRadius(&AType, 1.0));
    MXCTEST_CHECK(MxCParticleType_setSpecies(&AType, &slistA));

    MXCTEST_CHECK(MxCParticleType_setName(&BType, "BType"));
    MXCTEST_CHECK(MxCParticleType_setRadius(&BType, 4.0));
    MXCTEST_CHECK(MxCParticleType_setSpecies(&BType, &slistB));

    struct MxStyleHandle styleA, styleB;
    MXCTEST_CHECK(MxCStyle_init(&styleA));
    MXCTEST_CHECK(MxCStyle_newColorMapper(&styleA, &AType, "S2", "rainbow", 0.0, 1.0));
    MXCTEST_CHECK(MxCParticleType_setStyle(&AType, &styleA));
    
    MXCTEST_CHECK(MxCStyle_init(&styleB));
    MXCTEST_CHECK(MxCStyle_newColorMapper(&styleB, &BType, "S2", "rainbow", 0.0, 1.0));
    MXCTEST_CHECK(MxCParticleType_setStyle(&BType, &styleB));

    struct MxParticleHandleHandle part;
    int pid;
    MXCTEST_CHECK(MxCParticleType_createParticle(&AType, &pid, NULL, NULL));
    MXCTEST_CHECK(MxCParticleHandle_init(&part, pid));

    struct MxStateVectorHandle svec;
    struct MxSpeciesListHandle partSList;
    MXCTEST_CHECK(MxCParticleHandle_getSpecies(&part, &svec));
    MXCTEST_CHECK(MxCStateVector_getSpecies(&svec, &partSList));
    unsigned int sid;
    MXCTEST_CHECK(MxCSpeciesList_indexOf(&partSList, "S2", &sid));
    MXCTEST_CHECK(MxCStateVector_setItem(&svec, sid, 0.5));

    MXCTEST_CHECK(MxCParticleHandle_become(&part, &BType));

    MXCTEST_CHECK(MxCTest_runQuiet(100));

    return 0;
}