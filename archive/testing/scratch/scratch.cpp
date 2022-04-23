#include <iostream>
#include <string>
#include <vector>
#include <utility>

#include <Magnum/GL/Context.h>

#include <types/mx_cast.h>
#include <MxSimulator.h>
#include <MxLogger.h>
#include <MxParticle.h>
#include <MxPotential.h>
#include <MxBind.hpp>
#include <MxForce.h>
#include <MxUtil.h>
#include <rendering/NOMStyle.hpp>
#include <event/MxParticleTimeEvent.h>

#include <typeinfo>

#include <prototyping/constrained_trajectory.h>
#include <models/center/cell_polarity/cell_polarity.h>

#include <metrics.h>

struct TestParticleType : MxParticleType {

    TestParticleType() : MxParticleType(true) {
        radius = 0.5;
        // ::strncpy(name, "TestParticle", MxParticleType::MAX_NAME);

        registerType();
    }

};

int simpleStartupTest() {
    MxLogger::enableConsoleLogging(LOG_DEBUG);

    std::vector<std::string> args;
    args.push_back("Test Sim");
    args.push_back("bc=no_slip");
    args.push_back("dim=10,10,10");
    args.push_back("cutoff=5");

    MxSimulator_init(args);

    TestParticleType *TestParticle = new TestParticleType();
    TestParticle = (TestParticleType*)TestParticle->get();

    double *potMin, *potMax;
    potMin = new double(0.0);
    potMax = new double(100.0*TestParticle->radius);
    MxPotential *pot = MxPotential::harmonic(1000.0, 2*TestParticle->radius, potMin, potMax);
    MxBind::types(pot, TestParticle, TestParticle);

    MxForce *force = MxForce::random(1.0, 0.0);
    MxBind::force(force, TestParticle);

    auto off = MxVector3f(TestParticle->radius * 2.0, 0, 0);

    MxParticleHandle *cp0 = (*TestParticle)(new MxVector3f(MxUniverse::getCenter() + off), new MxVector3f(1.0, 0.0, 0.0));

    MxParticleHandle *cp1 = (*TestParticle)(new MxVector3f(MxUniverse::getCenter() - off), new MxVector3f(-1.0, 0.0, 0.0));

    auto sim = MxSimulator::get();

    auto result = sim->run(-1.0);

    std::cout << result << std::endl;

    return result;
}


struct CellParticleType : MxParticleType {
    CellParticleType() : MxParticleType(true) {
        radius = 0.5;
        dynamics = PARTICLE_OVERDAMPED;

        registerType();
    }
};

struct ConstraintParticleType : MxParticleType {
    ConstraintParticleType() : MxParticleType(true) {
        radius = 0.1;
        setFrozen(true);

        registerType();
    }
};


int constrainedTrajectoryTest() {
    float dim = 20.0;
    int constraintPts = 5;
    float constraintPer = 1.0;
    float dist = 3.0;

    MxSimulator_Config conf;
    conf.setTitle("Test sim");
    conf.universeConfig.dim = {20.0, 20.0, 20.0};
    MxSimulator_initC(conf);

    NOMStyle *blueStyle = new NOMStyle("blue");
    NOMStyle *redStyle = new NOMStyle("red");

    CellParticleType *CellParticle = new CellParticleType();
    ConstraintParticleType *ConstraintParticle = new ConstraintParticleType();

    CellParticle = (CellParticleType*)CellParticle->get();
    ConstraintParticle = (ConstraintParticleType*)ConstraintParticle->get();

    double *mu, *kc, *r0, *potMin, *potMax;
    mu = new double(10.0);
    kc = new double(10.0);
    r0 = new double(2.0*CellParticle->radius);
    potMin = new double(2.0*CellParticle->radius*0.001);
    potMax = new double(2.0*CellParticle->radius*3.0);
    MxPotential *pot = MxPotential::overlapping_sphere(mu, kc, NULL, r0, potMin, potMax);
    MxBind::types(pot, CellParticle, CellParticle);

    // Constraints here!
    float yMid = dim / 2.0;
    float zMid = dim / 2.0;
    float xIni = 5 * CellParticle->radius;
    float xFin = dim - xIni;

    MxVector3f *pos0, *pos1;
    float x = xIni;
    float yFlag = 1.0;
    float thisTime = 0.0;
    MxParticleHandle *cp0, *cp1, *constraintp;

    pos0 = new MxVector3f(xIni, yMid + dist, zMid);
    pos1 = new MxVector3f(xIni, yMid - dist, zMid);
    cp0 = (*CellParticle)(pos0);
    cp1 = (*CellParticle)(pos1);
    cp0->setStyle(blueStyle);
    cp1->setStyle(redStyle);

    while(x <= xFin) {
        pos0 = new MxVector3f(x, yMid + yFlag * dist, zMid);
        pos1 = new MxVector3f(x, yMid - yFlag * dist, zMid);
        x += dist;
        yFlag *= -1.0;

        // MxConstrainedTrajectory::linearRadialConstraint(cp0, thisTime, *pos0, 1.0);
        // MxConstrainedTrajectory::linearRadialConstraint(cp1, thisTime, *pos1, 1.0);
        MxConstrainedTrajectory::hermiteConstraint(cp0, thisTime, *pos0);
        MxConstrainedTrajectory::hermiteConstraint(cp1, thisTime, *pos1);

        constraintp = (*ConstraintParticle)(pos0);
        constraintp->setStyle(blueStyle);
        constraintp = (*ConstraintParticle)(pos1);
        constraintp->setStyle(redStyle);

        thisTime += constraintPer;

        delete pos0, pos1;
    }
    float finTime = thisTime - constraintPer;

    auto *constraint0 = MxConstrainedTrajectory::getConstraint(cp0);
    auto *constraint1 = MxConstrainedTrajectory::getConstraint(cp1);

    Log(LOG_INFORMATION) << constraint0->str().c_str();
    Log(LOG_INFORMATION) << constraint1->str().c_str();

    auto sim = MxSimulator::get();

    auto result = sim->run(finTime);

    std::cout << result << std::endl;

    return result;
}

float cellRadius = 0.25;

struct TPA : MxParticleType {
    TPA() : MxParticleType(true) {
        radius = cellRadius;
        style = new NOMStyle();
        style->setColor("blue");
        setFrozen(true);

        registerType();
    }
};

struct TPB : MxParticleType {
    TPB() : MxParticleType(true) {
        radius = cellRadius;
        style = new NOMStyle();
        style->setColor("green");
        setFrozen(true);

        registerType();
    }
};

struct TPC : MxParticleType {
    TPC() : MxParticleType(true) {
        radius = cellRadius;
        style = new NOMStyle();
        style->setColor("white");
        setFrozen(true);

        registerType();
    }
};

struct TPD : MxParticleType {
    TPD() : MxParticleType(true) {
        radius = cellRadius;
        style = new NOMStyle();
        style->setColor("orange");
        setFrozen(true);

        registerType();
    }
};

static inline MxMatrix3f tensorProduct(const MxVector3f &rowVec, const MxVector3f &colVec) {
    MxMatrix3f result(1.0);
    for(int i = 0; i < 3; ++i)
        for(int j = 0; j < 3; ++j)
            result[j][i] *= rowVec[i] * colVec[j];
    return result;
}

// int testing() {
//     MxSimulator_Config conf;
//     conf.universeConfig.dim = {10., 10., 10.};
//     MxSimulator_initC(conf);

//     TPA *tpa = new TPA();
//     TPB *tpb = new TPB();
//     TPC *tpc = new TPC();
//     TPD *tpd = new TPD();

//     tpa = (TPA*)tpa->get();
//     tpb = (TPB*)tpb->get();
//     tpc = (TPC*)tpc->get();
//     tpd = (TPD*)tpd->get();

//     MxVector3f pt0, pt1, ptinc, vi;
//     pt0 = {5.0, 5.0, 5.0};
//     pt1 = pt0 + MxVector3f{2*cellRadius, 0, 0};
//     ptinc = {0, 2*cellRadius, 0};
//     vi = MxVector3f(0.0);

//     MxParticleHandle *p0, *p1;
//     MxParticleList pList;

//     p0 = (*tpa)(&pt0, &vi);
//     p1 = (*tpb)(&pt1, &vi);
//     pList.insert(p0->id);
//     pList.insert(p1->id);

//     pt0 += ptinc;
//     pt1 += ptinc;
//     p0 = (*tpa)(&pt0, &vi);
//     p1 = (*tpc)(&pt1, &vi);
//     std::cout << "poi id: " << p0->id << std::endl;
//     pList.insert(p0->id);
//     pList.insert(p1->id);
    
//     pt0 += ptinc;
//     pt1 += ptinc;
//     p0 = (*tpd)(&pt0, &vi);
//     p1 = (*tpc)(&pt1, &vi);
//     // pList.insert(p0->id);
//     pList.insert(p1->id);

//     const float contactDistance = 2.2*cellRadius;
//     const float polarityMag = 0.;
//     const float polarityRate = 0.;
//     const float couplingFlat = 1.;
//     const float couplingOrtho = 0.;
//     const float couplingLateral = 0.;
//     const float distanceCoeff = 10. * cellRadius;
//     std::string contactType = "anisotropic";
//     const float bendingCoeff = 0.5;

//     MxCellPolarity_registerType(tpa);
//     MxCellPolarity_registerType(tpb);
//     MxCellPolarity_registerType(tpc);

//     // auto *faa = MxCellPolarity_createForce_contact(tpa, tpa, polarityMag, polarityRate, distanceCoeff, couplingFlat, couplingOrtho, couplingLateral, contactType, bendingCoeff);
//     // auto *fab = MxCellPolarity_createForce_contact(tpa, tpb, polarityMag, polarityRate, distanceCoeff, couplingFlat, couplingOrtho, couplingLateral, contactType, bendingCoeff);
//     // auto *fac = MxCellPolarity_createForce_contact(tpa, tpc, polarityMag, polarityRate, distanceCoeff, couplingFlat, couplingOrtho, couplingLateral, contactType, bendingCoeff);

//     auto paa = potential_create_cellpolarity(contactDistance, polarityMag, polarityRate, distanceCoeff, couplingFlat, couplingOrtho, couplingLateral, contactType, bendingCoeff);
//     MxBind::types(paa, tpa, tpa);
//     MxBind::types(paa, tpa, tpb);
//     MxBind::types(paa, tpa, tpc);
//     MxBind::types(paa, tpb, tpb);
//     MxBind::types(paa, tpb, tpc);
//     MxBind::types(paa, tpc, tpc);

//     MxVector3f pvecAB(1, 0, 0);
//     MxVector3f pvecPCP(0, 1, 0);
    
//     for(int i = 0; i < pList.nr_parts; ++i) {
//         auto pId = pList.item(i)->id;
//         MxCellPolarity_SetVectorAB(pId, pvecAB);
//         MxCellPolarity_SetVectorAB(pId, pvecAB, false);
//         MxCellPolarity_SetVectorPCP(pId, pvecPCP);
//         MxCellPolarity_SetVectorPCP(pId, pvecPCP, false);
//     }

//     MxCellPolarity_load();

//     MxSimulator *sim = MxSimulator::get();
//     // sim->show();

//     MxLogger::enableConsoleLogging(LOG_DEBUG);
//     MxUniverse *u = getUniverse();
//     sim->run(2.0 * u->getDt());

//     return 0;
// }

int main (int argc, char** argv) {
    // return simpleStartupTest();
    return constrainedTrajectoryTest();
    // return testing();
}
