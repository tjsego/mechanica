/*
 * CylinderModel.cpp
 *
 * Created on: Sep 20, 2018
 *      Author: andy
 */

#include "MxCylinderModel.h"
#include <MxDebug.h>
#include <iostream>
#include "MeshIO.h"
#include "MeshOperations.h"
#include "MxCellVolumeConstraint.h"
#include "MxPolygonAreaConstraint.h"
#include <MxPolygonSurfaceTensionForce.h>
#include <mx_error.h>

MxPolygonType basicPolygonType{"BasicPolygon"};
MxPolygonType growingPolygonType{"GrowingPolygon"};

MxCellVolumeConstraint cellVolumeConstraint{0., 0.};
MxPolygonAreaConstraint areaConstraint{0.1, 0.01};

MxPolygonSurfaceTensionForce stdPolygonForce{0.05};
MxPolygonSurfaceTensionForce growingPolygonForce{0.05};

static struct CylinderCellType : MxCellType
{
    virtual Magnum::Color4 color(struct MxCell *cell) {
        //return Color4{1.0f, 0.0f, 0.0f, 0.08f};
        return Color4::green();
    }

    virtual ~CylinderCellType() {};

    CylinderCellType() : MxCellType{"CylinderCell"} {};
} cylinderCellType;

static struct MeshObjectTypeHandler : IMeshObjectTypeHandler {
    virtual MxCellType *cellType(const char* cellName, int cellIndex) {
        return &cylinderCellType;
    }

    virtual MxPolygonType *polygonType(int polygonIndex) {
        return &basicPolygonType;
    }

    virtual MxPartialPolygonType *partialPolygonType(const MxCellType *cellType, const MxPolygonType *polyType) {
        return nullptr;
    }

    virtual ~MeshObjectTypeHandler() {};

} meshObjectTypeHandler;




MxCylinderModel::MxCylinderModel()  {
    growingPolygonType.centerColor = Magnum::Color4::red();
}


HRESULT MxCylinderModel::loadModel(const char* fileName) {
    loadAssImpModel(fileName);

    for(int i = 0; i < mesh->cells.size(); ++i) {
        CellPtr cell = mesh->cells[i];
        std::cout << "cell[" << i << "], id:" << cell->id << ", center: " << cell->centroid << std::endl;
    }

    testEdges();

    VERIFY(propagator->structureChanged());

    return S_OK;
}


void MxCylinderModel::loadAssImpModel(const char* fileName) {

    std::cout << MX_FUNCTION << ", fileName: " << fileName << std::endl;

    mesh = MxMesh_FromFile(fileName, 1.0, &meshObjectTypeHandler);

    cellVolumeConstraint.targetVolume = mesh->cells[1]->volume;
    cellVolumeConstraint.lambda = 0.5;

    propagator->bindConstraint(&cellVolumeConstraint, &cylinderCellType);

    propagator->bindForce(&stdPolygonForce, &basicPolygonType);

    propagator->bindForce(&growingPolygonForce, &growingPolygonType);

    mesh->selectObject(MxMesh_TYPESEL::MxMesh_TYPEPOLYGON, 367);

    CellPtr cell = mesh->cells[1];

    setTargetVolume(cell->volume);
    setTargetVolumeLambda(0.01);

    //mesh->setShortCutoff(0);
    //mesh->setLongCutoff(0.3);
}

void MxCylinderModel::testEdges() {
    return;
}

HRESULT MxCylinderModel::getStateVector(float *stateVector, uint32_t *count)
{
    *count = 0;
    return S_OK;
}

HRESULT MxCylinderModel::setStateVector(const float *stateVector)
{
    return S_OK;
}


HRESULT MxCylinderModel::getStateVectorRate(float time, const float *y, float* dydt)
{
    return S_OK;
}

void MxCylinderModel::setTargetVolume(float tv)
{
    cellVolumeConstraint.targetVolume = tv;
}

HRESULT MxCylinderModel::applyT1Edge2TransitionToSelectedEdge() {
    if(!mesh->selectedEdge()) return mx_error(E_FAIL, "no selected object, or selected object is not an edge");

    return Mx_FlipEdge(mesh, mesh->selectedObject<MxEdge>());
}

HRESULT MxCylinderModel::applyT2PolygonTransitionToSelectedPolygon()
{
    if(!mesh->selectedPolygon()) return mx_error(E_FAIL, "no selected object, or selected object is not a polygon");
    
    HRESULT result = Mx_CollapsePolygon(mesh, mesh->selectedObject<MxPolygon>());

    if(SUCCEEDED(result)) {

    }

    return result;
}

HRESULT MxCylinderModel::applyT3PolygonTransitionToSelectedPolygon() {
    if(!mesh->selectedPolygon()) return mx_error(E_FAIL, "no selected object, or selected object is not a polygon");

    MxPolygon *poly = mesh->selectedObject<MxPolygon>();

    // make an cut plane perpendicular to the zeroth vertex
    MxVector3f normal = poly->vertices[0]->position - poly->centroid;

    MxPolygon *p1, *p2;

    HRESULT result = Mx_SplitPolygonBisectPlane(mesh, poly, &normal, &p1, &p2);

    if(SUCCEEDED(result)) {

    }
    
    VERIFY(propagator->structureChanged());

    return result;
}

float MxCylinderModel::minTargetVolume()
{
    return 0.1 * cellVolumeConstraint.targetVolume;
}

float MxCylinderModel::maxTargetVolume()
{
    return 3 * cellVolumeConstraint.targetVolume;
}

float MxCylinderModel::targetVolume()
{
    return cellVolumeConstraint.targetVolume;
}

float MxCylinderModel::targetVolumeLambda()
{
    return cellVolumeConstraint.lambda;
}

void MxCylinderModel::setTargetVolumeLambda(float targetVolumeLambda)
{
    cellVolumeConstraint.lambda = targetVolumeLambda;
}

float MxCylinderModel::minTargetArea()
{
    return 0.1 * areaConstraint.targetArea;
}

float MxCylinderModel::maxTargetArea()
{
    return 3 * areaConstraint.targetArea;
}

float MxCylinderModel::targetArea()
{
    return areaConstraint.targetArea;
}

float MxCylinderModel::targetAreaLambda()
{
    return areaConstraint.lambda;
}

void MxCylinderModel::setTargetArea(float targetArea)
{
    areaConstraint.targetArea = targetArea;
}

void MxCylinderModel::setTargetAreaLambda(float targetAreaLambda)
{
    areaConstraint.lambda = targetAreaLambda;
}

static float PolyDistance = 1;

HRESULT MxCylinderModel::changePolygonTypes()
{
    if(mesh->selectedPolygon()) {
        MxPolygon *poly = mesh->selectedObject<MxPolygon>();

        for(PolygonPtr p : mesh->polygons) {
            
            float distance = (poly->centroid - p->centroid).length();
            if(distance <= PolyDistance) {
                p->forceType = &growingPolygonType;
                p->constraintType = &growingPolygonType;
            }
        }
        VERIFY(propagator->structureChanged());
        return S_OK;
    }
    else {
        return E_FAIL;
    }
}

HRESULT MxCylinderModel::activateAreaConstraint()
{
    propagator->bindConstraint(&areaConstraint, &growingPolygonType);
    return propagator->structureChanged();
}

float MxCylinderModel::stdSurfaceTension()
{
    return stdPolygonForce.surfaceTension;
}

void MxCylinderModel::setStdSurfaceTension(float val)
{
    stdPolygonForce.surfaceTension = val;
}

float MxCylinderModel::stdSurfaceTensionMin()
{
    return 0;
}

float MxCylinderModel::stdSurfaceTensionMax()
{
    return stdPolygonForce.surfaceTension * 5;
}

float MxCylinderModel::growSurfaceTension()
{
    return growingPolygonForce.surfaceTension;
}

void MxCylinderModel::setGrowStdSurfaceTension(float val)
{
    growingPolygonForce.surfaceTension = val;
}

float MxCylinderModel::growSurfaceTensionMin()
{
    return 0;
}

float MxCylinderModel::growSurfaceTensionMax()
{
    return 5 * growingPolygonForce.surfaceTension;
}
