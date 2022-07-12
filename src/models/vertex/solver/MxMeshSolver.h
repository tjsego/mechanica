/**
 * @file MxMeshSolver.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines the Mechanica mesh solver
 * @date 2022-04-26
 * 
 */
#ifndef MODELS_VERTEX_SOLVER_MXMESHSOLVER_H_
#define MODELS_VERTEX_SOLVER_MXMESHSOLVER_H_

#include <mx_port.h>

#include "MxMesh.h"
#include "MxMeshLogger.h"

#include <MxSubEngine.h>

class MxMeshRenderer;

struct CAPI_EXPORT MxMeshSolver : MxSubEngine { 

    const char *name = "MxMeshSolver";

    std::vector<MxMesh*> meshes;

    static HRESULT init();
    static MxMeshSolver *get();
    HRESULT compact();

    bool isDirty();
    HRESULT setDirty(const bool &_isDirty);

    MxMesh *newMesh();
    HRESULT loadMesh(MxMesh *mesh);
    HRESULT unloadMesh(MxMesh *mesh);

    HRESULT registerType(MxMeshBodyType *_type);
    HRESULT registerType(MxMeshSurfaceType *_type);

    MxMeshStructureType *getStructureType(const unsigned int &typeId);
    MxMeshBodyType *getBodyType(const unsigned int &typeId);
    MxMeshSurfaceType *getSurfaceType(const unsigned int &typeId);

    HRESULT positionChanged();
    HRESULT update(const bool &_force=false);

    HRESULT preStepStart();
    HRESULT preStepJoin();
    HRESULT postStepStart();
    HRESULT postStepJoin();

    std::vector<MxMeshLogEvent> getLog() {
        return MxMeshLogger::events();
    }
    HRESULT log(MxMesh *mesh, const MxMeshLogEventType &type, const std::vector<int> &objIDs, const std::vector<MxMeshObj::Type> &objTypes);

    friend MxMeshRenderer;

private:

    float *_forces;
    unsigned int _bufferSize;
    unsigned int _surfaceVertices;
    unsigned int _totalVertices;
    bool _isDirty;

    std::vector<MxMeshStructureType*> _structureTypes;
    std::vector<MxMeshBodyType*> _bodyTypes;
    std::vector<MxMeshSurfaceType*> _surfaceTypes;
    
};

#endif // MODELS_VERTEX_SOLVER_MXMESHSOLVER_H_