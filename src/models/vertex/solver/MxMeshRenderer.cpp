/**
 * @file MxMeshRenderer.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines the Mechanica mesh renderer
 * @date 2022-05-11
 * 
 */

#include "MxMeshRenderer.h"

#include "MxMeshSurface.h"
#include "MxMeshVertex.h"
#include "MxMeshSolver.h"

#include <rendering/MxUniverseRenderer.h>
#include <rendering/MxStyle.hpp>
#include <engine.h>
#include <MxLogger.h>
#include <MxSystem.h>

#include <Magnum/Mesh.h>

static MxMeshRenderer *_meshRenderer = NULL;


struct MeshFaceInstanceData {
    Magnum::Vector3 position;
    Magnum::Color3 color;
};

struct MeshEdgeInstanceData {
    Magnum::Vector3 position;
    Magnum::Color3 color;
};


static inline unsigned int render_meshedge(MeshEdgeInstanceData *edgeData, unsigned int i, MxMeshVertex *v0, MxMeshVertex *v1) {
    Magnum::Vector3 color = {0.f, 0.f, 0.f};
    MxParticle *pi = v0->particle()->part();
    MxParticle *pj = v1->particle()->part();
    
    double *oj = _Engine.s.celllist[pj->id]->origin;
    Magnum::Vector3 pj_origin = {static_cast<float>(oj[0]), static_cast<float>(oj[1]), static_cast<float>(oj[2])};
    
    int shift[3];
    Magnum::Vector3 pix;
    
    int *loci = _Engine.s.celllist[ pi->id ]->loc;
    int *locj = _Engine.s.celllist[ pj->id ]->loc;
    
    for ( int k = 0 ; k < 3 ; k++ ) {
        shift[k] = loci[k] - locj[k];
        if ( shift[k] > 1 )
            shift[k] = -1;
        else if ( shift[k] < -1 )
            shift[k] = 1;
        pix[k] = pi->x[k] + _Engine.s.h[k]* shift[k];
    }
                    
    edgeData[i].position = pix + pj_origin;
    edgeData[i].color = color;
    edgeData[i+1].position = pj->position + pj_origin;
    edgeData[i+1].color = color;
    return 2;
}

static inline HRESULT render_meshFacesEdges(MeshFaceInstanceData *faceData, 
                                            MeshEdgeInstanceData *edgeData, 
                                            const unsigned int &iFace, 
                                            const unsigned int &iEdge, 
                                            MxMeshSurface *s, 
                                            unsigned int &faceIncr, 
                                            unsigned int &edgeIncr) 
{
    Magnum::Vector3 color;
    
    MxStyle *style = s->style ? s->style : s->type()->style;
    if(!style) 
        color = {0.2f, 1.f, 1.f};
    else 
        color = style->color;

    faceIncr = 0;
    edgeIncr = 0;

    Magnum::Vector3 centroid = s->getCentroid();

    // cell id of centroid
    int cid, locj[3];

    if((cid = space_get_cellids_for_pos(&_Engine.s, centroid.data(), locj)) < 0) {
        Log(LOG_ERROR);
        return E_FAIL;
    }

    double *oj = _Engine.s.cells[cid].origin;
    Magnum::Vector3 pj_origin = {static_cast<float>(oj[0]), static_cast<float>(oj[1]), static_cast<float>(oj[2])};
    
    int shiftij[3], shiftkj[3];
    Magnum::Vector3 pixij, pixkj;

    std::vector<MxMeshVertex*> vertices = mx::models::vertex::vectorToDerived<MxMeshVertex>(s->parents());

    unsigned int j, k;
    for(j = 0; j < vertices.size(); j++) {
        MxMeshVertex *vi = vertices[j];
        MxMeshVertex *vk = vertices[j == vertices.size() - 1 ? 0 : j + 1];

        edgeIncr += render_meshedge(edgeData, iEdge + edgeIncr, vi, vk);

        MxParticle *pi = vi->particle()->part();
        MxParticle *pk = vk->particle()->part();

        int *loci = _Engine.s.celllist[pi->id]->loc;
        int *lock = _Engine.s.celllist[pk->id]->loc;

        for(k = 0; k < 3; k++) {
            int locjk = locj[k];
            shiftij[k] = loci[k] - locjk;
            shiftkj[k] = lock[k] - locjk;
            
            if(shiftij[k] > 1) shiftij[k] = -1;
            else if (shiftij[k] < -1) shiftij[k] = 1;
            
            if(shiftkj[k] > 1) shiftkj[k] = -1;
            else if (shiftkj[k] < -1) shiftkj[k] = 1;

            double h = _Engine.s.h[k];
            pixij[k] = pi->x[k] + h * shiftij[k];
            pixkj[k] = pk->x[k] + h * shiftkj[k];
        }

        Magnum::Vector3 posi = pixij + pj_origin;
        Magnum::Vector3 posk = pixkj + pj_origin;

        k = iFace + faceIncr;
        
        faceData[k].position = posi;
        faceData[k].color = color;
        faceData[k+1].position = centroid;
        faceData[k+1].color = color;
        faceData[k+2].position = posk;
        faceData[k+2].color = color;

        faceIncr += 3;
    }

    return S_OK;
}

MxMeshRenderer *MxMeshRenderer::get() {
    if(_meshRenderer == NULL) {
        MxUniverseRenderer *urenderer = MxSystem::getRenderer();
        if(urenderer) {
            _meshRenderer = new MxMeshRenderer();
            if(urenderer->registerSubRenderer(_meshRenderer) != S_OK) {
                delete _meshRenderer;
                _meshRenderer = NULL;
            }
        }
    }
    return _meshRenderer;
}

HRESULT MxMeshRenderer::start(const std::vector<MxVector4f> &clipPlanes) {
    // create the shaders
    _shaderFaces = Magnum::Shaders::MxFlat3D {
        Magnum::Shaders::MxFlat3D::Flag::VertexColor, 
        (unsigned int)MxUniverseRenderer::maxClipPlaneCount()
    };
    _shaderEdges = Magnum::Shaders::MxFlat3D{
        Magnum::Shaders::MxFlat3D::Flag::VertexColor, 
        (unsigned int)MxUniverseRenderer::maxClipPlaneCount()
    };
    
    // create the buffers
    _bufferFaces = Magnum::GL::Buffer();
    _bufferEdges = Magnum::GL::Buffer();

    // create the meshes
    _meshFaces = Magnum::GL::Mesh{};
    _meshFaces.setPrimitive(Magnum::MeshPrimitive::Triangles);
    _meshFaces.addVertexBuffer(_bufferFaces, 0, 
                               Shaders::MxPhong::Position{}, 
                               Shaders::MxPhong::Color3{});

    _meshEdges = Magnum::GL::Mesh{};
    _meshEdges.setPrimitive(Magnum::MeshPrimitive::Lines);
    _meshEdges.addVertexBuffer(_bufferEdges, 0,
                               Shaders::MxFlat3D::Position{}, 
                               Shaders::MxFlat3D::Color3{});

    return S_OK;
}

HRESULT MxMeshRenderer::draw(Magnum::Mechanica::ArcBallCamera *camera, const MxVector2i &viewportSize, const MxMatrix4f &modelViewMat) {
    
    MxMeshSolver *solver = MxMeshSolver::get();
    if(!solver) {
        Log(LOG_ERROR);
        return E_FAIL;
    }
    else if(solver->update() != S_OK) {
        Log(LOG_ERROR);
        return E_FAIL;
    }

    unsigned int vertexCountF = 3 * solver->_surfaceVertices;
    unsigned int vertexCountE = 2 * solver->_surfaceVertices;
    _meshFaces.setCount(vertexCountF);
    _meshEdges.setCount(vertexCountE);
    
    _bufferFaces.setData(
        {NULL, vertexCountF * sizeof(MeshFaceInstanceData)},
        GL::BufferUsage::DynamicDraw
    );
    _bufferEdges.setData(
        {NULL, vertexCountE * sizeof(MeshEdgeInstanceData)},
        GL::BufferUsage::DynamicDraw
    );
    
    // get pointer to data
    MeshFaceInstanceData* faceData = (MeshFaceInstanceData*)(void*)_bufferFaces.map(
        0,
        vertexCountF * sizeof(MeshFaceInstanceData),
        GL::Buffer::MapFlag::Write|GL::Buffer::MapFlag::InvalidateBuffer
    );
    MeshEdgeInstanceData* edgeData = (MeshEdgeInstanceData*)(void*)_bufferEdges.map(
        0,
        vertexCountE * sizeof(MeshEdgeInstanceData),
        GL::Buffer::MapFlag::Write|GL::Buffer::MapFlag::InvalidateBuffer
    );
    
    unsigned int iFaces = 0, iEdges = 0, faceIncr, edgeIncr;
    for(auto &m : solver->meshes) {
        for(auto &s : m->surfaces) {
            if(s) {
                render_meshFacesEdges(faceData, edgeData, iFaces, iEdges, s, faceIncr, edgeIncr);
                iFaces += faceIncr;
                iEdges += edgeIncr;
            }
        }
    }

    assert(iFaces == vertexCountF);
    assert(iEdges == vertexCountE);
    _bufferFaces.unmap();
    _bufferEdges.unmap();
    
    _shaderFaces
        .setTransformationProjectionMatrix(camera->projectionMatrix() * camera->cameraMatrix() * modelViewMat)
        .draw(_meshFaces);
    _shaderEdges
        .setTransformationProjectionMatrix(camera->projectionMatrix() * camera->cameraMatrix() * modelViewMat)
        .draw(_meshEdges);

    return S_OK;
}
