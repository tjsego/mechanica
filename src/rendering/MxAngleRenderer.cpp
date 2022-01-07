/**
 * @file MxAngleRenderer.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines renderer for visualizing MxAngles. 
 * @date 2021-09-13
 * 
 */

#include "MxAngleRenderer.h"

#include <angle.h>
#include <engine.h>
#include "MxUniverseRenderer.h"
#include <MxLogger.h>


HRESULT MxAngleRenderer::start(const std::vector<MxVector4f> &clipPlanes) {
    // create the shader
    _shader = Magnum::Shaders::Flat3D{Magnum::Shaders::Flat3D::Flag::VertexColor};
    
    // create the buffer
    _buffer = Magnum::GL::Buffer{};

    // create the mesh
    _mesh = Magnum::GL::Mesh{};
    _mesh.setPrimitive(Magnum::MeshPrimitive::Lines);
    _mesh.addVertexBuffer(_buffer, 0,
                          Shaders::Flat3D::Position{}, 
                          Shaders::Flat3D::Color3{});

    return S_OK;
}

static inline int render_angle(BondsInstanceData* angleData, int i, MxAngle *angle) {

    if(!(angle->flags & ANGLE_ACTIVE)) 
        return 0;

    Magnum::Vector3 *color = &angle->style->color;
    MxParticle *pi = _Engine.s.partlist[angle->i];
    MxParticle *pj = _Engine.s.partlist[angle->j];
    MxParticle *pk = _Engine.s.partlist[angle->k];
    
    double *oj = _Engine.s.celllist[pj->id]->origin;
    Magnum::Vector3 pj_origin = {static_cast<float>(oj[0]), static_cast<float>(oj[1]), static_cast<float>(oj[2])};
    
    int shiftij[3], shiftkj[3];
    Magnum::Vector3 pixij, pixkj;
    
    int *loci = _Engine.s.celllist[angle->i]->loc;
    int *locj = _Engine.s.celllist[angle->j]->loc;
    int *lock = _Engine.s.celllist[angle->k]->loc;
    
    for ( int k = 0 ; k < 3 ; k++ ) {
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
    Magnum::Vector3 posj = pj->position + pj_origin;
    Magnum::Vector3 posk = pixkj + pj_origin;
    
    angleData[i].position = posi;
    angleData[i].color = *color;
    angleData[i+1].position = posj;
    angleData[i+1].color = *color;
    
    angleData[i+2].position = posk;
    angleData[i+2].color = *color;
    angleData[i+3] = angleData[i+1];

    angleData[i+4].position = 0.5 * (posi + posj);
    angleData[i+4].color = *color;
    angleData[i+5].position = 0.5 * (posk + posj);
    angleData[i+5].color = *color;
    return 6;
}

HRESULT MxAngleRenderer::draw(Magnum::Mechanica::ArcBallCamera *camera, const MxVector2i &viewportSize, const MxMatrix4f &modelViewMat) {
    Log(LOG_DEBUG) << "";

    if(_Engine.nr_angles > 0) {
        int vertexCount = _Engine.nr_angles * 6;
        _mesh.setCount(vertexCount);
        
        _buffer.setData(
            {NULL, vertexCount * sizeof(BondsInstanceData)},
            GL::BufferUsage::DynamicDraw
        );
        
        BondsInstanceData* angleData = (BondsInstanceData*)(void*)_buffer.map(
           0,
           vertexCount * sizeof(BondsInstanceData),
           GL::Buffer::MapFlag::Write|GL::Buffer::MapFlag::InvalidateBuffer
        );
        
        int i = 0;
        for(int j = 0; j < _Engine.nr_angles; ++j) {
            MxAngle *angle = &_Engine.angles[j];
            i += render_angle(angleData, i, angle);
        }
        assert(i == vertexCount);
        _buffer.unmap();
        
        _shader
        .setTransformationProjectionMatrix(camera->projectionMatrix() * camera->cameraMatrix() * modelViewMat)
        .draw(_mesh);
    }
    return S_OK;
}

void MxAngleRenderer::setClipPlaneEquation(unsigned id, const Magnum::Vector4& pe) {}
