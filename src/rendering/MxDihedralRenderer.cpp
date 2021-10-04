/**
 * @file MxDihedralRenderer.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines renderer for visualizing MxDihedrals. 
 * @date 2021-10-04
 * 
 */

#include "MxDihedralRenderer.h"

#include <dihedral.h>
#include <engine.h>
#include "MxUniverseRenderer.h"
#include <MxLogger.h>


HRESULT MxDihedralRenderer::start(const std::vector<MxVector4f> &clipPlanes) {
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

static inline int render_dihedral(BondsInstanceData* dihedralData, int i, MxDihedral *dihedral) {
    
    Magnum::Vector3 *color = &dihedral->style->color;
    MxParticle *pi = _Engine.s.partlist[dihedral->i];
    MxParticle *pj = _Engine.s.partlist[dihedral->j];
    MxParticle *pk = _Engine.s.partlist[dihedral->k];
    MxParticle *pl = _Engine.s.partlist[dihedral->l];
    
    double *oj = _Engine.s.celllist[pj->id]->origin;
    double *ok = _Engine.s.celllist[pk->id]->origin;
    Magnum::Vector3 pj_origin = {static_cast<float>(oj[0]), static_cast<float>(oj[1]), static_cast<float>(oj[2])};
    Magnum::Vector3 pk_origin = {static_cast<float>(ok[0]), static_cast<float>(ok[1]), static_cast<float>(ok[2])};

    int shiftik[3], shiftlj[3];
    Magnum::Vector3 pixik, pixlj;
    
    int *loci = _Engine.s.celllist[dihedral->i]->loc;
    int *locj = _Engine.s.celllist[dihedral->j]->loc;
    int *lock = _Engine.s.celllist[dihedral->k]->loc;
    int *locl = _Engine.s.celllist[dihedral->l]->loc;
    
    for ( int k = 0 ; k < 3 ; k++ ) {
        shiftik[k] = loci[k] - lock[k];
        shiftlj[k] = locl[k] - locj[k];
        
        if(shiftik[k] > 1) shiftik[k] = -1;
        else if (shiftik[k] < -1) shiftik[k] = 1;
        
        if(shiftlj[k] > 1) shiftlj[k] = -1;
        else if (shiftlj[k] < -1) shiftlj[k] = 1;

        double h = _Engine.s.h[k];
        pixik[k] = pi->x[k] + h * shiftik[k];
        pixlj[k] = pl->x[k] + h * shiftlj[k];
    }

    Magnum::Vector3 posi = pixik + pk_origin;
    Magnum::Vector3 posj = pj->position + pj_origin;
    Magnum::Vector3 posk = pk->position + pk_origin;
    Magnum::Vector3 posl = pixlj + pj_origin;
    
    dihedralData[i].position = posi;
    dihedralData[i].color = *color;
    dihedralData[i+1].position = posk;
    dihedralData[i+1].color = *color;
    dihedralData[i+2].position = posj;
    dihedralData[i+2].color = *color;
    dihedralData[i+3].position = posl;
    dihedralData[i+3].color = *color;
    dihedralData[i+4].position = 0.5 * (posi + posk);
    dihedralData[i+4].color = *color;
    dihedralData[i+5].position = 0.5 * (posl + posj);
    dihedralData[i+5].color = *color;
    return 6;
}

HRESULT MxDihedralRenderer::draw(Magnum::Mechanica::ArcBallCamera *camera, const MxVector2i &viewportSize, const MxMatrix4f &modelViewMat) {
    Log(LOG_DEBUG) << "";

    if(_Engine.nr_dihedrals > 0) {
        int vertexCount = _Engine.nr_dihedrals * 6;
        _mesh.setCount(vertexCount);
        
        _buffer.setData(
            {NULL, vertexCount * sizeof(BondsInstanceData)},
            GL::BufferUsage::DynamicDraw
        );
        
        BondsInstanceData* dihedralData = (BondsInstanceData*)(void*)_buffer.map(
           0,
           vertexCount * sizeof(BondsInstanceData),
           GL::Buffer::MapFlag::Write|GL::Buffer::MapFlag::InvalidateBuffer
        );
        
        int i = 0;
        for(int j = 0; j < _Engine.nr_dihedrals; ++j) {
            MxDihedral *dihedral = &_Engine.dihedrals[j];
            i += render_dihedral(dihedralData, i, dihedral);
        }
        assert(i == vertexCount);
        _buffer.unmap();
        
        _shader
        .setTransformationProjectionMatrix(camera->projectionMatrix() * camera->cameraMatrix() * modelViewMat)
        .draw(_mesh);
    }
    return S_OK;
}

void MxDihedralRenderer::setClipPlaneEquation(unsigned id, const Magnum::Vector4& pe) {}
