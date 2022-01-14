/**
 * @file MxBondRenderer.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines renderer for visualizing MxBonds. 
 * @date 2022-01-14
 * 
 */

#include "MxBondRenderer.h"

#include <fptype.h>
#include <bond.h>
#include <engine.h>
#include "MxUniverseRenderer.h"
#include <MxLogger.h>


HRESULT MxBondRenderer::start(const std::vector<MxVector4f> &clipPlanes) {
    
    // create the shader
    _shader = Magnum::Shaders::MxFlat3D{
        Magnum::Shaders::MxFlat3D::Flag::VertexColor, 
        (unsigned int)MxUniverseRenderer::maxClipPlaneCount()
    };
    
    // create the buffer
    _buffer = Magnum::GL::Buffer{};

    // create the mesh
    _mesh = Magnum::GL::Mesh{};
    _mesh.setPrimitive(Magnum::MeshPrimitive::Lines);
    _mesh.addVertexBuffer(_buffer, 0,
                          Shaders::MxFlat3D::Position{}, 
                          Shaders::MxFlat3D::Color3{});

    return S_OK;
}

static inline int render_bond(BondsInstanceData* bondData, int i, MxBond *bond) {

    if(!(bond->flags & BOND_ACTIVE)) 
        return 0;

    Magnum::Vector3 *color = &bond->style->color;
    MxParticle *pi = _Engine.s.partlist[bond->i];
    MxParticle *pj = _Engine.s.partlist[bond->j];
    
    double *oj = _Engine.s.celllist[pj->id]->origin;
    Magnum::Vector3 pj_origin = {static_cast<float>(oj[0]), static_cast<float>(oj[1]), static_cast<float>(oj[2])};
    
    int shift[3];
    Magnum::Vector3 pix;
    
    int *loci = _Engine.s.celllist[ bond->i ]->loc;
    int *locj = _Engine.s.celllist[ bond->j ]->loc;
    
    for ( int k = 0 ; k < 3 ; k++ ) {
        shift[k] = loci[k] - locj[k];
        if ( shift[k] > 1 )
            shift[k] = -1;
        else if ( shift[k] < -1 )
            shift[k] = 1;
        pix[k] = pi->x[k] + _Engine.s.h[k]* shift[k];
    }
                    
    bondData[i].position = pix + pj_origin;
    bondData[i].color = *color;
    bondData[i+1].position = pj->position + pj_origin;
    bondData[i+1].color = *color;
    return 2;
}

HRESULT MxBondRenderer::draw(Magnum::Mechanica::ArcBallCamera *camera, const MxVector2i &viewportSize, const MxMatrix4f &modelViewMat) {
    Log(LOG_DEBUG) << "";

    if(_Engine.nr_active_bonds > 0) {
        int vertexCount = _Engine.nr_active_bonds * 2;
        _mesh.setCount(vertexCount);
        
        _buffer.setData(
            {NULL, vertexCount * sizeof(BondsInstanceData)},
            GL::BufferUsage::DynamicDraw
        );
        
        // get pointer to data, give me the damned bytes
        BondsInstanceData* bondData = (BondsInstanceData*)(void*)_buffer.map(
           0,
           vertexCount * sizeof(BondsInstanceData),
           GL::Buffer::MapFlag::Write|GL::Buffer::MapFlag::InvalidateBuffer
        );
        
        int i = 0;
        Magnum::Vector3 *color;
        for(int j = 0; j < _Engine.nr_bonds; ++j) {
            MxBond *bond = &_Engine.bonds[j];
            i += render_bond(bondData, i, bond);
        }
        assert(i == 2 * _Engine.nr_active_bonds);
        _buffer.unmap();
        
        _shader
        .setTransformationProjectionMatrix(camera->projectionMatrix() * camera->cameraMatrix() * modelViewMat)
        .draw(_mesh);
    }
    return S_OK;
}

const unsigned MxBondRenderer::addClipPlaneEquation(const Magnum::Vector4& pe) {
    unsigned int id = _clipPlanes.size();
    _clipPlanes.push_back(pe);
    _shader.setclipPlaneEquation(id, pe);
    return id;
}

const unsigned MxBondRenderer::removeClipPlaneEquation(const unsigned int &id) {
    _clipPlanes.erase(_clipPlanes.begin() + id);

    for(unsigned int i = id; i < _clipPlanes.size(); i++) {
        _shader.setclipPlaneEquation(i, _clipPlanes[i]);
    }

    return _clipPlanes.size();
}

void MxBondRenderer::setClipPlaneEquation(unsigned id, const Magnum::Vector4& pe) {
    if(id > _shader.clipPlaneCount()) {
        mx_exp(std::invalid_argument("invalid id for clip plane"));
    }

    _shader.setclipPlaneEquation(id, pe);
    _clipPlanes[id] = pe;
}
