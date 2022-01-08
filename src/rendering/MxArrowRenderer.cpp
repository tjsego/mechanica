/**
 * @file MxArrowRenderer.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines renderer for visualizing vectors. 
 * @date 2021-09-08
 * 
 */
#include "MxArrowRenderer.h"

#include <engine.h>
#include <MxLogger.h>
#include <MxSimulator.h>
#include <rendering/MxUniverseRenderer.h>

#include <Magnum/MeshTools/Compile.h>
#include <Magnum/MeshTools/Transform.h>
#include <Magnum/Primitives/Cone.h>
#include <Magnum/Primitives/Cylinder.h>
#include <Magnum/Shaders/Phong.h>
#include <Magnum/Trade/MeshData.h>

#include <limits>
#include <string>

struct MxArrowInstanceData {
    Magnum::Matrix4 transformationMatrix;
    Magnum::Matrix3 normalMatrix;
    Magnum::Color4 color;
};

// todo: make arrow geometry settable

static float _arrowHeadHeight = 0.5;
static float _arrowHeadRadius = 0.25;
static float _arrowCylHeight = 0.5;
static float _arrowCylRadius = 0.1;

/**
 * @brief Generates a 3x3 rotation matrix into the frame of a vector. 
 * 
 * The orientation of the second and third axes of the resulting transformation are arbitrary. 
 * 
 * @param vec Vector along which the first axis of the transformed frame is aligned. 
 * @return MxMatrix3f 
 */
static MxMatrix3f MxVectorFrameRotation(const MxVector3f &vec) {
    MxVector3f u1, u2, u3, p;

    float vec_len = vec.length();
    if(vec_len == 0.0) {
        mx_error(E_FAIL, "Cannot pass zero vector");
        return MxMatrix3f();
    }

    u2 = vec / vec_len;

    if(vec[0] != 0) {
        p[1] = 1.0;
        p[2] = 1.0;
        p[0] = - (vec[1] + vec[2]) / vec[0];
    }
    else if (vec[1] != 0) {
        p[0] = 1.0;
        p[2] = 1.0;
        p[1] = - (vec[0] + vec[2]) / vec[1];
    }
    else {
        p[0] = 1.0;
        p[1] = 1.0;
        p[2] = - (vec[0] + vec[1]) / vec[2];
    }
    u3 = p.normalized();

    u1 = Magnum::Math::cross(u2, u3);

    MxMatrix3f result(u1, u2, u3);

    return result;
}

static inline HRESULT render_arrow(MxArrowInstanceData *pDataArrow, MxArrowInstanceData *pDataCylinder, int idx, MxArrowData *p, const MxVector3f &origin) {
    MxVector3f position = p->position + origin;
    float vec_len = p->components.length();
    Magnum::Color4 color = {p->style.color[0], p->style.color[1], p->style.color[2], (float)p->style.getVisible()};

    float arrowHeadHeightProbably = _arrowHeadHeight;
    float arrowHeadHeight = std::min<float>(arrowHeadHeightProbably, vec_len);
    float arrowHeadScaling = arrowHeadHeight / arrowHeadHeightProbably;

    MxMatrix3f rotationMatrix = MxVectorFrameRotation(p->components);
    Magnum::Matrix4 transformBase = Magnum::Matrix4::from(rotationMatrix, position) * Magnum::Matrix4::scaling(MxVector3f(p->scale));

    Magnum::Matrix4 transformHead = transformBase * Magnum::Matrix4::translation(MxVector3f(0.0, vec_len - arrowHeadHeight, 0.0));
    transformHead = transformHead * Magnum::Matrix4::scaling(MxVector3f(_arrowHeadRadius, _arrowHeadRadius * arrowHeadScaling, _arrowHeadRadius));
    transformHead = transformHead * Magnum::Matrix4::translation(MxVector3f(0.0, arrowHeadHeightProbably / 2.0 / _arrowHeadRadius, 0.0));

    Magnum::Matrix4 transformCyl = transformBase * Magnum::Matrix4::scaling(MxVector3f(_arrowCylRadius, _arrowCylRadius * (vec_len - arrowHeadHeight) / _arrowCylHeight, _arrowCylRadius));
    transformCyl = transformCyl * Magnum::Matrix4::translation(MxVector3f(0.0, _arrowCylHeight / _arrowCylRadius * 0.5, 0.0));

    pDataArrow[idx].transformationMatrix = transformHead;
    pDataArrow[idx].normalMatrix = transformHead.normalMatrix();
    pDataArrow[idx].color = color;

    pDataCylinder[idx].transformationMatrix = transformCyl;
    pDataCylinder[idx].normalMatrix = transformCyl.normalMatrix();
    pDataCylinder[idx].color = color;

    return S_OK;
}

MxArrowRenderer::MxArrowRenderer() {
    this->nr_arrows = 0;
}

MxArrowRenderer::MxArrowRenderer(const MxArrowRenderer &other) {
    this->arrows = other.arrows;
    this->nr_arrows = other.nr_arrows;

    this->_arrowDetail = other._arrowDetail;
}

MxArrowRenderer::~MxArrowRenderer() {
    this->arrows.clear();
}

HRESULT MxArrowRenderer::start(const std::vector<MxVector4f> &clipPlanes) {

    // create the shaders
    unsigned int clipPlaneCount = clipPlanes.size();
    _shader = Magnum::Shaders::MxPhong {
        Magnum::Shaders::MxPhong::Flag::VertexColor | 
        Magnum::Shaders::MxPhong::Flag::InstancedTransformation, 
        1, 
        clipPlaneCount
    };
    _shader.setShininess(2000.0f)
        .setLightPositions({{-20, 40, 20, 0.f}})
        .setLightColors({Magnum::Color3{0.9, 0.9, 0.9}})
        .setShininess(100)
        .setAmbientColor({0.4, 0.4, 0.4, 1})
        .setDiffuseColor({1, 1, 1, 0})
        .setSpecularColor({0.2, 0.2, 0.2, 0});
    for(int i = 0; i < clipPlaneCount; ++i)
        _shader.setclipPlaneEquation(i, clipPlanes[i]);
    
    // create the buffers
    _bufferHead = Magnum::GL::Buffer();
    _bufferCylinder = Magnum::GL::Buffer();

    // create the meshes
    unsigned int numRingsHead = this->_arrowDetail;
    unsigned int numSegmentsHead = this->_arrowDetail;
    unsigned int numRingsCyl = this->_arrowDetail;
    unsigned int numSegmentsCyl = this->_arrowDetail;
    _meshHead = Magnum::MeshTools::compile(Magnum::Primitives::coneSolid(
        numRingsHead, 
        numSegmentsHead, 
        0.5 * _arrowHeadHeight / _arrowHeadRadius, 
        Magnum::Primitives::ConeFlag::CapEnd
    ));
    _meshCylinder = Magnum::MeshTools::compile(Magnum::Primitives::cylinderSolid(
        numRingsCyl, 
        numSegmentsCyl, 
        0.5 * _arrowCylHeight / _arrowCylRadius, 
        Magnum::Primitives::CylinderFlag::CapEnds
    ));

    _meshHead.addVertexBufferInstanced(_bufferHead, 1, 0, 
        Magnum::Shaders::Phong::TransformationMatrix{}, 
        Magnum::Shaders::Phong::NormalMatrix{}, 
        Magnum::Shaders::Phong::Color4{}
    );
    _meshCylinder.addVertexBufferInstanced(_bufferCylinder, 1, 0, 
        Magnum::Shaders::Phong::TransformationMatrix{}, 
        Magnum::Shaders::Phong::NormalMatrix{}, 
        Magnum::Shaders::Phong::Color4{}
    );
    _meshHead.setInstanceCount(0);
    _meshCylinder.setInstanceCount(0);

    return S_OK;
}

HRESULT MxArrowRenderer::draw(Magnum::Mechanica::ArcBallCamera *camera, const MxVector2i& viewportSize, const MxMatrix4f &modelViewMat) {
    Log(LOG_DEBUG) << "";

    _meshHead.setInstanceCount(this->nr_arrows);
    _meshCylinder.setInstanceCount(this->nr_arrows);

    _bufferHead.setData({NULL, this->nr_arrows * sizeof(MxArrowInstanceData)}, Magnum::GL::BufferUsage::DynamicDraw);
    _bufferCylinder.setData({NULL, this->nr_arrows * sizeof(MxArrowInstanceData)}, Magnum::GL::BufferUsage::DynamicDraw);

    MxArrowInstanceData *pArrowData = (MxArrowInstanceData*)(void*)_bufferHead.map(
        0, 
        this->nr_arrows * sizeof(MxArrowInstanceData), 
        Magnum::GL::Buffer::MapFlag::Write | Magnum::GL::Buffer::MapFlag::InvalidateBuffer
    );
    MxArrowInstanceData *pCylinderData = (MxArrowInstanceData*)(void*)_bufferCylinder.map(
        0, 
        this->nr_arrows * sizeof(MxArrowInstanceData), 
        Magnum::GL::Buffer::MapFlag::Write | Magnum::GL::Buffer::MapFlag::InvalidateBuffer
    );

    int i = 0;
    MxVector3f origin(_Engine.s.origin[0], _Engine.s.origin[1], _Engine.s.origin[2]);
    
    for (int aid = 0; aid < this->arrows.size(); aid++) {
        MxArrowData *ad = this->arrows[aid];
        if(ad != NULL) {
            render_arrow(pArrowData, pCylinderData, i, ad, origin);
            i++;
        }
    }

    assert(i == this->nr_arrows);

    _bufferHead.unmap();
    _bufferCylinder.unmap();

    _shader
        .setProjectionMatrix(camera->projectionMatrix())
        .setTransformationMatrix(camera->cameraMatrix() * modelViewMat)
        .setNormalMatrix(camera->viewMatrix().normalMatrix());
    _shader.draw(_meshHead);
    _shader.draw(_meshCylinder);

    return S_OK;
}

void MxArrowRenderer::setClipPlaneEquation(unsigned id, const Magnum::Vector4& pe) {
    if(id > _shader.clipPlaneCount()) mx_exp(std::invalid_argument("invalid id for clip plane"));

    _shader.setclipPlaneEquation(id, pe);
}

int MxArrowRenderer::nextDataId() {
    if(this->nr_arrows == this->arrows.size()) return this->nr_arrows;

    for(int i = 0; i < this->arrows.size(); ++i)
        if(this->arrows[i] == NULL)
            return i;

    mx_error(E_FAIL, "Could not identify new arrow id");
    return -1;
}

int MxArrowRenderer::addArrow(MxArrowData *arrow) {
    int arrowId = this->nextDataId();
    arrow->id = arrowId;
    
    if(arrowId == this->arrows.size()) this->arrows.push_back(arrow);
    else this->arrows[arrowId] = arrow;

    this->nr_arrows++;

    Log(LOG_DEBUG) << arrowId;

    return arrowId;
}

std::pair<int, MxArrowData*> MxArrowRenderer::addArrow(const MxVector3f &position, 
                                                       const MxVector3f &components, 
                                                       const MxStyle &style, 
                                                       const float &scale) 
{
    MxArrowData *arrow = new MxArrowData();
    arrow->position = position;
    arrow->components = components;
    arrow->style = style;
    arrow->scale = scale;

    int aid = this->addArrow(arrow);
    return std::make_pair(aid, arrow);
}

HRESULT MxArrowRenderer::removeArrow(const int &arrowId) {
    this->nr_arrows--;
    this->arrows[arrowId] = NULL;
    return S_OK;
}

MxArrowData *MxArrowRenderer::getArrow(const int &arrowId) {
    return this->arrows[arrowId];
}

MxArrowRenderer *MxArrowRenderer::get() {
    auto *sim = MxSimulator::get();
    auto *renderer = sim->getRenderer();
    return &renderer->arrowRenderer;
}
