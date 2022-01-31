/**
 * @file MxOrientationRenderer.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines renderer for visualizing orientation. 
 * @date 2022-01-31
 * 
 */
#include "MxOrientationRenderer.h"

#include <engine.h>
#include <MxLogger.h>
#include <MxSystem.h>
#include <rendering/MxUniverseRenderer.h>

#include <Magnum/MeshTools/Compile.h>
#include <Magnum/MeshTools/Transform.h>
#include <Magnum/Primitives/Cone.h>
#include <Magnum/Primitives/Cylinder.h>
#include <Magnum/Primitives/Icosphere.h>
#include <Magnum/Shaders/Phong.h>
#include <Magnum/Trade/MeshData.h>

#include <limits>
#include <string>

struct OriginInstanceData {
    Matrix4 transformationMatrix;
    Matrix3 normalMatrix;
    Color4 color;
};

// todo: make arrow geometry settable

static float _arrowHeadHeight = 0.5;
static float _arrowHeadRadius = 0.25;
static float _arrowCylHeight = 0.5;
static float _arrowCylRadius = 0.1;

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

MxOrientationRenderer::MxOrientationRenderer() : 
    staticTransformationMat{Matrix4::translation({0.85f, -0.8f, -0.8f}) * Matrix4::scaling({0.1f, 0.1f, 0.1f})}
{}

MxOrientationRenderer::~MxOrientationRenderer() {
    this->arrows.clear();
}

void loadMesh(Magnum::GL::Buffer *_bufferHead,
              Magnum::GL::Buffer *_bufferCylinder, 
              Magnum::GL::Buffer *_bufferOrigin, 
              Magnum::GL::Mesh *_meshHead, 
              Magnum::GL::Mesh *_meshCylinder, 
              Magnum::GL::Mesh *_meshOrigin, 
              const std::vector<MxArrowData *> &arrows)
{
    _meshHead->setInstanceCount(arrows.size());
    _meshCylinder->setInstanceCount(arrows.size());
    _meshOrigin->setInstanceCount(1);

    _bufferHead->setData({NULL, arrows.size() * sizeof(MxArrowInstanceData)});
    _bufferCylinder->setData({NULL, arrows.size() * sizeof(MxArrowInstanceData)});

    MxArrowInstanceData *pArrowData = (MxArrowInstanceData*)(void*)_bufferHead->map(
        0, 
        arrows.size() * sizeof(MxArrowInstanceData), 
        Magnum::GL::Buffer::MapFlag::Write | Magnum::GL::Buffer::MapFlag::InvalidateBuffer
    );
    MxArrowInstanceData *pCylinderData = (MxArrowInstanceData*)(void*)_bufferCylinder->map(
        0, 
        arrows.size() * sizeof(MxArrowInstanceData), 
        Magnum::GL::Buffer::MapFlag::Write | Magnum::GL::Buffer::MapFlag::InvalidateBuffer
    );

    int i = 0;
    MxVector3f origin(_Engine.s.origin[0], _Engine.s.origin[1], _Engine.s.origin[2]);
    
    for (int aid = 0; aid < arrows.size(); aid++) {
        MxArrowData *ad = arrows[aid];
        if(ad != NULL) {
            render_arrow(pArrowData, pCylinderData, i, ad, origin);
            i++;
        }
    }

    assert(i == arrows.size());

    Matrix4 tm = Matrix4::from(Matrix4::scaling(Vector3(0.125f)).rotationScaling(), {});
    OriginInstanceData originData[] {
        {tm, tm.normalMatrix(), Color4{0.1f, 0.5f, 0.5f, 1.f}}
    };
    _bufferOrigin->setData(originData);

    _bufferHead->unmap();
    _bufferCylinder->unmap();
    _bufferOrigin->unmap();
}

HRESULT MxOrientationRenderer::start(const std::vector<MxVector4f> &clipPlanes) {

    // create the shaders
    _shader = Magnum::Shaders::MxPhong {
        Magnum::Shaders::MxPhong::Flag::VertexColor | 
        Magnum::Shaders::MxPhong::Flag::InstancedTransformation, 
        1, 0
    };
    _shader.setShininess(2000.0f)
        .setLightPositions({{-20, 40, 20, 0.f}})
        .setLightColors({Magnum::Color3{0.9, 0.9, 0.9}})
        .setShininess(100)
        .setAmbientColor({0.4, 0.4, 0.4, 1})
        .setDiffuseColor({1, 1, 1, 0})
        .setSpecularColor({0.2, 0.2, 0.2, 0});
    
    // create the buffers
    _bufferHead = Magnum::GL::Buffer();
    _bufferCylinder = Magnum::GL::Buffer();
    _bufferOrigin = Magnum::GL::Buffer();

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
    _meshOrigin = MeshTools::compile(Primitives::icosphereSolid(2));

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
    _meshOrigin.addVertexBufferInstanced(_bufferOrigin, 1, 0, 
        Magnum::Shaders::Phong::TransformationMatrix{}, 
        Magnum::Shaders::Phong::NormalMatrix{}, 
        Magnum::Shaders::Phong::Color4{}
    );

    this->modelViewMat = MxMatrix4f::translation(-(MxUniverse::dim() + MxUniverse::origin()) / 2.);

    this->arrowx = new MxArrowData();
    this->arrowy = new MxArrowData();
    this->arrowz = new MxArrowData();

    this->arrowx->style.color = Color3::red();
    this->arrowy->style.color = Color3::green();
    this->arrowz->style.color = Color3::blue();

    this->arrowx->components = MxVector3f::xAxis();
    this->arrowy->components = MxVector3f::yAxis();
    this->arrowz->components = MxVector3f::zAxis();

    this->arrowx->scale = 0.5f;
    this->arrowy->scale = 0.5f;
    this->arrowz->scale = 0.5f;

    this->arrows = {this->arrowx, this->arrowy, this->arrowz};

    loadMesh(&_bufferHead, &_bufferCylinder, &_bufferOrigin, &_meshHead, &_meshCylinder, &_meshOrigin, this->arrows);

    return S_OK;
}

HRESULT MxOrientationRenderer::draw(Magnum::Mechanica::ArcBallCamera *camera, const MxVector2i& viewportSize, const MxMatrix4f &modelViewMat) {
    Log(LOG_DEBUG) << "";

    if(!this->_showAxes) 
        return S_OK;

    _shader
        .setProjectionMatrix(Matrix4())
        .setTransformationMatrix(
            this->staticTransformationMat * 
            Matrix4::from((camera->projectionMatrix() * camera->cameraMatrix() * this->modelViewMat).rotationScaling(), {})
        )
        .setNormalMatrix(camera->viewMatrix().normalMatrix());
    _shader.draw(_meshHead);
    _shader.draw(_meshCylinder);
    _shader.draw(_meshOrigin);

    return S_OK;
}

void MxOrientationRenderer::setAmbientColor(const Magnum::Color3& color) {
    _shader.setAmbientColor(color);
}

void MxOrientationRenderer::setDiffuseColor(const Magnum::Color3& color) {
    _shader.setDiffuseColor(color);
}

void MxOrientationRenderer::setSpecularColor(const Magnum::Color3& color) {
    _shader.setSpecularColor(color);
}

void MxOrientationRenderer::setShininess(float shininess) {
    _shader.setShininess(shininess);
}

void MxOrientationRenderer::setLightDirection(const MxVector3f& lightDir) {
    _shader.setLightPosition(lightDir);
}

void MxOrientationRenderer::setLightColor(const Magnum::Color3 &color) {
    _shader.setLightColor(color);
}

MxOrientationRenderer *MxOrientationRenderer::get() {
    auto *renderer = MxSystem::getRenderer();
    return (MxOrientationRenderer*)renderer->getSubRenderer(MxSubRendererFlag::SUBRENDERER_ORIENTATION);
}
