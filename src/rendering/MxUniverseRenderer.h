/*
    This file is part of Mechanica.

    Based on Magnum example

    Original authors — credit is appreciated but not required:

        2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019 —
            Vladimír Vondruš <mosra@centrum.cz>
        2019 — Nghia Truong <nghiatruong.vn@gmail.com>

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2.1 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.
 */


#pragma once

#include <vector>

#include <Magnum/GL/Buffer.h>
#include <Magnum/GL/Mesh.h>

#include <MxUniverse.h>
#include <rendering/MxRenderer.h>
#include <rendering/MxGlfwWindow.h>
#include <shaders/ParticleSphereShader.h>

#include <shaders/MxPhong.h>

#include <Corrade/Containers/Pointer.h>

#include <Corrade/Containers/Pointer.h>
#include <Magnum/Math/Color.h>
#include <Magnum/Primitives/Cube.h>
#include <Magnum/Platform/GlfwApplication.h>
#include <Magnum/SceneGraph/MatrixTransformation3D.h>
#include <Magnum/Timeline.h>

#include <Magnum/Shaders/Phong.h>
#include <Magnum/Shaders/Flat.h>

#include <rendering/MxUniverseRenderer.h>
#include <rendering/MxGlfwWindow.h>

#include <Magnum/Platform/GlfwApplication.h>

#include <rendering/MxWindow.h>
#include <rendering/ArcBallCamera.h>

#include <rendering/MxArrowRenderer.h>

struct MxSimulator_Config;

class WireframeGrid;
class WireframeBox;

struct SphereInstanceData {
    Magnum::Matrix4 transformationMatrix;
    Magnum::Matrix3x3 normalMatrix;
    Magnum::Color4 color;
};

struct BondsInstanceData {
    Magnum::Vector3 position;
    Magnum::Color3 color;
};

struct CuboidInstanceData {
    Magnum::Matrix4 transformationMatrix;
    Magnum::Matrix3x3 normalMatrix;
    Magnum::Color4 color;
};

struct MxUniverseRenderer : MxRenderer {


    // TODO, implement the event system instead of hard coding window events.
    MxUniverseRenderer(const MxSimulator_Config &conf, MxWindow *win);

    template<typename T>
    MxUniverseRenderer& draw(T& camera, const MxVector2i& viewportSize);

    bool& isDirty() { return _dirty; }

    MxUniverseRenderer& setDirty() {
        _dirty = true;
        return *this;
    }

    ParticleSphereShader::ColorMode& colorMode() { return _colorMode; }

    MxUniverseRenderer& setColorMode(ParticleSphereShader::ColorMode colorMode) {
        _colorMode = colorMode;
        return *this;
    }

    Color3& ambientColor() { return _ambientColor; }

    MxUniverseRenderer& setAmbientColor(const Color3& color) {
        _ambientColor = color;
        return *this;
    }

    Color3& diffuseColor() { return _diffuseColor; }

    MxUniverseRenderer& setDiffuseColor(const Color3& color) {
        _diffuseColor = color;
        return *this;
    }

    Color3& specularColor() { return _specularColor; }

    MxUniverseRenderer& setSpecularColor(const Color3& color) {
        _specularColor = color;
        return *this;
    }

    Float& shininess() { return _shininess; }

    MxUniverseRenderer& setShininess(float shininess) {
        _shininess = shininess;
        return *this;
    }

    MxVector3f& lightDirection() { return _lightDir; }

    MxUniverseRenderer& setLightDirection(const MxVector3f& lightDir) {
        _lightDir = lightDir;
        return *this;
    }

    MxUniverseRenderer& setModelViewTransform(const Magnum::Matrix4& mat) {
        modelViewMat = mat;
        return *this;
    }

    MxUniverseRenderer& setProjectionTransform(const Magnum::Matrix4& mat) {
        projMat = mat;
        return *this;
    }
    
    const MxVector3f& defaultEye() const {
        return _eye;
    }
    
    const MxVector3f& defaultCenter() const {
        return _center;
    }
    
    const MxVector3f& defaultUp() const {
        return _up;
    }

    bool renderUniverse = true;


    void onCursorMove(double xpos, double ypos);

    void onCursorEnter(int entered);

    void onMouseButton(int button, int action, int mods);

    void onRedraw();

    void onWindowMove(int x, int y);

    void onWindowSizeChange(int x, int y);

    void onFramebufferSizeChange( int x, int y);

    void viewportEvent(const int w, const int h);

    void draw();
    
    int clipPlaneCount() const;
    
    void setClipPlaneEquation(unsigned id, const Magnum::Vector4& pe);
    
    const Magnum::Vector4& getClipPlaneEquation(unsigned id);

    const float getZoomRate();

    void setZoomRate(const float &zoomRate);

    const float getSpinRate();

    void setSpinRate(const float &spinRate);

    const float getMoveRate();

    void setMoveRate(const float &moveRate);

    void viewportEvent(Platform::GlfwApplication::ViewportEvent& event);

    /**
     * @brief Key press event handling. 
     * 
     * @details When a key is pressed, actions are as follows by key and modifier: 
     * 
     * - L: toggle lagging
     * - R: reset camera
     * - Arrow down: translate camera down
     * - Arrow left: translate camera left
     * - Arrow right: translate camera right
     * - Arrow up: translate camera up
     * - Ctrl + arrow down: zoom camera out
     * - Ctrl + arrow left: rotate camera left
     * - Ctrl + arrow right: rotate camera right
     * - Ctrl + arrow up: zoom camera in
     * - Shift + B: bottom view
     * - Shift + F: front view
     * - Shift + K: back view
     * - Shift + L: left view
     * - Shift + R: right view
     * - Shift + T: top view
     * - Shift + arrow down: rotate camera down
     * - Shift + arrow left: rotate camera left
     * - Shift + arrow right: rotate camera right
     * - Shift + arrow up: rotate camera up
     * - Shift + Ctrl + arrow down: translate backward
     * - Shift + Ctrl + arrow up: translate forward
     * 
     * @param event 
     */
    void keyPressEvent(Platform::GlfwApplication::KeyEvent& event);
    void mousePressEvent(Platform::GlfwApplication::MouseEvent& event);
    void mouseReleaseEvent(Platform::GlfwApplication::MouseEvent& event);

    /**
     * @brief Mouse move event handling. 
     * 
     * @details When a mouse button is pressed, actions are as follows by modifier:
     *  - Shift: translates the camera
     *  - Ctrl: zooms the camera
     *  - None: rotates the camera
     *  .
     * 
     * @param event 
     */
    void mouseMoveEvent(Platform::GlfwApplication::MouseMoveEvent& event);
    void mouseScrollEvent(Platform::GlfwApplication::MouseScrollEvent& event);


    bool _dirty = false;
    ParticleSphereShader::ColorMode _colorMode = ParticleSphereShader::ColorMode::ConsistentRandom;
    Color3 _ambientColor{0.1f};
    Color3 _diffuseColor{0.0f, 0.5f, 0.9f};
    Color3 _specularColor{ 1.0f};
    Float _shininess = 150.0f;
    MxVector3f _lightDir{1.0f, 1.0f, 2.0f};
    
    MxVector3f _eye, _center, _up;
    
    std::vector<Magnum::Vector4> _clipPlanes;
    
    /**
     * Only set a single combined matrix in the shader, this way,
     * the shader only performs a single matrix multiply of the vertices, update the
     * shader matrix whenever any of these change.
     *
     * multiplication order is the reverse of the pipeline.
     * Therefore you do totalmat = proj * view * model.
     */
    Magnum::Matrix4 modelViewMat = Matrix4{Math::IdentityInit};
    Magnum::Matrix4 projMat =  Matrix4{Math::IdentityInit};

    MxVector2i _prevMousePosition;
    MxVector3f  _rotationPoint, _translationPoint;
    Float _lastDepth;
    
    float sideLength;
    float _zoomRate;
    float _spinRate;
    float _moveRate;

    Magnum::Mechanica::ArcBallCamera *_arcball;
    
    /* ground grid */
    GL::Mesh gridMesh{NoCreate};
    Magnum::Matrix4 gridModelView;
    
    GL::Mesh sceneBox{NoCreate};

    
    /* Spheres rendering */
    
    Shaders::MxPhong sphereShader{NoCreate};
    
    Shaders::Flat3D flatShader{NoCreate};
    
    Shaders::Flat3D wireframeShader{NoCreate};
    
    GL::Buffer sphereInstanceBuffer{NoCreate};
    
    GL::Buffer largeSphereInstanceBuffer{NoCreate};

    GL::Mesh sphereMesh{NoCreate};

    GL::Mesh largeSphereMesh{NoCreate};
    
    GL::Mesh bondsMesh{NoCreate};
    
    GL::Mesh cuboidMesh{NoCreate};
    
    GL::Buffer cuboidInstanceBuffer{NoCreate};
    
    GL::Buffer bondsVertexBuffer{NoCreate};

    MxArrowRenderer arrowRenderer;

    MxVector3f center;

    MxWindow *window;

    MxVector3f unproject(const MxVector2i& windowPosition, float depth) const;

    // todo: implement MxUniverseRenderer::setupCallbacks
    void setupCallbacks();
    
    ~MxUniverseRenderer();
};
