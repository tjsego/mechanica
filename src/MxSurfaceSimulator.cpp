/*
 * MxSurfaceSimulator.cpp
 *
 *  Created on: Mar 28, 2019
 *      Author: andy
 */

#include <mx_config.h>
#include <MxSurfaceSimulator.h>
#include "MeshOperations.h"
#include "Magnum/GL/Version.h"
#include "Magnum/Platform/GLContext.h"

#include <Magnum/GL/Buffer.h>
#include <Magnum/GL/DefaultFramebuffer.h>
#include <Magnum/Mesh.h>
#include <Magnum/Math/Vector3.h>
#include <Magnum/Math/Matrix3.h>
#include <Magnum/Platform/GlfwApplication.h>
#include <Magnum/Shaders/VertexColor.h>
#include <Magnum/Primitives/Cube.h>
#include <Magnum/GL/Version.h>
#include <Magnum/GL/Renderer.h>
#include <Magnum/Trade/MeshData.h>

#include <Magnum/PixelFormat.h>
#include <Magnum/Image.h>
#include <Corrade/Utility/Directory.h>
#include <rendering/MxImageConverters.h>



#include <memory>
#include <iostream>

using namespace Math::Literals;


MxSurfaceSimulator::MxSurfaceSimulator(const Configuration& config) :
        frameBuffer{Magnum::NoCreate}
{
    std::cout << MX_FUNCTION << std::endl;

    createContext(config);

    // need to enabler depth testing. The graphics processor can draw each facet in any order it wants.
    // Depth testing makes sure that front facing facts are drawn after back ones, so that back facets
    // don't cover up front ones.
    GL::Renderer::enable(GL::Renderer::Feature::DepthTest);

    // don't draw facets that face away from us. We have A LOT of these INSIDE cells, no need to
    // draw them.
    GL::Renderer::enable(GL::Renderer::Feature::FaceCulling);

    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable( GL_BLEND );

    renderer = new MxMeshRenderer{MxMeshRenderer::Flag::Wireframe};

    GL::Renderer::setClearColor(Color4{1.0f, 1.0f, 1.0f, 1.0f});

    renderBuffer.setStorage(GL::RenderbufferFormat::RGBA8, {config.frameBufferSize[0], config.frameBufferSize[1]});

    frameBuffer = GL::Framebuffer{{{0,0}, {config.frameBufferSize[0], config.frameBufferSize[1]}}};

    frameBuffer.attachRenderbuffer(GL::Framebuffer::ColorAttachment{0}, renderBuffer)
            .clear(GL::FramebufferClear::Color)
            .bind();
    
    loadModel(config.modelPath);
}


HRESULT MxSurfaceSimulator::createContext(const Configuration& configuration) {

    //CORRADE_ASSERT(context->version() ==
    //        GL::Version::None,
    //        "Platform::GlfwApplication::tryCreateContext(): context already created",
    //        false);

    /* Window flags */


    
    assert(Magnum::GL::Context::hasCurrent() && "must have context, should be created by application");

    return S_OK;
}

void MxSurfaceSimulator::loadModel(const char* fileName)
{
    std::cout << MX_FUNCTION << ", fileName: " << fileName << std::endl;

    delete model;
    delete propagator;

    model = new MxCylinderModel{};

    propagator = new LangevinPropagator{};

    VERIFY(MxBind_PropagatorModel(propagator, model));

    VERIFY(model->loadModel(fileName));

    renderer->setMesh(model->mesh);

    draw();
}

void MxSurfaceSimulator::step(float dt) {
    MX_NOTIMPLEMENTED_NORET
}

void MxSurfaceSimulator::draw() {
    
    frameBuffer.bind();

    Vector3 min, max;
    std::tie(min, max) = model->mesh->extents();

    center = (max + min)/2;

    frameBuffer.clear(GL::FramebufferClear::Color|GL::FramebufferClear::Depth);

    renderer->setViewportSize(Vector2{frameBuffer.viewport().size()});

    projection = Matrix4::perspectiveProjection(35.0_degf,
        Vector2{frameBuffer.viewport().size()}.aspectRatio(),
        0.01f, 100.0f
        );

    renderer->setProjectionMatrix(projection);

    //rotation = arcBall.rotation();
    
    rotation = Matrix4::rotationZ(-1.4_radf);
    
    rotation = rotation * Matrix4::rotationX(0.5_radf);

    Matrix4 mat = Matrix4::translation(centerShift) * rotation * Matrix4::translation(-center) ;

    renderer->setViewMatrix(mat);

    renderer->setColor(Color4::yellow());

    renderer->setWireframeColor(Color4{0., 0., 0.});

    renderer->setWireframeWidth(2.0);

    Debug{} << "viewport size: " << frameBuffer.viewport().size();
    Debug{} << "center: " << center;
    Debug{} << "centerShift: " << centerShift;
    Debug{} << "projection: " << projection;
    Debug{} << "view matrix: " << mat;

    renderer->draw();
}

void MxSurfaceSimulator::mouseMove(double xpos, double ypos) {
    MX_NOTIMPLEMENTED_NORET
}

void MxSurfaceSimulator::mouseClick(int button, int action, int mods) {
    MX_NOTIMPLEMENTED_NORET
}

std::tuple<char*, size_t> MxSurfaceSimulator::imageData(const char* path)
{
    const GL::PixelFormat format = this->frameBuffer.implementationColorReadFormat();
    Image2D image = this->frameBuffer.read(this->frameBuffer.viewport(), PixelFormat::RGBA8Unorm);

    auto jpegData = convertImageDataToJpeg(image);

    /* Open file */
    if(!Utility::Directory::write(path, jpegData)) {
        Error() << "Trade::AbstractImageConverter::exportToFile(): cannot write to file" << "triangle.jpg";
        return std::make_tuple((char*)NULL, (size_t)0);
    }

    return std::make_tuple(jpegData.data(), jpegData.size());
}

PyObject* MxSurfaceSimulatorPy::imageDataPy(const char* path)
{
    char *data;
    size_t size;
    std::tie(data, size) = imageData(path);
    if(data==NULL) {
        Error() << "Trade::AbstractImageConverter::exportToFile(): cannot write to file" << "triangle.jpg";
        return NULL;
    }
    return PyBytes_FromStringAndSize(data, size);
}
