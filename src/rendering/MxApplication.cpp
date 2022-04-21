/*
 * MxApplication.cpp
 *
 *  Created on: Mar 27, 2019
 *      Author: andy
 */

#include <rendering/MxApplication.h>
#include <rendering/MxWindowlessApplication.h>

#include <Magnum/GL/Buffer.h>
#include <Magnum/GL/Framebuffer.h>
#include <Magnum/GL/Mesh.h>
#include <Magnum/GL/Renderbuffer.h>
#include <Magnum/GL/RenderbufferFormat.h>
#include <Magnum/GL/DefaultFramebuffer.h>
#include <Magnum/GL/Renderer.h>
#include <Magnum/GL/PixelFormat.h>

#include <Magnum/Shaders/VertexColor.h>
#include <Magnum/PixelFormat.h>
#include <Magnum/Image.h>
#include <Magnum/Animation/Easing.h>


#include <Corrade/Utility/Directory.h>
#include <Corrade/Utility/String.h>
#include <Magnum/Math/Color.h>

#include <MxUtil.h>
#include <MxLogger.h>

using namespace Magnum;

#include <iostream>


static void _test_draw(Magnum::GL::AbstractFramebuffer &framebuffer) {
    
    framebuffer
    .clear(GL::FramebufferClear::Color)
    .bind();
    
    using namespace Math::Literals;
    
    struct TriangleVertex {
        Vector2 position;
        Color3 color;
    };
    
    const TriangleVertex data[]{
        {{-0.5f, -0.5f}, 0xff0000_rgbf},    /* Left vertex, red color */
        {{ 0.5f, -0.5f}, 0x00ff00_rgbf},    /* Right vertex, green color */
        {{ 0.0f,  0.5f}, 0x0000ff_rgbf}     /* Top vertex, blue color */
    };
    
    GL::Buffer buffer;
    buffer.setData(data);
    
    GL::Mesh mesh;
    mesh.setCount(3)
    .addVertexBuffer(std::move(buffer), 0,
                     Shaders::VertexColor2D::Position{},
                     Shaders::VertexColor2D::Color3{});
    
  
    
    Shaders::VertexColor2D shader;
    shader.draw(mesh);
}

std::tuple<char*, size_t> MxTestImage() {
    GL::Renderbuffer renderbuffer;
    renderbuffer.setStorage(GL::RenderbufferFormat::RGBA8, {640, 480});
    GL::Framebuffer framebuffer{{{}, {640, 480}}};
    
    framebuffer.attachRenderbuffer(GL::Framebuffer::ColorAttachment{0}, renderbuffer);
 
    _test_draw(framebuffer);

    const GL::PixelFormat format = framebuffer.implementationColorReadFormat();
    Image2D image = framebuffer.read(framebuffer.viewport(), PixelFormat::RGBA8Unorm);

    auto jpegData = convertImageDataToJpeg(image);

    /* Open file */
    if(!Utility::Directory::write("triangle.jpg", jpegData)) {
        Error() << "Trade::AbstractImageConverter::exportToFile(): cannot write to file" << "triangle.jpg";
        return std::make_tuple((char*)NULL, (size_t)0);
    }

    return std::make_tuple(jpegData.data(), jpegData.size());
}

Magnum::GL::AbstractFramebuffer *MxGetFrameBuffer() {
    PerformanceTimer t1(engine_timer_image_data);
    PerformanceTimer t2(engine_timer_render_total);
    
    Log(LOG_TRACE);
    
    if(!Magnum::GL::Context::hasCurrent()) {
        mx_error(E_FAIL, "No current OpenGL context");
        return NULL;
    }
    
    MxSimulator *sim = MxSimulator::get();
    
    sim->app->redraw();
    
    return &sim->app->framebuffer();
}

typedef Corrade::Containers::Array<char> (*imgCnv_t)(ImageView2D);
typedef Corrade::Containers::Array<char> (*imgGen_t)();

Corrade::Containers::Array<char> MxImageData(imgCnv_t imgCnv, const PixelFormat &format) {
    PerformanceTimer t1(engine_timer_image_data);
    PerformanceTimer t2(engine_timer_render_total);
    
    Log(LOG_TRACE);
    
    Magnum::GL::AbstractFramebuffer *framebuffer = MxGetFrameBuffer();

    if(!framebuffer) 
        return Corrade::Containers::Array<char>();

    return imgCnv(framebuffer->read(framebuffer->viewport(), format));
}

Corrade::Containers::Array<char> MxJpegImageData() {
    return MxImageData((imgCnv_t)[](ImageView2D image) { return convertImageDataToJpeg(image, 100); }, PixelFormat::RGB8Unorm);
}

Corrade::Containers::Array<char> MxBMPImageData() {
    return MxImageData((imgCnv_t)convertImageDataToBMP, PixelFormat::RGB8Unorm);
}

Corrade::Containers::Array<char> MxHDRImageData() {
    return MxImageData((imgCnv_t)convertImageDataToHDR, PixelFormat::RGB32F);
}

Corrade::Containers::Array<char> MxPNGImageData() {
    return MxImageData((imgCnv_t)convertImageDataToPNG, PixelFormat::RGBA8Unorm);
}

Corrade::Containers::Array<char> MxTGAImageData() {
    return MxImageData((imgCnv_t)convertImageDataToTGA, PixelFormat::RGBA8Unorm);
}

std::tuple<char*, size_t> MxFramebufferImageData() {
    PerformanceTimer t1(engine_timer_image_data);
    PerformanceTimer t2(engine_timer_render_total);
    
    Log(LOG_TRACE);
    
    auto jpegData = MxJpegImageData();

    return std::make_tuple(jpegData.data(), jpegData.size());
}

HRESULT MxScreenshot(const std::string &filePath) {
    Log(LOG_TRACE);

    std::string filePath_l = Utility::String::lowercase(filePath);

    imgGen_t imgGen;

    if(Utility::String::endsWith(filePath_l, ".bmp"))
        imgGen = MxBMPImageData;
    else if(Utility::String::endsWith(filePath_l, ".hdr"))
        imgGen = MxHDRImageData;
    else if(Utility::String::endsWith(filePath_l, ".jpe") || Utility::String::endsWith(filePath_l, ".jpg") || Utility::String::endsWith(filePath_l, ".jpeg"))
        imgGen = MxJpegImageData;
    else if(Utility::String::endsWith(filePath_l, ".png"))
        imgGen = MxPNGImageData;
    else if(Utility::String::endsWith(filePath_l, ".tga"))
        imgGen = MxTGAImageData;
    else {
        Log(LOG_ERROR) << "Cannot determined file format from file path: " << filePath;
        return E_FAIL;
    }
    
    if(!Utility::Directory::write(filePath, imgGen())) {
        std::string msg = "Cannot write to file: " + filePath;
        mx_error(E_FAIL, msg.c_str());
        return E_FAIL;
    }

    return S_OK;
}

HRESULT MxApplication::simulationStep() {
    
    /* Pause simulation if the mouse was pressed (camera is moving around).
     This avoid freezing GUI while running the simulation */
    
    /*
     if(!_pausedSimulation && !_mousePressed) {
     // Adjust the substep number to maximize CPU usage each frame
     const Float lastAvgStepTime = _timeline.previousFrameDuration()/Float(_substeps);
     const Int newSubsteps = lastAvgStepTime > 0 ? Int(1.0f/60.0f/lastAvgStepTime) + 1 : 1;
     if(Math::abs(newSubsteps - _substeps) > 1) _substeps = newSubsteps;
     
     // TODO: move substeps to universe step.
     if(MxUniverse_Flag(MxUniverse_Flags::MX_RUNNING)) {
     for(Int i = 0; i < _substeps; ++i) {
     MxUniverse_Step(0, 0);
     }
     }
     }
     */
    
    static Float offset = 0.0f;
    if(_dynamicBoundary) {
        /* Change fluid boundary */
        static Float step = 2.0e-3f;
        if(_boundaryOffset > 1.0f || _boundaryOffset < 0.0f) {
            step *= -1.0f;
        }
        _boundaryOffset += step;
        offset = Math::lerp(0.0f, 0.5f, Animation::Easing::quadraticInOut(_boundaryOffset));
    }
    
    currentStep += 1;
    
    // TODO: get rid of this
    return MxUniverse::step(0,0);
}

HRESULT MxApplication::run(double et)
{
    Log(LOG_TRACE);
    MxUniverse_SetFlag(MX_RUNNING, true);
    HRESULT result = messageLoop(et);
    MxUniverse_SetFlag(MX_RUNNING, false);
    return result;
}
