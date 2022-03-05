/*
 * MxSurfaceSimulator.h
 *
 *  Created on: Mar 28, 2019
 *      Author: andy
 */

#ifndef SRC_MXSURFACESIMULATOR_H_
#define SRC_MXSURFACESIMULATOR_H_

#include <Mechanica.h>
#include "mechanica_private.h"
#include "rendering/MxApplication.h"
#include "MxCylinderModel.h"
#include "LangevinPropagator.h"
#include <rendering/MxMeshRenderer.h>
#include <rendering/ArcBallInteractor.h>

#include <Magnum/GL/Buffer.h>
#include <Magnum/GL/Framebuffer.h>
#include <Magnum/GL/Mesh.h>
#include <Magnum/GL/Renderbuffer.h>
#include <Magnum/GL/RenderbufferFormat.h>


struct MxSurfaceSimulator_Config {

    /**
     * Size of the frame buffer, width x height
     */
    int frameBufferSize[2];

    /**
     * path to model to load on initialization.
     */
    const char* modelPath;

    /**
     * Ignored if application already exists.
     */
    MxApplicationConfig applicationConfig;
};

struct MxSurfaceSimulator
{
    typedef MxSurfaceSimulator_Config Configuration;

    /**
     * Create a basic simulator
     */
    MxSurfaceSimulator(const Configuration &config);

    MxCylinderModel *model = nullptr;

    LangevinPropagator *propagator = nullptr;

    Magnum::Matrix4 transformation, projection;
    MxVector2f previousMousePosition;

    Magnum::Matrix4 rotation;

    MxVector3f centerShift{0., 0., -18};


    Color4 color; // = Color4::fromHsv(color.hue() + 50.0_degf, 1.0f, 1.0f);
    MxVector3f center;

    // distance from camera, move by mouse
    float distance = -3;


    MxMeshRenderer *renderer = nullptr;

    void loadModel(const char* fileName);

    // todo: implement MxSurfaceSimulator::step
    void step(float dt);

    void draw();

    // todo: implement MxSurfaceSimulator::mouseMove
    void mouseMove(double xpos, double ypos);

    // todo: implement MxSurfaceSimulator::mouseClick
    void mouseClick(int button, int action, int mods);

    int timeSteps = 0;

    ArcBallInteractor arcBall;

    GL::Renderbuffer renderBuffer;

    GL::Framebuffer frameBuffer;

    HRESULT createContext(const Configuration& configuration);

    std::tuple<char*, size_t> imageData(const char* path);

};

struct MxSurfaceSimulatorPy : MxSurfaceSimulator {

    MxSurfaceSimulatorPy(const Configuration &config) : MxSurfaceSimulator(config) {};

    PyObject *imageDataPy(const char* path);

};


#endif /* SRC_MXSURFACESIMULATOR_H_ */
