/**
 * @file MxOrientationRenderer.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines renderer for visualizing orientaton. 
 * @date 2022-01-31
 * 
 */
#ifndef SRC_RENDERING_MXORIENTATIONRENDERER_H_
#define SRC_RENDERING_MXORIENTATIONRENDERER_H_

#include "MxSubRenderer.h"
#include "MxArrowRenderer.h"

#include <shaders/MxPhong.h>
#include <rendering/MxStyle.hpp>

#include <Magnum/GL/Mesh.h>
#include <Magnum/Math/Color.h>
#include <Magnum/Mesh.h>

#include <utility>
#include <vector>

/**
 * @brief Orientaton renderer. 
 * 
 * Visualizes the orientation of the scene. 
 * 
 */
struct MxOrientationRenderer : MxSubRenderer{

    // Arrow inventory
    std::vector<MxArrowData *> arrows;

    MxOrientationRenderer();
    ~MxOrientationRenderer();

    HRESULT start(const std::vector<MxVector4f> &clipPlanes);
    HRESULT draw(Magnum::Mechanica::ArcBallCamera *camera, const MxVector2i &viewportSize, const MxMatrix4f &modelViewMat);
    void setAmbientColor(const Magnum::Color3& color);
    void setDiffuseColor(const Magnum::Color3& color);
    void setSpecularColor(const Magnum::Color3& color);
    void setShininess(float shininess);
    void setLightDirection(const MxVector3f& lightDir);
    void setLightColor(const Magnum::Color3 &color);

    /**
     * @brief Gets the global instance of the renderer. 
     * 
     * Cannot be used until the universe renderer has been initialized. 
     * 
     * @return MxOrientationRenderer* 
     */
    static MxOrientationRenderer *get();

    void showAxes(const bool &show) {
        _showAxes = show;
    }

private:

    int _arrowDetail = 10;
    int _showAxes;

    Magnum::GL::Buffer _bufferHead{Corrade::Containers::NoCreate};
    Magnum::GL::Buffer _bufferCylinder{Corrade::Containers::NoCreate};
    Magnum::GL::Buffer _bufferOrigin{Corrade::Containers::NoCreate};
    Magnum::GL::Mesh _meshHead{Corrade::Containers::NoCreate};
    Magnum::GL::Mesh _meshCylinder{Corrade::Containers::NoCreate};
    Magnum::GL::Mesh _meshOrigin{Corrade::Containers::NoCreate};
    Magnum::Shaders::MxPhong _shader{Corrade::Containers::NoCreate};

    MxMatrix4f modelViewMat;
    MxMatrix4f staticTransformationMat;

    MxArrowData *arrowx=NULL, *arrowy=NULL, *arrowz=NULL;

};

#endif // SRC_RENDERING_MXORIENTATIONRENDERER_H_
