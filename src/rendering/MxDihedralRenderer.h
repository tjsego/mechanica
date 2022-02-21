/**
 * @file MxDihedralRenderer.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines renderer for visualizing MxDihedrals. 
 * @date 2021-10-04
 * 
 */
#ifndef SRC_RENDERING_MXDIHEDRALRENDERER_H_
#define SRC_RENDERING_MXDIHEDRALRENDERER_H_

#include "MxSubRenderer.h"

#include <shaders/MxFlat3D.h>
#include <rendering/MxStyle.hpp>

#include <Magnum/GL/Mesh.h>
#include <Magnum/Math/Color.h>
#include <Magnum/Mesh.h>


struct MxDihedralRenderer : MxSubRenderer {
    HRESULT start(const std::vector<MxVector4f> &clipPlanes);
    HRESULT draw(Magnum::Mechanica::ArcBallCamera *camera, const MxVector2i &viewportSize, const MxMatrix4f &modelViewMat);
    const unsigned addClipPlaneEquation(const Magnum::Vector4& pe);
    const unsigned removeClipPlaneEquation(const unsigned int &id);
    void setClipPlaneEquation(unsigned id, const Magnum::Vector4& pe);

private:
    
    std::vector<Magnum::Vector4> _clipPlanes;

    Magnum::Shaders::MxFlat3D _shader{Corrade::Containers::NoCreate};
    Magnum::GL::Buffer _buffer{Corrade::Containers::NoCreate};
    Magnum::GL::Mesh _mesh{Corrade::Containers::NoCreate};
};

#endif // SRC_RENDERING_MXDIHEDRALRENDERER_H_