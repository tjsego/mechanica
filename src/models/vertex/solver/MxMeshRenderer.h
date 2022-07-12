/**
 * @file MxMeshRenderer.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines the Mechanica mesh renderer
 * @date 2022-05-11
 * 
 */

#ifndef MODELS_VERTEX_SOLVER_MXMESHRENDERER_H_
#define MODELS_VERTEX_SOLVER_MXMESHRENDERER_H_

#include <rendering/MxSubRenderer.h>
#include <shaders/MxFlat3D.h>

#include <Magnum/GL/GL.h>
#include <Magnum/GL/Buffer.h>
#include <Magnum/GL/Mesh.h>
#include <Magnum/Shaders/Shaders.h>


struct MxMeshRenderer : MxSubRenderer {

    static MxMeshRenderer *get();
    
    HRESULT start(const std::vector<MxVector4f> &clipPlanes);

    HRESULT draw(Magnum::Mechanica::ArcBallCamera *camera, const MxVector2i &viewportSize, const MxMatrix4f &modelViewMat);

private:

    std::vector<Magnum::Vector4> _clipPlanes;

    Magnum::GL::Buffer _bufferFaces{Corrade::Containers::NoCreate};
    Magnum::GL::Buffer _bufferEdges{Corrade::Containers::NoCreate};
    Magnum::GL::Mesh _meshFaces{Corrade::Containers::NoCreate};
    Magnum::GL::Mesh _meshEdges{Corrade::Containers::NoCreate};
    Magnum::Shaders::MxFlat3D _shaderFaces{Corrade::Containers::NoCreate};
    Magnum::Shaders::MxFlat3D _shaderEdges{Corrade::Containers::NoCreate};

};

#endif // MODELS_VERTEX_SOLVER_MXMESHRENDERER_H_