/**
 * @file MxSubRenderer.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines interface for subrenderer children of MxUniverseRenderer
 * @date 2021-09-07
 * 
 */
#ifndef SRC_RENDERING_MXSUBRENDERER_H_
#define SRC_RENDERING_MXSUBRENDERER_H_

#include <mechanica_private.h>

#include <rendering/ArcBallCamera.h>


struct MxSubRenderer{

    virtual ~MxSubRenderer() {}

    /**
     * @brief Starts the renderer. 
     * 
     * Called by parent renderer once backend is initialized. 
     * 
     * @param clipPlanes clip plane specification
     * @return HRESULT 
     */
    virtual HRESULT start(const std::vector<MxVector4f> &clipPlanes) = 0;

    /**
     * @brief Updates visualization. 
     * 
     * @param camera scene camera
     * @param viewportSize scene viewport size
     * @param modelViewMat scene model view matrix
     * @return HRESULT 
     */
    virtual HRESULT draw(Magnum::Mechanica::ArcBallCamera *camera, const MxVector2i &viewportSize, const MxMatrix4f &modelViewMat) = 0;

    /**
     * @brief Adds a clip plane equation
     * 
     * @param pe clip plane equation
     * @return const unsigned 
     */
    virtual const unsigned addClipPlaneEquation(const Magnum::Vector4& pe) { return E_NOTIMPL; }
    
    /**
     * @brief Removes a clip plane equation
     * 
     * @param id id of clip plane equation
     * @return const unsigned 
     */
    virtual const unsigned removeClipPlaneEquation(const unsigned int &id) { return E_NOTIMPL; }

    /**
     * @brief Sets a clip plane equation
     * 
     * @param id id of clip plane equation
     * @param pe clip plane equation
     */
    virtual void setClipPlaneEquation(unsigned id, const Magnum::Vector4& pe) {}

};

#endif // SRC_RENDERING_MXSUBRENDERER_H_