/**
 * @file MxArrowRenderer.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines renderer for visualizing vectors. 
 * @date 2021-09-08
 * 
 */
#ifndef SRC_RENDERING_MXARROWRENDERER_H_
#define SRC_RENDERING_MXARROWRENDERER_H_

#include "MxSubRenderer.h"

#include <shaders/MxPhong.h>
#include <rendering/MxStyle.hpp>

#include <Magnum/GL/Mesh.h>
#include <Magnum/Math/Color.h>
#include <Magnum/Mesh.h>

#include <utility>
#include <vector>

/**
 * @brief Vector visualization specification. 
 * 
 * MxArrowRenderer uses instances of MxArrowData to visualize 
 * vectors as arrows in the simulation domain. 
 * 
 */
struct MxArrowData{
    // Position of the arrow
    MxVector3f position;

    // Vector components of the arrow
    MxVector3f components;

    // Mechanica style
    MxStyle style;

    // Scaling applied to arrow
    float scale = 1.0;

private:

    int id;
    friend struct MxArrowRenderer;
};

/**
 * @brief Vector renderer. 
 * 
 * Vector visualization specification can be passed 
 * dynamically. Visualization specs are not managed by 
 * the renderer. It is the responsibility of the client 
 * to manage specs appropriately. 
 * 
 * By default, a vector is visualized with the same 
 * orientation as its underlying data, where one unit 
 * of magnitude of the vector corresponds to a visualized 
 * arrow with a length of one in the scene. 
 * 
 */
struct MxArrowRenderer : MxSubRenderer{

    // Current number of arrows in inventory
    int nr_arrows;

    // Arrow inventory
    std::vector<MxArrowData *> arrows;

    MxArrowRenderer();
    MxArrowRenderer(const MxArrowRenderer &other);
    ~MxArrowRenderer();

    HRESULT start(const std::vector<MxVector4f> &clipPlanes);
    HRESULT draw(Magnum::Mechanica::ArcBallCamera *camera, const MxVector2i &viewportSize, const MxMatrix4f &modelViewMat);
    const unsigned addClipPlaneEquation(const Magnum::Vector4& pe);
    const unsigned removeClipPlaneEquation(const unsigned int &id);
    void setClipPlaneEquation(unsigned id, const Magnum::Vector4& pe);

    /**
     * @brief Adds a vector visualization specification. 
     * 
     * The passed pointer is borrowed. The client is 
     * responsible for maintaining the underlying data. 
     * The returned integer can be used to reference the 
     * arrow when doing subsequent operations with the 
     * renderer (e.g., removing an arrow from the scene). 
     * 
     * @param arrow pointer to visualization specs
     * @return int id of arrow according to the renderer
     */
    int addArrow(MxArrowData *arrow);

    /**
     * @brief Adds a vector visualization specification. 
     * 
     * The passed pointer is borrowed. The client is 
     * responsible for maintaining the underlying data. 
     * The returned integer can be used to reference the 
     * arrow when doing subsequent operations with the 
     * renderer (e.g., removing an arrow from the scene). 
     * 
     * @param position position of vector
     * @param components components of vector
     * @param style style of vector
     * @param scale scale of vector; defaults to 1.0
     * @return std::pair<int, MxArrowData*> id of arrow according to the renderer and arrow
     */
    std::pair<int, MxArrowData*> addArrow(const MxVector3f &position, 
                                          const MxVector3f &components, 
                                          const MxStyle &style, 
                                          const float &scale=1.0);

    /**
     * @brief Removes a vector visualization specification. 
     * 
     * The removed pointer is only forgotten. The client is 
     * responsible for clearing the underlying data. 
     * 
     * @param arrowId id of arrow according to the renderer
     * @return HRESULT 
     */
    HRESULT removeArrow(const int &arrowId);

    /**
     * @brief Gets a vector visualization specification. 
     * 
     * @param arrowId id of arrow according to the renderer
     * @return MxArrowData* 
     */
    MxArrowData *getArrow(const int &arrowId);

    /**
     * @brief Gets the global instance of the renderer. 
     * 
     * Cannot be used until the universe renderer has been initialized. 
     * 
     * @return MxArrowRenderer* 
     */
    static MxArrowRenderer *get();

private:

    int _arrowDetail = 10;
    
    std::vector<Magnum::Vector4> _clipPlanes;

    Magnum::GL::Buffer _bufferHead{Corrade::Containers::NoCreate};
    Magnum::GL::Buffer _bufferCylinder{Corrade::Containers::NoCreate};
    Magnum::GL::Mesh _meshHead{Corrade::Containers::NoCreate};
    Magnum::GL::Mesh _meshCylinder{Corrade::Containers::NoCreate};
    Magnum::Shaders::MxPhong _shader{Corrade::Containers::NoCreate};

    /**
     * @brief Get the next data id
     * 
     * @return int 
     */
    int nextDataId();

};

#endif // SRC_RENDERING_MXARROWRENDERER_H_