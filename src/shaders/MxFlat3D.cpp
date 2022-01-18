/*
    This file is part of Magnum.

    Copyright © 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019,
                2020, 2021 Vladimír Vondruš <mosra@centrum.cz>

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included
    in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.
*/

#include "MxFlat3D.h"

#include <Corrade/Containers/EnumSet.hpp>
#include <Corrade/Containers/Reference.h>
#include <Corrade/Utility/FormatStl.h>
#include <Corrade/Utility/Resource.h>

#include "Magnum/GL/Context.h"
#include "Magnum/GL/Extensions.h"
#include "Magnum/GL/Shader.h"
#include "Magnum/GL/Texture.h"
#include "Magnum/Math/Color.h"
#include "Magnum/Math/Matrix3.h"
#include "Magnum/Math/Matrix4.h"

#include "Magnum/Shaders/Implementation/CreateCompatibilityShader.h"

namespace Magnum { namespace Shaders {

namespace {
    enum: Int { TextureUnit = 0 };
}

MxFlat3D::MxFlat3D(const Flags flags, unsigned clipPlaneCount): 
    _flags(flags), 
    _clipPlaneCount{clipPlaneCount}
{
    CORRADE_ASSERT(!(flags & Flag::TextureTransformation) || (flags & Flag::Textured),
        "Shaders::Flat: texture transformation enabled but the shader is not textured", );

    Utility::Resource rs("MxMeshShaderProgram");

    #ifndef MAGNUM_TARGET_GLES
    const GL::Version version = GL::Context::current().supportedVersion({GL::Version::GL320, GL::Version::GL310, GL::Version::GL300, GL::Version::GL210});
    #else
    const GL::Version version = GL::Context::current().supportedVersion({GL::Version::GLES300, GL::Version::GLES200});
    #endif

    GL::Shader vert = Implementation::createCompatibilityShader(rs, version, GL::Shader::Type::Vertex);
    GL::Shader frag = Implementation::createCompatibilityShader(rs, version, GL::Shader::Type::Fragment);

    vert.addSource(flags & Flag::Textured ? "#define TEXTURED\n" : "")
        .addSource(flags & Flag::VertexColor ? "#define VERTEX_COLOR\n" : "")
        .addSource(flags & Flag::TextureTransformation ? "#define TEXTURE_TRANSFORMATION\n" : "")
        .addSource(Utility::formatString("#define CLIP_PLANE_COUNT {}\n", clipPlaneCount))
        .addSource(Utility::formatString("#define CLIP_PLANES_UNIFORM {}\n", _clipPlanesUniform))
        #ifndef MAGNUM_TARGET_GLES2
        .addSource(flags >= Flag::InstancedObjectId ? "#define INSTANCED_OBJECT_ID\n" : "")
        #endif
        .addSource(flags & Flag::InstancedTransformation ? "#define INSTANCED_TRANSFORMATION\n" : "")
        .addSource(flags >= Flag::InstancedTextureOffset ? "#define INSTANCED_TEXTURE_OFFSET\n" : "")
        .addSource(rs.get("generic.glsl"))
        .addSource(rs.get("MxFlat3D.vert"));
    frag.addSource(flags & Flag::Textured ? "#define TEXTURED\n" : "")
        .addSource(flags & Flag::AlphaMask ? "#define ALPHA_MASK\n" : "")
        .addSource(flags & Flag::VertexColor ? "#define VERTEX_COLOR\n" : "")
        #ifndef MAGNUM_TARGET_GLES2
        .addSource(flags & Flag::ObjectId ? "#define OBJECT_ID\n" : "")
        .addSource(flags >= Flag::InstancedObjectId ? "#define INSTANCED_OBJECT_ID\n" : "")
        #endif
        .addSource(rs.get("generic.glsl"))
        .addSource(rs.get("MxFlat3D.frag"));

    CORRADE_INTERNAL_ASSERT_OUTPUT(GL::Shader::compile({vert, frag}));

    attachShaders({vert, frag});

    /* ES3 has this done in the shader directly and doesn't even provide
       bindFragmentDataLocation() */
    #if !defined(MAGNUM_TARGET_GLES) || defined(MAGNUM_TARGET_GLES2)
    #ifndef MAGNUM_TARGET_GLES
    if(!GL::Context::current().isExtensionSupported<GL::Extensions::ARB::explicit_attrib_location>(version))
    #endif
    {
        bindAttributeLocation(Position::Location, "position");
        if(flags & Flag::Textured)
            bindAttributeLocation(TextureCoordinates::Location, "textureCoordinates");
        if(flags & Flag::VertexColor)
            bindAttributeLocation(Color3::Location, "vertexColor"); /* Color4 is the same */
        #ifndef MAGNUM_TARGET_GLES2
        if(flags & Flag::ObjectId) {
            bindFragmentDataLocation(ColorOutput, "color");
            bindFragmentDataLocation(ObjectIdOutput, "objectId");
        }
        if(flags >= Flag::InstancedObjectId)
            bindAttributeLocation(ObjectId::Location, "instanceObjectId");
        #endif
        if(flags & Flag::InstancedTransformation)
            bindAttributeLocation(TransformationMatrix::Location, "instancedTransformationMatrix");
        if(flags >= Flag::InstancedTextureOffset)
            bindAttributeLocation(TextureOffset::Location, "instancedTextureOffset");
    }
    #endif

    CORRADE_INTERNAL_ASSERT_OUTPUT(link());

    #ifndef MAGNUM_TARGET_GLES
    if(!GL::Context::current().isExtensionSupported<GL::Extensions::ARB::explicit_uniform_location>(version))
    #endif
    {
        _transformationProjectionMatrixUniform = uniformLocation("transformationProjectionMatrix");
        if(flags & Flag::TextureTransformation)
            _textureMatrixUniform = uniformLocation("textureMatrix");
        _colorUniform = uniformLocation("color");
        if(flags & Flag::AlphaMask) _alphaMaskUniform = uniformLocation("alphaMask");
        #ifndef MAGNUM_TARGET_GLES2
        if(flags & Flag::ObjectId) _objectIdUniform = uniformLocation("objectId");
        #endif
        
        if(clipPlaneCount) {
            _clipPlanesUniform = uniformLocation("clipPlanes");
        }
    }

    #ifndef MAGNUM_TARGET_GLES
    if(!GL::Context::current().isExtensionSupported<GL::Extensions::ARB::shading_language_420pack>(version))
    #endif
    {
        if(flags & Flag::Textured) setUniform(uniformLocation("textureData"), TextureUnit);
    }

    /* Set defaults in OpenGL ES (for desktop they are set in shader code itself) */
    #ifdef MAGNUM_TARGET_GLES
    setTransformationProjectionMatrix(MatrixTypeFor<3, Float>{Math::IdentityInit});
    if(flags & Flag::TextureTransformation)
        setTextureMatrix(Matrix3{Math::IdentityInit});
    setColor(Magnum::Color4{1.0f});
    if(flags & Flag::AlphaMask) setAlphaMask(0.5f);
    /* Object ID is zero by default */
    #endif
}

MxFlat3D& MxFlat3D::setTransformationProjectionMatrix(const MatrixTypeFor<3, Float>& matrix) {
    setUniform(_transformationProjectionMatrixUniform, matrix);
    return *this;
}

MxFlat3D& MxFlat3D::setTextureMatrix(const Matrix3& matrix) {
    CORRADE_ASSERT(_flags & Flag::TextureTransformation,
        "Shaders::MxFlat3D::setTextureMatrix(): the shader was not created with texture transformation enabled", *this);
    setUniform(_textureMatrixUniform, matrix);
    return *this;
}

MxFlat3D& MxFlat3D::setColor(const Magnum::Color4& color) {
    setUniform(_colorUniform, color);
    return *this;
}

MxFlat3D& MxFlat3D::bindTexture(GL::Texture2D& texture) {
    CORRADE_ASSERT(_flags & Flag::Textured,
        "Shaders::MxFlat3D::bindTexture(): the shader was not created with texturing enabled", *this);
    texture.bind(TextureUnit);
    return *this;
}

MxFlat3D& MxFlat3D::setAlphaMask(Float mask) {
    CORRADE_ASSERT(_flags & Flag::AlphaMask,
        "Shaders::MxFlat3D::setAlphaMask(): the shader was not created with alpha mask enabled", *this);
    setUniform(_alphaMaskUniform, mask);
    return *this;
}

MxFlat3D& MxFlat3D::setclipPlaneEquation(const UnsignedInt id, const Vector4& planeEquation) {
    CORRADE_ASSERT(id < _clipPlaneCount,
        "Shaders::MxFlat3D::setclipPlaneEquation(): plane ID" << id << "is out of bounds for" << _clipPlaneCount << "planes", *this);
    setUniform(_clipPlanesUniform + id, planeEquation);
    return *this;
}

#ifndef MAGNUM_TARGET_GLES2
MxFlat3D& MxFlat3D::setObjectId(UnsignedInt id) {
    CORRADE_ASSERT(_flags & Flag::ObjectId,
        "Shaders::MxFlat3D::setObjectId(): the shader was not created with object ID enabled", *this);
    setUniform(_objectIdUniform, id);
    return *this;
}
#endif

Debug& operator<<(Debug& debug, const MxFlat3D::Flag value) {
    debug << "Shaders::MxFlat3D::Flag" << Debug::nospace;

    switch(value) {
        /* LCOV_EXCL_START */
        #define _c(v) case MxFlat3D::Flag::v: return debug << "::" #v;
        _c(Textured)
        _c(AlphaMask)
        _c(VertexColor)
        _c(TextureTransformation)
        #ifndef MAGNUM_TARGET_GLES2
        _c(ObjectId)
        _c(InstancedObjectId)
        #endif
        _c(InstancedTransformation)
        _c(InstancedTextureOffset)
        #undef _c
        /* LCOV_EXCL_STOP */
    }

    return debug << "(" << Debug::nospace << reinterpret_cast<void*>(UnsignedByte(value)) << Debug::nospace << ")";
}

Debug& operator<<(Debug& debug, const MxFlat3D::Flags value) {
    return Containers::enumSetDebugOutput(debug, value, "Shaders::MxFlat3D::Flags{}", {
        MxFlat3D::Flag::Textured,
        MxFlat3D::Flag::AlphaMask,
        MxFlat3D::Flag::VertexColor,
        MxFlat3D::Flag::InstancedTextureOffset, /* Superset of TextureTransformation */
        MxFlat3D::Flag::TextureTransformation,
        #ifndef MAGNUM_TARGET_GLES2
        MxFlat3D::Flag::InstancedObjectId, /* Superset of ObjectId */
        MxFlat3D::Flag::ObjectId,
        #endif
        MxFlat3D::Flag::InstancedTransformation});
}

}}