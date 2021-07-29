/**
 * @file MxMatrix3.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines basic matrix template wrap of Matrix3 from Magnum/Math
 * @date 2021-07-16
 * 
 */
#ifndef _SRC_TYPES_MXMATRIX3_H_
#define _SRC_TYPES_MXMATRIX3_H_

#include "MxMatrix.h"
#include "MxVector3.h"

#include <Magnum/Math/Matrix3.h>

namespace mx { namespace type {

template<class T>
class MxMatrix3 : public Magnum::Math::Matrix3<T> {
    public:
        static MxMatrix3<T> rotation(T angle) { return (MxMatrix3<T>)Magnum::Math::Matrix3<T>::rotation(Magnum::Math::Rad<T>(angle)); }

        constexpr static MxMatrix3<T> shearingX(T amount) { return (MxMatrix3<T>)Magnum::Math::Matrix3<T>::shearingX(amount); }

        constexpr static MxMatrix3<T> shearingY(T amount) { return (MxMatrix3<T>)Magnum::Math::Matrix3<T>::shearingY(amount); }

        constexpr MxMatrix3() noexcept: Magnum::Math::Matrix3<T>() {}
        
        constexpr MxMatrix3(const MxVector3<T>& first, const MxVector3<T>& second, const MxVector3<T>& third) noexcept: 
            Magnum::Math::Matrix3<T>(Magnum::Math::Vector3<T>(first), Magnum::Math::Vector3<T>(second), Magnum::Math::Vector3<T>(third)) {}

        constexpr explicit MxMatrix3(T value) noexcept: Magnum::Math::Matrix3<T>(value) {}

        template<class U> constexpr explicit MxMatrix3(const MxMatrix3<U>& other) noexcept: Magnum::Math::Matrix3<T>((Magnum::Math::Matrix3<U>)other) {}

        template<std::size_t otherSize> constexpr explicit MxMatrix3(const MxMatrix<otherSize, T>& other) noexcept: Magnum::Math::Matrix3<T>{(Matrix<otherSize, T>)other} {}

        bool isRigidTransformation() const { return Magnum::Math::Matrix3<T>::isRigidTransformation(); }

        MxMatrix3<T> invertedRigid() const { return (MxMatrix3<T>)Magnum::Math::Matrix3<T>::invertedRigid(); }

        // Implementing this constructor here for now, as the template is too nasty to be useful in a macro

        MxMatrix3(const std::vector<T> &v1, const std::vector<T> &v2, const std::vector<T> &v3) : 
            MxMatrix3<T>(MxVector3<T>(v1), MxVector3<T>(v2), MxVector3<T>(v3)) {}

        MAGNUM_BASE_MATRIX_CAST_METHODS(3, MxMatrix3, Magnum::Math::Matrix3)

        REVISED_MAGNUM_MATRIX_SUBCLASS_IMPLEMENTATION(3, MxMatrix3, Magnum::Math::Matrix3, MxVector3)

        #ifdef SWIGPYTHON
        SWIGPYTHON_MAGNUM_MATRIX_SUBCLASS_IMPLEMENTATION(3, MxMatrix3, MxVector3)
        #endif

};

}}

MXMATRIX_IMPL_OSTREAM(mx::type::MxMatrix3)

#endif // _SRC_TYPES_MXMATRIX3_H_
