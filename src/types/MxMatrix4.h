/**
 * @file MxMatrix4.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines basic matrix template wrap of Matrix4 from Magnum/Math
 * @date 2021-07-17
 * 
 */
#ifndef _SRC_TYPES_MXMATRIX4_H_
#define _SRC_TYPES_MXMATRIX4_H_

#include "MxMatrix3.h"
#include "MxVector4.h"

#include <Magnum/Math/Matrix4.h>

namespace mx { namespace type {

template<class T>
class MxMatrix4 : public Magnum::Math::Matrix4<T> {
    public:
        constexpr static MxMatrix4<T> translation(const MxVector3<T>& vector) {
            return Magnum::Math::Matrix4<T>::translation(vector);
        }
        constexpr static MxMatrix4<T> scaling(const MxVector3<T>& vector) {
            return Magnum::Math::Matrix4<T>::scaling(vector);
        }
        static MxMatrix4<T> rotation(T angle, const MxVector3<T>& normalizedAxis) {
            return Magnum::Math::Matrix4<T>::rotation(Magnum::Math::Rad<T>(angle), normalizedAxis);
        }
        static MxMatrix4<T> rotationX(T angle) {
            return Magnum::Math::Matrix4<T>::rotationX(Magnum::Math::Rad<T>(angle));
        }
        static MxMatrix4<T> rotationY(T angle) {
            return Magnum::Math::Matrix4<T>::rotationY(Magnum::Math::Rad<T>(angle));
        }
        static MxMatrix4<T> rotationZ(T angle) {
            return Magnum::Math::Matrix4<T>::rotationZ(Magnum::Math::Rad<T>(angle));
        }
        static MxMatrix4<T> reflection(const MxVector3<T>& normal) {
            return Magnum::Math::Matrix4<T>::reflection(normal);
        }
        constexpr static MxMatrix4<T> shearingXY(T amountX, T amountY) {
            return Magnum::Math::Matrix4<T>::shearingXY(amountX, amountY);
        }
        constexpr static MxMatrix4<T> shearingXZ(T amountX, T amountZ) {
            return Magnum::Math::Matrix4<T>::shearingXZ(amountX, amountZ);
        }
        constexpr static MxMatrix4<T> shearingYZ(T amountY, T amountZ) {
            return Magnum::Math::Matrix4<T>::shearingYZ(amountY, amountZ);
        }
        static MxMatrix4<T> orthographicProjection(const MxVector2<T>& size, T near, T far) {
            return Magnum::Math::Matrix4<T>::orthographicProjection(size, near, far);
        }
        static MxMatrix4<T> perspectiveProjection(const MxVector2<T>& size, T near, T far) {
            return Magnum::Math::Matrix4<T>::perspectiveProjection(size, near, far);
        }
        static MxMatrix4<T> perspectiveProjection(T fov, T aspectRatio, T near, T far) {
            return Magnum::Math::Matrix4<T>::perspectiveProjection(Magnum::Math::Rad<T>(fov), aspectRatio, near, far);
        }
        static MxMatrix4<T> perspectiveProjection(const MxVector2<T>& bottomLeft, const MxVector2<T>& topRight, T near, T far) {
            return Magnum::Math::Matrix4<T>::perspectiveProjection(bottomLeft, topRight, near, far);
        }
        static MxMatrix4<T> lookAt(const MxVector3<T>& eye, const MxVector3<T>& target, const MxVector3<T>& up) {
            return Magnum::Math::Matrix4<T>::lookAt(eye, target, up);
        }
        constexpr static MxMatrix4<T> from(const MxMatrix3<T>& rotationScaling, const MxVector3<T>& translation) {
            return Magnum::Math::Matrix4<T>::from((const Magnum::Math::Matrix3<T>&)rotationScaling, translation);
        }

        constexpr MxMatrix4() noexcept: Magnum::Math::Matrix4<T>() {}
        constexpr MxMatrix4(const MxVector4<T>& first, const MxVector4<T>& second, const MxVector4<T>& third, const MxVector4<T>& fourth) noexcept: 
            Magnum::Math::Matrix4<T>(first, second, third, fourth) {}
        constexpr explicit MxMatrix4(T value) noexcept: Magnum::Math::Matrix4<T>{value} {}
        template<std::size_t otherSize> constexpr explicit MxMatrix4(const MxMatrix<otherSize, T>& other) noexcept: Magnum::Math::Matrix4<T>{other} {}
        constexpr MxMatrix4(const MxMatrix4<T>& other) noexcept: Magnum::Math::Matrix4<T>(other) {}

        bool isRigidTransformation() const { return Magnum::Math::Matrix4<T>::isRigidTransformation(); }
        constexpr MxMatrix3<T> rotationScaling() const { return Magnum::Math::Matrix4<T>::rotationScaling(); }
        MxMatrix3<T> rotationShear() const { return Magnum::Math::Matrix4<T>::rotationShear(); }
        MxMatrix3<T> rotation() const { return Magnum::Math::Matrix4<T>::rotation(); }
        MxMatrix3<T> rotationNormalized() const { return Magnum::Math::Matrix4<T>::rotationNormalized(); }
        MxVector3<T> scalingSquared() const { return Magnum::Math::Matrix4<T>::scalingSquared(); }
        MxVector3<T> scaling() const { return Magnum::Math::Matrix4<T>::scaling(); }
        T uniformScalingSquared() const { return Magnum::Math::Matrix4<T>::uniformScalingSquared(); }
        T uniformScaling() const { return Magnum::Math::Matrix4<T>::uniformScaling(); }
        MxMatrix3<T> normalMatrix() const { return Magnum::Math::Matrix4<T>::normalMatrix(); }
        MxVector3<T>& right() { return (MxVector3<T>&)Magnum::Math::Matrix4<T>::right(); }
        constexpr MxVector3<T> right() const { return Magnum::Math::Matrix4<T>::right(); }
        MxVector3<T>& up() { return (MxVector3<T>&)Magnum::Math::Matrix4<T>::up(); }
        constexpr MxVector3<T> up() const { return Magnum::Math::Matrix4<T>::up(); }
        MxVector3<T>& backward() { return (MxVector3<T>&)Magnum::Math::Matrix4<T>::backward(); }
        constexpr MxVector3<T> backward() const { return Magnum::Math::Matrix4<T>::backward(); }
        MxVector3<T>& translation() { return (MxVector3<T>&)Magnum::Math::Matrix4<T>::translation(); }
        constexpr MxVector3<T> translation() const { return Magnum::Math::Matrix4<T>::translation(); }
        MxMatrix4<T> invertedRigid() const { return Magnum::Math::Matrix4<T>::invertedRigid(); }
        MxVector3<T> transformVector(const MxVector3<T>& vector) const { return Magnum::Math::Matrix4<T>::transformVector(vector); }
        MxVector3<T> transformPoint(const MxVector3<T>& vector) const { return Magnum::Math::Matrix4<T>::transformPoint(vector); }

        MAGNUM_BASE_MATRIX_CAST_METHODS(4, MxMatrix4, Magnum::Math::Matrix4)

        REVISED_MAGNUM_MATRIX_SUBCLASS_IMPLEMENTATION(4, MxMatrix4, Magnum::Math::Matrix4, MxVector4)

        #ifdef SWIGPYTHON
        SWIGPYTHON_MAGNUM_MATRIX_SUBCLASS_IMPLEMENTATION(4, MxMatrix4, MxVector4)
        #endif

};

}}

MXMATRIX_IMPL_OSTREAM(mx::type::MxMatrix4)

#endif // _SRC_TYPES_MXMATRIX4_H_