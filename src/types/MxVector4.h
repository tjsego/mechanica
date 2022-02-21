/**
 * @file MxVector4.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines basic vector template wrap of Vector4 from Magnum/Math
 * @date 2021-07-16
 * 
 */
#ifndef _SRC_TYPES_MXVECTOR4_H_
#define _SRC_TYPES_MXVECTOR4_H_

#include "MxVector3.h"
#include <Magnum/Math/Vector4.h>

namespace mx { namespace type {

template<typename T>
class MxVector4 : public Magnum::Math::Vector4<T> {
    public:
        template<class U = T, typename std::enable_if<std::is_floating_point<U>::value, bool>::type = true>
        static MxVector4<T> planeEquation(const MxVector3<T> &normal, const MxVector3<T> &point) {
            return Magnum::Math::planeEquation(normal, point);
        }
        template<class U = T, typename std::enable_if<std::is_floating_point<U>::value, bool>::type = true>
        static MxVector4<T> planeEquation(const MxVector3<T>& p0, const MxVector3<T>& p1, const MxVector3<T>& p2) {
            return Magnum::Math::planeEquation(p0, p1, p2);
        }

        constexpr MxVector4() noexcept: Magnum::Math::Vector4<T>() {}

        constexpr explicit MxVector4(T value) noexcept: Magnum::Math::Vector4<T>(value) {}

        constexpr MxVector4(T x, T y, T z, T w) noexcept: Magnum::Math::Vector4<T>(x, y, z, w) {}

        constexpr MxVector4(const MxVector3<T>& xyz, T w) noexcept: Magnum::Math::Vector4<T>(xyz[0], xyz[1], xyz[2], w) {}

        template<class U> constexpr explicit MxVector4(const MxVector4<U>& other) noexcept: Magnum::Math::Vector4<T>(other) {}

        T& x() { return Magnum::Math::Vector4<T>::x(); }
        T& y() { return Magnum::Math::Vector4<T>::y(); }
        T& z() { return Magnum::Math::Vector4<T>::z(); }
        T& w() { return Magnum::Math::Vector4<T>::w(); }

        constexpr T x() const { return Magnum::Math::Vector4<T>::x(); }
        constexpr T y() const { return Magnum::Math::Vector4<T>::y(); }
        constexpr T z() const { return Magnum::Math::Vector4<T>::z(); }
        constexpr T w() const { return Magnum::Math::Vector4<T>::w(); }

        T& r() { return Magnum::Math::Vector4<T>::r(); }
        T& g() { return Magnum::Math::Vector4<T>::g(); }
        T& b() { return Magnum::Math::Vector4<T>::b(); }
        T& a() { return Magnum::Math::Vector4<T>::a(); }

        constexpr T r() const { return Magnum::Math::Vector4<T>::r(); }
        constexpr T g() const { return Magnum::Math::Vector4<T>::g(); }
        constexpr T b() const { return Magnum::Math::Vector4<T>::b(); }
        constexpr T a() const { return Magnum::Math::Vector4<T>::a(); }

        MxVector3<T>& xyz() { return MxVector3<T>::from(Magnum::Math::Vector4<T>::data()); }
        constexpr const MxVector3<T> xyz() const {
            return {Magnum::Math::Vector4<T>::_data[0], Magnum::Math::Vector4<T>::_data[1], Magnum::Math::Vector4<T>::_data[2]};
        }

        MxVector3<T>& rgb() { return MxVector3<T>::from(Magnum::Math::Vector4<T>::data()); }
        constexpr const MxVector3<T> rgb() const {
            return {Magnum::Math::Vector4<T>::_data[0], Magnum::Math::Vector4<T>::_data[1], Magnum::Math::Vector4<T>::_data[2]};
        }

        MxVector2<T>& xy() { return MxVector2<T>::from(Magnum::Math::Vector4<T>::data()); }
        constexpr const MxVector2<T> xy() const {
            return {Magnum::Math::Vector4<T>::_data[0], Magnum::Math::Vector4<T>::_data[1]};
        }

        template<class U = T, typename std::enable_if<std::is_floating_point<U>::value, bool>::type = true>
        T distance(const MxVector3<T> &point) const {
            return Magnum::Math::Distance::pointPlane(point, *this);
        }
        template<class U = T, typename std::enable_if<std::is_floating_point<U>::value, bool>::type = true>
        T distanceScaled(const MxVector3<T> &point) const {
            return Magnum::Math::Distance::pointPlaneScaled(point, *this);
        }

        MAGNUM_BASE_VECTOR_CAST_METHODS(4, MxVector4, Magnum::Math::Vector4)

        REVISED_MAGNUM_VECTOR_SUBCLASS_IMPLEMENTATION(4, MxVector4, Magnum::Math::Vector4)

        #ifdef SWIGPYTHON
        SWIGPYTHON_MAGNUM_VECTOR_SUBCLASS_IMPLEMENTATION(4, MxVector4)
        #endif
};

}}

MXVECTOR_IMPL_OSTREAM(mx::type::MxVector4)

#endif // _SRC_TYPES_MXVECTOR4_H_