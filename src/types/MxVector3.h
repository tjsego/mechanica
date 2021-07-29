/**
 * @file MxVector3.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines basic vector template wrap of Vector3 from Magnum/Math
 * @date 2021-07-16
 * 
 */
#ifndef _SRC_TYPES_MXVECTOR3_H_
#define _SRC_TYPES_MXVECTOR3_H_

#include "MxVector2.h"

#include <Magnum/Math/Vector3.h>
#include <Magnum/Math/Distance.h>

namespace mx { namespace type {

template<typename T>
class MxVector3 : public Magnum::Math::Vector3<T> {
    public:
        constexpr static MxVector3<T> xAxis(T length = T(1)) { return (MxVector3<T>)Magnum::Math::Vector3<T>::xAxis(length); }

        constexpr static MxVector3<T> yAxis(T length = T(1)) { return (MxVector3<T>)Magnum::Math::Vector3<T>::yAxis(length); }

        constexpr static MxVector3<T> zAxis(T length = T(1)) { return (MxVector3<T>)Magnum::Math::Vector3<T>::zAxis(length); }

        constexpr static MxVector3<T> xScale(T scale) { return (MxVector3<T>)Magnum::Math::Vector3<T>::xScale(scale); }

        constexpr static MxVector3<T> yScale(T scale) { return (MxVector3<T>)Magnum::Math::Vector3<T>::yScale(scale); }

        constexpr static MxVector3<T> zScale(T scale) { return (MxVector3<T>)Magnum::Math::Vector3<T>::zScale(scale); }

        constexpr MxVector3() noexcept: Magnum::Math::Vector3<T>() {}

        constexpr explicit MxVector3(T value) noexcept: Magnum::Math::Vector3<T>(value) {}

        constexpr MxVector3(T x, T y, T z) noexcept: Magnum::Math::Vector3<T>(x, y, z) {}

        constexpr MxVector3(const MxVector2<T>& xy, T z) noexcept: Magnum::Math::Vector3<T>(xy[0], xy[1], z) {}

        template<class U> constexpr explicit MxVector3(const MxVector3<U>& other) noexcept: Magnum::Math::Vector3<T>(other) {}

        T& x() { return Magnum::Math::Vector3<T>::x(); }
        T& y() { return Magnum::Math::Vector3<T>::y(); }
        T& z() { return Magnum::Math::Vector3<T>::z(); }

        constexpr T x() const { return Magnum::Math::Vector3<T>::x(); }
        constexpr T y() const { return Magnum::Math::Vector3<T>::y(); }
        constexpr T z() const { return Magnum::Math::Vector3<T>::z(); }

        T& r() { return Magnum::Math::Vector3<T>::r(); }
        T& g() { return Magnum::Math::Vector3<T>::g(); }
        T& b() { return Magnum::Math::Vector3<T>::b(); }

        constexpr T r() const { return Magnum::Math::Vector3<T>::r(); }
        constexpr T g() const { return Magnum::Math::Vector3<T>::g(); }
        constexpr T b() const { return Magnum::Math::Vector3<T>::b(); }

        MxVector2<T>& xy() { return MxVector2<T>::from(Magnum::Math::Vector3<T>::data()); }
        constexpr const MxVector2<T> xy() const {
            return {Magnum::Math::Vector3<T>::_data[0], Magnum::Math::Vector3<T>::_data[1]};
        }

        template<class U = T, typename std::enable_if<std::is_floating_point<U>::value, bool>::type = true>
        T distance(const MxVector3<T> &lineStartPt, const MxVector3<T> &lineEndPt) {
            return Magnum::Math::Distance::lineSegmentPoint(lineStartPt, lineEndPt, *this);
        }

        MAGNUM_BASE_VECTOR_CAST_METHODS(3, MxVector3, Magnum::Math::Vector3)

        REVISED_MAGNUM_VECTOR_SUBCLASS_IMPLEMENTATION(3, MxVector3, Magnum::Math::Vector3)

        #ifdef SWIGPYTHON
        SWIGPYTHON_MAGNUM_VECTOR_SUBCLASS_IMPLEMENTATION(3, MxVector3)
        #endif
};

}}

MXVECTOR_IMPL_OSTREAM(mx::type::MxVector3)

#endif // _SRC_TYPES_MXVECTOR3_H_