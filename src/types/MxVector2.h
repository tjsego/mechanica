/**
 * @file MxVector2.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines basic vector template wrap of Vector2 from Magnum/Math
 * @date 2021-07-16
 * 
 */
#ifndef _SRC_TYPES_MXVECTOR2_H_
#define _SRC_TYPES_MXVECTOR2_H_

#include "MxVector.h"

#include <Magnum/Math/Vector2.h>
#include <Magnum/Math/Distance.h>

namespace mx { namespace type {

template<typename T>
class MxVector2 : public Magnum::Math::Vector2<T> {
    public:
        constexpr static MxVector2<T> xAxis(T length = T(1)) { return (MxVector2<T>)Magnum::Math::Vector2<T>::xAxis(length); }

        constexpr static MxVector2<T> yAxis(T length = T(1)) { return (MxVector2<T>)Magnum::Math::Vector2<T>::yAxis(length); }

        constexpr static MxVector2<T> xScale(T scale) { return (MxVector2<T>)Magnum::Math::Vector2<T>::xScale(scale); }

        constexpr static MxVector2<T> yScale(T scale) { return (MxVector2<T>)Magnum::Math::Vector2<T>::yScale(scale); }

        constexpr MxVector2() noexcept: Magnum::Math::Vector2<T>() {}

        constexpr explicit MxVector2(T value) noexcept: Magnum::Math::Vector2<T>(value) {}

        constexpr MxVector2(T x, T y) noexcept: Magnum::Math::Vector2<T>(x, y) {}

        template<class U> constexpr explicit MxVector2(const MxVector2<U>& other) noexcept: Magnum::Math::Vector2<T>(other) {}

        T& x() { return Magnum::Math::Vector2<T>::x(); }
        T& y() { return Magnum::Math::Vector2<T>::y(); }

        constexpr T x() const { return Magnum::Math::Vector2<T>::x(); }
        constexpr T y() const { return Magnum::Math::Vector2<T>::y(); }

        template<class U = T, typename std::enable_if<std::is_floating_point<U>::value, bool>::type = true>
        T distance(const MxVector2<T> &lineStartPt, const MxVector2<T> &lineEndPt) {
            return Magnum::Math::Distance::lineSegmentPoint(lineStartPt, lineEndPt, *this);
        }

        MAGNUM_BASE_VECTOR_CAST_METHODS(2, MxVector2, Magnum::Math::Vector2)

        REVISED_MAGNUM_VECTOR_SUBCLASS_IMPLEMENTATION(2, MxVector2, Magnum::Math::Vector2)

        #ifdef SWIGPYTHON
        SWIGPYTHON_MAGNUM_VECTOR_SUBCLASS_IMPLEMENTATION(2, MxVector2)
        #endif
};

}}

MXVECTOR_IMPL_OSTREAM(mx::type::MxVector2)

#endif // _SRC_TYPES_MXVECTOR2_H_
