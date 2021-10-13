/**
 * @file MxQuaternion.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines basic quaternion template wrap of Quaternion from Magnum/Math
 * @date 2021-07-18
 * 
 */
#ifndef _SRC_TYPES_MXQUATERNION_H_
#define _SRC_TYPES_MXQUATERNION_H_

#include <Magnum/Math/Quaternion.h>

#include "MxVector3.h"
#include "MxMatrix3.h"

namespace mx { namespace type {

template<class T> using Quaternion = Magnum::Math::Quaternion<T>;

template<class T>
class MxQuaternion : public Quaternion<T> {
    public:
        static MxQuaternion<T> rotation(T angle, const MxVector3<T>& normalizedAxis) {
            return Quaternion<T>::rotation(Magnum::Math::Rad<T>(angle), normalizedAxis);
        }
        static MxQuaternion<T> fromMatrix(const MxMatrix3<T>& matrix) {
            return Quaternion<T>::fromMatrix((const Magnum::Math::Matrix<3, T>&)matrix);
        }

        constexpr MxQuaternion() noexcept: Quaternion<T>() {}
        constexpr MxQuaternion(const MxVector3<T>& vector, T scalar) noexcept: Quaternion<T>(vector, scalar) {}
        constexpr explicit MxQuaternion(const MxVector3<T>& vector) noexcept: Quaternion<T>(vector) {}
        template<class U> constexpr explicit MxQuaternion(const MxQuaternion<U>& other) noexcept: Quaternion<T>(other) {}

        MxQuaternion(const Quaternion<T>& other) noexcept: Quaternion<T>(other) {}
        MxQuaternion(const Quaternion<T>* other) noexcept: Quaternion<T>(*other) {}
        operator Quaternion<T>*() { return static_cast<Quaternion<T>>(this); }
        operator const Quaternion<T>*() { return static_cast<const Quaternion<T>>(this); }
        operator Quaternion<T>&() const { return *static_cast<Quaternion<T>>(this); }
        operator const Quaternion<T>&() const { return *static_cast<const Quaternion<T>>(this); }

        T* data() { return Quaternion<T>::data(); }
        constexpr const T* data() const { return Quaternion<T>::data(); }
        bool operator==(const MxQuaternion<T>& other) const { return Quaternion<T>::operator==(other); }
        bool operator!=(const MxQuaternion<T>& other) const { return Quaternion<T>::operator!=(other); }
        bool isNormalized() const { return Quaternion<T>::isNormalized(); }
        #ifndef SWIGPYTHON
        MxVector3<T>& vector() { return (MxVector3<T>&)Quaternion<T>::vector(); }
        T& scalar() { return Quaternion<T>::scalar(); }
        #endif
        constexpr const MxVector3<T> vector() const { return Quaternion<T>::vector(); }
        constexpr T scalar() const { return Quaternion<T>::scalar(); }
        T angle() const { return T(Quaternion<T>::angle()); }
        T angle(const MxQuaternion& other) const { return T(Magnum::Math::angle(this->normalized(), other.normalized())); }
        MxVector3<T> axis() const { return Quaternion<T>::axis(); }
        MxMatrix3<T> toMatrix() const { return Quaternion<T>::toMatrix(); }
        MxVector3<T> toEuler() const {
            auto v = Quaternion<T>::toEuler();
            return MxVector3<T>(T(v[0]), T(v[1]), T(v[2]));;
        }
        MxQuaternion<T> operator-() const { return Quaternion<T>::operator-(); }
        MxQuaternion<T>& operator+=(const MxQuaternion<T>& other) { return (MxQuaternion<T>&)Quaternion<T>::operator+=(other); }
        MxQuaternion<T> operator+(const MxQuaternion<T>& other) const { return Quaternion<T>::operator+(other); }
        MxQuaternion<T>& operator-=(const MxQuaternion<T>& other) { return (MxQuaternion<T>&)Quaternion<T>::operator-=(other); }
        MxQuaternion<T> operator-(const MxQuaternion<T>& other) const { return Quaternion<T>::operator-(other); }
        MxQuaternion<T>& operator*=(T scalar) { return (MxQuaternion<T>&)Quaternion<T>::operator*=(scalar); }
        MxQuaternion<T> operator*(T scalar) const { return Quaternion<T>::operator*(scalar); }
        MxQuaternion<T>& operator/=(T scalar) { return (MxQuaternion<T>&)Quaternion<T>::operator/=(scalar); }
        MxQuaternion<T> operator/(T scalar) const { return Quaternion<T>::operator/(scalar); }
        MxQuaternion<T> operator*(const Quaternion<T>& other) const { return Quaternion<T>::operator*(other); }
        T dot() const { return Quaternion<T>::dot(); }
        T length() const { return Quaternion<T>::length(); }
        MxQuaternion<T> normalized() const { return Quaternion<T>::normalized(); }
        MxQuaternion<T> conjugated() const { return Quaternion<T>::conjugated(); }
        MxQuaternion<T> inverted() const { return Quaternion<T>::inverted(); }
        MxQuaternion<T> invertedNormalized() const { return Quaternion<T>::invertedNormalized(); }
        MxVector3<T> transformVector(const MxVector3<T>& vector) const { return Quaternion<T>::transformVector(vector); }
        MxVector3<T> transformVectorNormalized(const MxVector3<T>& vector) const { return Quaternion<T>::transformVectorNormalized(vector); }

        operator std::vector<T>&() const {
            std::vector<T>& v = vector();
            std::vector<T> *result = new std::vector<T>(v.begin(), v.end());
            result->push_back(scalar());
            return *result;
        }
        #ifdef SWIGPYTHON
        constexpr explicit MxQuaternion(const std::vector<T>& vector, T scalar) noexcept: Quaternion<T>(MxVector3<T>(vector), scalar) {}
        std::vector<T>& asVector() { 
            std::vector<T> *result = new std::vector<T>(*this);
            return *result;
        }
        #endif // SWIGPYTHON
};

}}

template<typename T>
inline std::ostream& operator<<(std::ostream& os, const mx::type::MxQuaternion<T>& q)
{
    auto vec = q.vector();
    os << std::string("{") << vec[0];
    for(int i = 1; i < vec.length(); ++i) os << std::string(",") << vec[i];
    os << "," << q.scalar() << std::string("}");
    return os;
}

#endif // _SRC_TYPES_MXQUATERNION_H_