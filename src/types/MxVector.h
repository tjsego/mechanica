/**
 * @file MxVector.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines basic vector template wrap of Vector from Magnum/Math
 * @date 2021-07-16
 * 
 */
#ifndef _SRC_TYPES_MXVECTOR_H_
#define _SRC_TYPES_MXVECTOR_H_

#include "../mx_error.h"

#include <Magnum/Magnum.h>
#include <Magnum/Math/Vector.h>

#include <string>
#include <vector>

namespace mx { namespace type {

template<std::size_t size, class T> using Vector = Magnum::Math::Vector<size, T>;

template<std::size_t size, class T> 
class MxVector : public Vector<size, T> {
    public:
        static MxVector<size, T>& from(T* data) { return *reinterpret_cast<MxVector<size, T>*>(&Vector<size, T>::from(data)); }

        static const MxVector<size, T>& from(const T* data) { return *reinterpret_cast<const MxVector<size, T>*>(&Vector<size, T>::from(data)); }

        template<std::size_t otherSize> constexpr static MxVector<size, T> pad(const MxVector<otherSize, T>& a, T value = T()) {
            return (MxVector<size, T>)Vector<size, T>::pad<otherSize(a, value);
        }

        MxVector() : Vector<size, T>() {}

        MxVector(const MxVector<size, T>&) = default;

        template<class ...U, class V = typename std::enable_if<sizeof...(U)+1 == size, T>::type> constexpr MxVector(T first, U... next) : Vector<size, T>(first, next...) {}

        T* data() { return Vector<size, T>::data(); }
        constexpr const T* data() const { return Vector<size, T>::data(); }

        T& operator[](std::size_t pos) { return Vector<size, T>::operator[](pos); }
        constexpr T operator[](std::size_t pos) const { return Vector<size, T>::operator[](pos); }

        bool operator==(const Vector<size, T>& other) const { return Vector<size, T>::operator==(other); }

        bool operator!=(const Vector<size, T>& other) const { return Vector<size, T>::operator!=(other); }

        MxVector<size, T>& operator+=(const MxVector<size, T>& other) {
            MxVector<size, T>::operator+=(other);
            return *this;
        }

        MxVector<size, T> operator+(const MxVector<size, T>& other) const { return (MxVector<size, T>)Vector<size, T>::operator+((Vector<size, T>)other); }

        MxVector<size, T>& operator-=(const MxVector<size, T>& other) {
            MxVector<size, T>::operator-=(other);
            return *this;
        }

        MxVector<size, T> operator-(const MxVector<size, T>& other) const { return (MxVector<size, T>)Vector<size, T>::operator-((Vector<size, T>)other); }

        MxVector<size, T>& operator*=(T scalar) {
            MxVector<size, T>::operator*=(scalar);
            return *this;
        }

        MxVector<size, T> operator*(T scalar) const { return MxVector<size, T>(Vector<size, T>::operator*(scalar)); }

        MxVector<size, T>& operator/=(T scalar) {
            Vector<size, T>::operator/=(scalar);
            return *this;
        }

        MxVector<size, T> operator/(T scalar) const { return MxVector<size, T>(Vector<size, T>::operator/(scalar)); }

        MxVector<size, T>& operator*=(const Vector<size, T>& other) { 
            Vector<size, T>::operator*=(other);
            return *this;
         }

        MxVector<size, T> operator*(const Vector<size, T>& other) const { return (MxVector<size, T>)Vector<size, T>::operator*((Vector<size, T>)other); }

        MxVector<size, T>& operator/=(const Vector<size, T>& other) { 
            Vector<size, T>::operator/=(other);
            return *this;
         }

        MxVector<size, T> operator/(const Vector<size, T>& other) const {
            return MxVector<size, T>(Vector<size, T>::operator/(other));
        }

        T dot() const { return Vector<size, T>::dot(); }

        T dot(const MxVector<size, T>& other) const { return Magnum::Math::dot(*this, other); }

        T length() const { return Vector<size, T>::length(); }

        template<class U = T> typename std::enable_if<std::is_floating_point<U>::value, T>::type
        lengthInverted() const { return Vector<size, T>::lengthInverted<U>(); }

        template<class U = T> typename std::enable_if<std::is_floating_point<U>::value, Vector<size, T>>::type
        normalized() const { return Vector<size, T>::normalized<U>(); }

        template<class U = T> typename std::enable_if<std::is_floating_point<U>::value, Vector<size, T>>::type
        resized(T length) const { return Vector<size, T>::resized<U>(length); }

        template<class U = T> typename std::enable_if<std::is_floating_point<U>::value, Vector<size, T>>::type
        projected(const Vector<size, T>& line) const { return Vector<size, T>::projected<U>(line); }

        template<class U = T> typename std::enable_if<std::is_floating_point<U>::value, Vector<size, T>>::type
        projectedOntoNormalized(const Vector<size, T>& line) const { return Vector<size, T>::projectedOntoNormalized<U>(line); }

        constexpr MxVector<size, T> flipped() const { return Vector<size, T>::flipped(); }

        T sum() const { return Vector<size, T>::sum(); }

        T product() const { return Vector<size, T>::product(); }

        T min() const { return Vector<size, T>::min(); }

        T max() const { return Vector<size, T>::max(); }

        std::pair<T, T> minmax() const { return Vector<size, T>::minmax(); }

        MxVector<size, T>(const Vector<size, T> &other) : Vector<size, T>() {
            for(int i = 0; i < other.Size; ++i) this->_data[i] = other[i];
        }

        operator Vector<size, T>*() { return static_cast<Vector<size, T>*>(this); }

        operator Vector<size, T>&() const { return *static_cast<Vector<size, T>*>(this); }

        #ifdef SWIGPYTHON
        T __getitem__(std::size_t i) { return this->operator[](i); }
        void __setitem__(std::size_t i, const T &val) { this->operator[](i) = val; }
        #endif

};

#define REVISED_MAGNUM_VECTOR_SUBCLASS_IMPLEMENTATION(size, Type, MagnumImplType)       \
    static Type<T>& from(T* data) {                                                     \
        return *reinterpret_cast<Type<T>*>(data);                                       \
    }                                                                                   \
    static const Type<T>& from(const T* data) {                                         \
        return *reinterpret_cast<const Type<T>*>(data);                                 \
    }                                                                                   \
    template<std::size_t otherSize>                                                     \
    constexpr static Type<T> pad(const Type<T>& a, T value = T()) {                     \
        return MagnumImplType<T>::pad(a, value);                                        \
    }                                                                                   \
                                                                                        \
    template<class U = T>                                                               \
    typename std::enable_if<std::is_signed<U>::value, Type<T>>::type                    \
    operator-() const {                                                                 \
        return MagnumImplType<T>::operator-();                                          \
    }                                                                                   \
    Type<T>& operator+=(const Type<T>& other) {                                         \
        MagnumImplType<T>::operator+=(other);                                           \
        return *this;                                                                   \
    }                                                                                   \
    Type<T> operator+(const Type<T>& other) const {                                     \
        return MagnumImplType<T>::operator+(other);                                     \
    }                                                                                   \
    Type<T>& operator-=(const Type<T>& other) {                                         \
        MagnumImplType<T>::operator-=(other);                                           \
        return *this;                                                                   \
    }                                                                                   \
    Type<T> operator-(const Type<T>& other) const {                                     \
        return MagnumImplType<T>::operator-(other);                                     \
    }                                                                                   \
    Type<T>& operator*=(T number) {                                                     \
        MagnumImplType<T>::operator*=(number);                                          \
        return *this;                                                                   \
    }                                                                                   \
    Type<T> operator*(T number) const {                                                 \
        return MagnumImplType<T>::operator*(number);                                    \
    }                                                                                   \
    Type<T>& operator/=(T number) {                                                     \
        MagnumImplType<T>::operator/=(number);                                          \
        return *this;                                                                   \
    }                                                                                   \
    Type<T> operator/(T number) const {                                                 \
        return MagnumImplType<T>::operator/(number);                                    \
    }                                                                                   \
    Type<T>& operator*=(const Type<T>& other) {                                         \
        MagnumImplType<T>::operator*=(other);                                           \
        return *this;                                                                   \
    }                                                                                   \
    Type<T> operator*(const Type<T>& other) const {                                     \
        return MagnumImplType<T>::operator*(other);                                     \
    }                                                                                   \
    Type<T>& operator/=(const Type<T>& other) {                                         \
        MagnumImplType<T>::operator/=(other);                                           \
        return *this;                                                                   \
    }                                                                                   \
    Type<T> operator/(const Type<T>& other) const {                                     \
        return MagnumImplType<T>::operator/(other);                                     \
    }                                                                                   \
                                                                                        \
    template<class U = T>                                                               \
    typename std::enable_if<std::is_floating_point<U>::value, T>::type                  \
    length() const { return MagnumImplType<T>::length(); }                              \
    template<class U = T>                                                               \
    typename std::enable_if<std::is_floating_point<U>::value, Type<T>>::type            \
    normalized() const { return MagnumImplType<T>::normalized(); }                      \
    template<class U = T>                                                               \
    typename std::enable_if<std::is_floating_point<U>::value, Type<T>>::type            \
    resized(T length) const {                                                           \
        return MagnumImplType<T>::resized(length);                                      \
    }                                                                                   \
    template<class U = T>                                                               \
    typename std::enable_if<std::is_floating_point<U>::value, Type<T>>::type            \
    projected(const Type<T>& other) const {                                             \
        return MagnumImplType<T>::projected(other);                                     \
    }                                                                                   \
    template<class U = T>                                                               \
    typename std::enable_if<std::is_floating_point<U>::value, Type<T>>::type            \
    projectedOntoNormalized(const Type<T>& other) const {                               \
        return MagnumImplType<T>::projectedOntoNormalized(other);                       \
    }                                                                                   \
    constexpr Type<T> flipped() const { return MagnumImplType<T>::flipped(); }          \
    T dot() const { return Magnum::Math::dot(*this, *this); }                           \
    T dot(const Type<T>& other) const { return Magnum::Math::dot(*this, other); }       \
    T& operator[](std::size_t pos) { return MagnumImplType<T>::operator[](pos); }       \
    constexpr T operator[](std::size_t pos) const {                                     \
        return MagnumImplType<T>::operator[](pos);                                      \
    }                                                                                   \

#define SWIGPYTHON_MAGNUM_VECTOR_SUBCLASS_IMPLEMENTATION(size, Type)                    \
    T _getitem(int i) {                                                                 \
        if(i < 0) return _getitem(size - 1 + i);                                        \
        return this->operator[](i);                                                     \
    }                                                                                   \
    void _setitem(int i, const T &val) {                                                \
        if(i < 0) return _setitem(size - 1 + i, val);                                   \
        this->operator[](i) = val;                                                      \
    }                                                                                   \
    int __len__() { return size; }                                                      \
    static Type<T>& fromData(T* data) { return Type<T>::from(data); }                   \
    static const Type<T>& fromData(const T* data) { return Type<T>::from(data); }       \
    std::vector<T>& asVector() {                                                        \
        std::vector<T> *result = new std::vector<T>(*this);                             \
        return *result;                                                                 \
    }                                                                                   \
                                                                                        \
    Type<T>& operator+=(const std::vector<T>& other) {                                  \
        Type<T>::operator+=(Type<T>(other));                                            \
        return *this;                                                                   \
    }                                                                                   \
    Type<T> operator+(const std::vector<T>& other) const {                              \
        return Type<T>::operator+(Type<T>(other));                                      \
    }                                                                                   \
    Type<T>& operator-=(const std::vector<T>& other) {                                  \
        Type<T>::operator-=(Type<T>(other));                                            \
        return *this;                                                                   \
    }                                                                                   \
    Type<T> operator-(const std::vector<T>& other) const {                              \
        return Type<T>::operator-(Type<T>(other));                                      \
    }                                                                                   \
    Type<T>& operator*=(const std::vector<T>& other) {                                  \
        Type<T>::operator*=(Type<T>(other));                                            \
        return *this;                                                                   \
    }                                                                                   \
    Type<T> operator*(const std::vector<T>& other) const {                              \
        return Type<T>::operator*(Type<T>(other));                                      \
    }                                                                                   \
    Type<T>& operator/=(const std::vector<T>& other) {                                  \
        Type<T>::operator/=(Type<T>(other));                                            \
        return *this;                                                                   \
    }                                                                                   \
    Type<T> operator/(const std::vector<T>& other) const {                              \
        return Type<T>::operator/(Type<T>(other));                                      \
    }                                                                                   \


#define MAGNUM_BASE_VECTOR_CAST_METHODS(size, Type, MagnumImplType)                     \
                                                                                        \
    Type(const MagnumImplType<T> &other) : MagnumImplType<T>(other) {}                  \
    Type(const MagnumImplType<T> *other) : MagnumImplType<T>(*other) {}                 \
    operator MagnumImplType<T>*() {                                                     \
        return static_cast<MagnumImplType<T>*>(this);                                   \
    }                                                                                   \
    operator const MagnumImplType<T>*() {                                               \
        return static_cast<const MagnumImplType<T>*>(this);                             \
    }                                                                                   \
    operator MagnumImplType<T>&() const {                                               \
        return *static_cast<MagnumImplType<T>*>(this);                                  \
    }                                                                                   \
    operator const MagnumImplType<T>&() const {                                         \
        return *static_cast<const MagnumImplType<T>*>(this);                            \
    }                                                                                   \
                                                                                        \
    Type(const Vector<size, T> &other) : MagnumImplType<T>() {                          \
        for(int i = 0; i < other.Size; ++i) this->_data[i] = other[i];                  \
    }                                                                                   \
    operator Vector<size, T>*() { return static_cast<Vector<size, T>*>(this); }         \
    operator Vector<size, T>&() const { return *static_cast<Vector<size, T>*>(this); }  \
    operator const Vector<size, T>*() {                                                 \
        return static_cast<const Vector<size, T>*>(this);                               \
    }                                                                                   \
    operator const Vector<size, T>&() const {                                           \
        return *static_cast<const Vector<size, T>*>(this);                              \
    }                                                                                   \
    Type(const std::vector<T> &v) : MagnumImplType<T>() {                               \
        for(int i = 0; i < size; ++i) this->_data[i] = v[i];                            \
    }                                                                                   \
    operator std::vector<T>&() const {                                                  \
        std::vector<T> *result = new std::vector<T>(std::begin(this->_data),            \
            std::end(this->_data));                                                     \
        return *result;                                                                 \
    }                                                                                   \

}}

template<std::size_t size, typename T>
inline std::ostream& operator<<(std::ostream& os, const mx::type::MxVector<size, T>& vec)
{
    os << std::string("{") << vec[0];
    for(int i = 1; i < vec.Size; ++i) os << std::string(",") << vec[i];
    os << std::string("}");
    return os;
}


#define MXVECTOR_IMPL_OSTREAM(type)                                                     \
    template<typename T>                                                                \
    inline std::ostream& operator<<(std::ostream& os, const type<T>& vec)               \
    {                                                                                   \
        os << std::string("{") << vec[0];                                               \
        for(int i = 1; i < vec.Size; ++i) os << std::string(",") << vec[i];             \
        os << std::string("}");                                                         \
        return os;                                                                      \
    }                                                                                   \

#endif
