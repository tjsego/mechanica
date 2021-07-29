/**
 * @file MxMatrix.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines basic matrix template wrap of Matrix from Magnum/Math
 * @date 2021-07-16
 * 
 */
#ifndef _SRC_TYPES_MXMATRIX_H_
#define _SRC_TYPES_MXMATRIX_H_

#include "../mx_error.h"

#include "MxVector.h"

#include <Magnum/Magnum.h>
#include <Magnum/Math/Matrix.h>

namespace mx { namespace type {

template<std::size_t size, class T> using Matrix = Magnum::Math::Matrix<size, T>;

template<std::size_t size, class T> 
class MxMatrix : public Matrix<size, T> {
    public:
        constexpr MxMatrix() noexcept: Matrix<size, T>() {}

        template<class ...U> constexpr MxMatrix(const Vector<size, T>& first, const U&... next) noexcept: Matrix<size, T>(first, next...) {}

        constexpr explicit MxMatrix(T value) noexcept: Matrix<size, T>(value) {}

        template<class U> constexpr explicit MxMatrix(const MxMatrix<size, U>& other) noexcept: Matrix<size, T>((Matrix<size, U>)other) {}

        template<std::size_t otherSize> constexpr explicit MxMatrix(const MxMatrix<otherSize, T>& other) noexcept: Matrix<size, T>((Matrix<otherSize, T>)other) {}

        constexpr /*implicit*/ MxMatrix(const MxMatrix<size, T>& other) noexcept: Matrix<size, T>((Matrix<size, T>)other) {}

        bool isOrthogonal() const { return Matrix<size, T>::isOrthogonal(); };

        T trace() const { return Matrix<size, T>::trace(); }

        MxMatrix<size-1, T> ij(std::size_t skipCol, std::size_t skipRow) const { return (MxMatrix<size-1, T>)Matrix<size-1, T>::ij(skipCol, skipRow); }

        T cofactor(std::size_t col, std::size_t row) const { return Matrix<size, T>::cofactor(col, row); }

        MxMatrix<size, T> comatrix() const { return (MxMatrix<size, T>)Matrix<size, T>::comatrix(); }

        MxMatrix<size, T> adjugate() const { return (MxMatrix<size, T>)Matrix<size, T>::adjugate(); }

        T determinant() const { return Matrix<size, T>::determinant(); }

        MxMatrix<size, T> inverted() const { return (MxMatrix<size, T>)Matrix<size, T>::inverted(); }

        MxMatrix<size, T> invertedOrthogonal() const { return (MxMatrix<size, T>)Matrix<size, T>::invertedOrthogonal(); }

        /* Reimplementation of functions to return correct type */
        MxMatrix<size, T> operator*(const MxMatrix<size, T>& other) const { return (MxMatrix<size, T>)Matrix<size, T>::operator*((Matrix<size, T>)other); }
        MxVector<size, T> operator*(const Vector<size, T>& other) const { return (MxVector<size, T>)Matrix<size, T>::operator*(other); }
        MxMatrix<size, T> transposed() const { return (MxMatrix<size, T>)Matrix<size, T>::transposed(); }

        static MxMatrix<size, T>& from(T* data) { return (MxMatrix<size, T>)Matrix<size, T>::from(data); }
        static const MxMatrix<size, T>& from(const T* data) { return (MxMatrix<size, T>)Matrix<size, T>::from(data); }
        
        MxMatrix<size, T> operator-() const {
            return (MxMatrix<size, T>)Matrix<size, T>::operator-();
        }
        MxMatrix<size, T>& operator+=(const MxMatrix<size, T>& other) {
            Matrix<size, T>::operator+=((Matrix<size, T>)other);
            return *this;
        }
        MxMatrix<size, T> operator+(const MxMatrix<size, T>& other) const {
            return (MxMatrix<size, T>)Matrix<size, T>::operator+((Matrix<size, T>)other);
        }
        MxMatrix<size, T>& operator-=(const MxMatrix<size, T>& other) {
            Matrix<size, T>::operator-=((Matrix<size, T>)other);
            return *this;
        }
        MxMatrix<size, T> operator-(const MxMatrix<size, T>& other) const {
            return (MxMatrix<size, T>)Matrix<size, T>::operator-((Matrix<size, T>)other);
        }
        MxMatrix<size, T>& operator*=(T number) {
            Matrix<size, T>::operator*=(number);
            return *this;
        }
        MxMatrix<size, T> operator*(T number) const {
            return (MxMatrix<size, T>)Matrix<size, T>::operator*(number);
        }
        MxMatrix<size, T>& operator/=(T number) {
            Matrix<size, T>::operator/=(number);
            return *this;
        }
        MxMatrix<size, T> operator/(T number) const {
            return (MxMatrix<size, T>)Matrix<size, T>::operator/(number);
        }
        constexpr MxMatrix<size, T> flippedCols() const {
            return (MxMatrix<size, T>)Matrix<size, T>::flippedCols();
        }
        constexpr MxMatrix<size, T> flippedRows() const {
            return (MxMatrix<size, T>)Matrix<size, T>::flippedRows();
        }
        
};

#define REVISED_MAGNUM_MATRIX_SUBCLASS_IMPLEMENTATION(size, Type, MagnumImplType, VectorType) \
    static Type<T>& from(T* data) {                                                     \
    return *reinterpret_cast<Type<T>*>(&MagnumImplType<T>::from(data));                 \
    }                                                                                   \
    static const Type<T>& from(const T* data) {                                         \
        return (Type<T>)MagnumImplType<T>::from(data);                                  \
    }                                                                                   \
                                                                                        \
    Type<T> operator-() const {                                                         \
        return (Type<T>)MagnumImplType<T>::operator-();                                 \
    }                                                                                   \
    Type<T>& operator+=(const Type<T>& other) {                                         \
        MagnumImplType<T>::operator+=((MagnumImplType<T>)other);                        \
        return *this;                                                                   \
    }                                                                                   \
    Type<T> operator+(const Type<T>& other) const {                                     \
        return (Type<T>)MagnumImplType<T>::operator+((MagnumImplType<T>)other);         \
    }                                                                                   \
    Type<T>& operator-=(const Type<T>& other) {                                         \
        MagnumImplType<T>::operator-=((MagnumImplType<T>)other);                        \
        return *this;                                                                   \
    }                                                                                   \
    Type<T> operator-(const Type<T>& other) const {                                     \
        return (Type<T>)MagnumImplType<T>::operator-((MagnumImplType<T>)other);         \
    }                                                                                   \
    Type<T>& operator*=(T number) {                                                     \
        MagnumImplType<T>::operator*=(number);                                          \
        return *this;                                                                   \
    }                                                                                   \
    Type<T> operator*(T number) const {                                                 \
        return (Type<T>)MagnumImplType<T>::operator*(number);                           \
    }                                                                                   \
    Type<T>& operator/=(T number) {                                                     \
        MagnumImplType<T>::operator/=(number);                                          \
        return *this;                                                                   \
    }                                                                                   \
    Type<T> operator/(T number) const {                                                 \
        return (Type<T>)MagnumImplType<T>::operator/(number);                           \
    }                                                                                   \
    constexpr Type<T> flippedCols() const {                                             \
        return (Type<T>)MagnumImplType<T>::flippedCols();                               \
    }                                                                                   \
    constexpr Type<T> flippedRows() const {                                             \
        return (Type<T>)MagnumImplType<T>::flippedRows();                               \
    }                                                                                   \
    VectorType<T>& operator[](std::size_t col) {                                        \
        return static_cast<VectorType<T>&>(MagnumImplType<T>::operator[](col));         \
    }                                                                                   \
    constexpr const VectorType<T> operator[](std::size_t col) const {                   \
        return VectorType<T>(MagnumImplType<T>::operator[](col));                       \
    }                                                                                   \
    VectorType<T> row(std::size_t row) const {                                          \
        return VectorType<T>(MagnumImplType<T>::row(row));                              \
    }                                                                                   \
                                                                                        \
    Type<T> operator*(const Type<T>& other) const {                                     \
        return MagnumImplType<T>::operator*(other);                                     \
    }                                                                                   \
    VectorType<T> operator*(const Vector<size, T>& other) const {                       \
        return (VectorType<T>)Matrix<size, T>::operator*(other);                        \
    }                                                                                   \
                                                                                        \
    Type<T> transposed() const { return Matrix<size, T>::transposed(); }                \
    constexpr VectorType<T> diagonal() const {                                          \
        return (VectorType<T>)Matrix<size, T>::diagonal();                              \
    }                                                                                   \
    Type<T> inverted() const { return Matrix<size, T>::inverted(); }                    \
    Type<T> invertedOrthogonal() const {                                                \
        return Matrix<size, T>::invertedOrthogonal();                                   \
    }                                                                                   \
    T* data() { return MagnumImplType<T>::data(); }                                     \
    constexpr const T* data() const { return MagnumImplType<T>::data(); }               \

#define SWIGPYTHON_MAGNUM_MATRIX_SUBCLASS_IMPLEMENTATION(size, Type, VectorType)        \
    VectorType<T>& _getitem(int i) {                                                    \
        if(i < 0) return _getitem(size - 1 + i);                                        \
        return this->operator[](i);                                                     \
    }                                                                                   \
    void _setitem(int i, const VectorType<T> &val) {                                    \
        if(i < 0) return _setitem(size - 1 + i, val);                                   \
        VectorType<T> &item = this->operator[](i);                                      \
        item = val;                                                                     \
    }                                                                                   \
    int __len__() { return size; }                                                      \
    std::vector<std::vector<T> >& asVectors() {                                         \
        std::vector<std::vector<T> > *result = new std::vector<std::vector<T> >(size);  \
        for(int i = 0; i < size; ++i) {                                                 \
            std::vector<T> &item = (*result)[i];                                        \
            item = this->operator[](i);                                                 \
        }                                                                               \
        return *result;                                                                 \
    }                                                                                   \

#define MAGNUM_BASE_MATRIX_CAST_METHODS(size, Type, MagnumImplType)                     \
    Type(const MagnumImplType<T>& other) : MagnumImplType<T>(other) {}                  \
    operator MagnumImplType<T>*() { return static_cast<MagnumImplType<T>*>(this); }     \
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
    Type(const Matrix<size, T>& other) : MagnumImplType<T>() {                          \
        for(int i = 0; i < other.Size; ++i) this->operator[](i) = other[i];             \
    }                                                                                   \
    operator Matrix<size, T>*() { return reinterpret_cast<Matrix<size, T>*>(this); }    \
    operator Matrix<size, T>&() const {                                                 \
        return *reinterpret_cast<Matrix<size, T>*>(this);                               \
    }                                                                                   \
    operator const Matrix<size, T>*() {                                                 \
        return reinterpret_cast<const Matrix<size, T>*>(this);                          \
    }                                                                                   \
    operator const Matrix<size, T>&() const {                                           \
        return *reinterpret_cast<const Matrix<size, T>*>(this);                         \
    }                                                                                   \
    operator std::vector<std::vector<T> >&() const {                                    \
        std::vector<T> *result = new std::vector<T>(size);                              \
        for(int i = 0; i < size; ++i) {                                                 \
            std::vector<T> *c = new std::vector<T>(this->operator[](i));                \
            *result[i] = *c;                                                            \
        }                                                                               \
        return *result;                                                                 \
    }                                                                                   \

}}

template<std::size_t size, class T>
inline std::ostream& operator<<(std::ostream& os, const mx::type::MxMatrix<size, T>& m)
{
    os << "{";
    for(int i = 0; i < size; ++i) os << m.row(i) << "," << std::endl;
    os << "}";
    return os;
}

#define MXMATRIX_IMPL_OSTREAM(type)                                                     \
    template<class T>                                                                   \
    inline std::ostream& operator<<(std::ostream& os, const type<T>& m)                 \
    {                                                                                   \
        os << "{";                                                                      \
        for(int i = 0; i < m.Size; ++i) os << m.row(i) << "," << std::endl;             \
        os << "}";                                                                      \
        return os;                                                                      \
    }                                                                                   \

#endif