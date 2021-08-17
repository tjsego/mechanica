// Making return by reference comprehensible for basic types
%typemap(in) int* (int temp) {
    if($input == Py_None) $1 = NULL;
    else {
        temp = (int) PyInt_AsLong($input);
        $1 = &temp;
    }
}
%typemap(out) int* {
    if($1 == NULL) $result = Py_None;
    else $result = PyInt_FromLong(*$1);
}

%typemap(in) int& {
    $1 = (int) PyInt_AsLong($input);
}
%typemap(out) int& {
    $result = PyInt_FromLong(*$1);
}

%typemap(in) float* (float temp) {
    if($input == Py_None) $1 = NULL;
    else {
        temp = (float) PyFloat_AsDouble($input);
        $1 = &temp;
    }
}
%typemap(out) float* {
    if($1 == NULL) $result = Py_None;
    else $result = PyFloat_FromDouble((double) *$1);
}

%typemap(in) float& {
    $1 = (float) PyFloat_AsDouble($input);
}
%typemap(out) float& {
    $result = PyFloat_FromDouble((double) *$1);
}

%typemap(in) double* (double temp) {
    if($input == Py_None) $1 = NULL;
    else{
        temp = PyFloat_AsDouble($input);
        $1 = &temp;
    }
}
%typemap(out) double* {
    if($1 == NULL) $result = Py_None;
    else $result = PyFloat_FromDouble(*$1);
}

%typemap(in) double& {
    $1 = PyFloat_AsDouble($input);
}
%typemap(out) double& {
    $result = PyFloat_FromDouble(*$1);
}

%typemap(in) std::string* (std::string temp) {
    if($input == Py_None) $1 = NULL;
    else {
        std::string temp = std::string(PyUnicode_AsUTF8($input));
        $1 = &temp;
    }
}
%typemap(out) std::string* {
    if(!$1) $result = Py_None;
    else $result = PyUnicode_FromString($1->c_str());
}

%typemap(in) bool* (bool temp) {
    if($input == Py_None) $1 = NULL;
    else {
        if($input == Py_True) temp = true;
        else temp = false;
        $1 = &temp;
    }
}
%typemap(out) bool* {
    if(!$1) $result = Py_None;
    else $result = PyBool_FromLong(*$1);
}

typedef short int16_t;
typedef int int32_t;
typedef long long int64_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

%inline %{
    using namespace std;
%}

%{
    #include <types/MxVector.h>
    #include <types/MxVector2.h>
    #include <types/MxVector3.h>
    #include <types/MxVector4.h>

    #include <types/MxMatrix.h>
    #include <types/MxMatrix3.h>
    #include <types/MxMatrix4.h>

    #include <types/MxQuaternion.h>

    using namespace mx::type;
%}

%include "MxVector.h"
%include "MxVector2.h"
%include "MxVector3.h"
%include "MxVector4.h"

%include "MxMatrix.h"
%include "MxMatrix3.h"
%include "MxMatrix4.h"

%include "MxQuaternion.h"

typedef long HRESULT;

typedef mx::type::MxVector2<double> MxVector2d;
typedef mx::type::MxVector3<double> MxVector3d;
typedef mx::type::MxVector4<double> MxVector4d;

typedef mx::type::MxVector2<float> MxVector2f;
typedef mx::type::MxVector3<float> MxVector3f;
typedef mx::type::MxVector4<float> MxVector4f;

typedef mx::type::MxVector2<int> MxVector2i;
typedef mx::type::MxVector3<int> MxVector3i;
typedef mx::type::MxVector4<int> MxVector4i;

typedef mx::type::MxMatrix3<double> MxMatrix3d;
typedef mx::type::MxMatrix4<double> MxMatrix4d;

typedef mx::type::MxMatrix3<float> MxMatrix3f;
typedef mx::type::MxMatrix4<float> MxMatrix4f;

typedef mx::type::MxQuaternion<double> MxQuaterniond;
typedef mx::type::MxQuaternion<float> MxQuaternionf;

%template(lists) std::list<std::string>;
%template(pairff) std::pair<float, float>;
%template(umapsb) std::unordered_map<std::string, bool>;
%template(vectord) std::vector<double>;
%template(vectorf) std::vector<float>;
%template(vectori) std::vector<int>;
%template(vectors) std::vector<std::string>;

%template(vector2f) std::vector<std::vector<float>>;

%template(MxVector2d) mx::type::MxVector2<double>;
%template(MxVector2f) mx::type::MxVector2<float>;
%template(MxVector2i) mx::type::MxVector2<int>;

%template(MxVector3d) mx::type::MxVector3<double>;
%template(MxVector3f) mx::type::MxVector3<float>;
%template(MxVector3i) mx::type::MxVector3<int>;

%template(MxVector4d) mx::type::MxVector4<double>;
%template(MxVector4f) mx::type::MxVector4<float>;
%template(MxVector4i) mx::type::MxVector4<int>;

%template(MxMatrix3d) mx::type::MxMatrix3<double>;
%template(MxMatrix3f) mx::type::MxMatrix3<float>;

%template(MxMatrix4d) mx::type::MxMatrix4<double>;
%template(MxMatrix4f) mx::type::MxMatrix4<float>;

%template(MxQuaterniond) mx::type::MxQuaternion<double>;
%template(MxQuaternionf) mx::type::MxQuaternion<float>;

%define vector_list_cast_add(name, dataType, vectorName)

%template(vectorName) std::vector<name<dataType>>;
%template(vectorName ## _p) std::vector<name<dataType>*>;

%extend name<dataType>{
    %pythoncode %{
        def __getitem__(self, index: int):
            if index >= len(self):
                raise IndexError('Valid indices < ' + str(len(self)))
            return self._getitem(index)

        def __setitem__(self, index: int, val):
            if index >= len(self):
                raise IndexError('Valid indices < ' + str(len(self)))
            self._setitem(index, val)

        def as_list(self) -> list:
            return list(self.asVector())
    %}
}

%enddef

vector_list_cast_add(mx::type::MxVector2, double, vectorMxVector2d)
vector_list_cast_add(mx::type::MxVector3, double, vectorMxVector3d)
vector_list_cast_add(mx::type::MxVector4, double, vectorMxVector4d)
vector_list_cast_add(mx::type::MxVector2, float, vectorMxVector2f)
vector_list_cast_add(mx::type::MxVector3, float, vectorMxVector3f)
vector_list_cast_add(mx::type::MxVector4, float, vectorMxVector4f)
vector_list_cast_add(mx::type::MxVector2, int, vectorMxVector2i)
vector_list_cast_add(mx::type::MxVector3, int, vectorMxVector3i)
vector_list_cast_add(mx::type::MxVector4, int, vectorMxVector4i)
vector_list_cast_add(mx::type::MxQuaternion, double, vectorMxQuaterniond)
vector_list_cast_add(mx::type::MxQuaternion, float, vectorMxQuaternionf)

%define matrix_list_cast_add(name, dataType, vectorName)

%template(vectorName) std::vector<name<dataType>>;
%template(vectorName ## _p) std::vector<name<dataType>*>;

%extend name<dataType>{
    %pythoncode %{
        def __getitem__(self, index: int):
            if index >= len(self):
                raise IndexError('Valid indices < ' + str(len(self)))
            return self._getitem(index)

        def __setitem__(self, index: int, val):
            if index >= len(self):
                raise IndexError('Valid indices < ' + str(len(self)))
            self._setitem(index, val)

        def as_lists(self) -> list:
            return [list(v) for v in self.asVectors()]
    %}
}

%enddef

matrix_list_cast_add(mx::type::MxMatrix3, double, vectorMxMatrix3d)
matrix_list_cast_add(mx::type::MxMatrix4, double, vectorMxMatrix4d)
matrix_list_cast_add(mx::type::MxMatrix3, float, vectorMxMatrix3f)
matrix_list_cast_add(mx::type::MxMatrix4, float, vectorMxMatrix4f)
