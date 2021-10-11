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

%typemap(in) unsigned int* (unsigned int temp) {
    if($input == Py_None) $1 = NULL;
    else {
        temp = (unsigned int) PyInt_AsLong($input);
        $1 = &temp;
    }
}
%typemap(out) unsigned int* {
    if($1 == NULL) $result = Py_None;
    else $result = PyInt_FromLong(*$1);
}

%typemap(in) unsigned int& {
    $1 = (unsigned int) PyInt_AsLong($input);
}
%typemap(out) unsigned int& {
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
%template(vectori) std::vector<int16_t>;
%template(vectorl) std::vector<int32_t>;
%template(vectorll) std::vector<int64_t>;
%template(vectors) std::vector<std::string>;
%template(vectoru) std::vector<uint16_t>;
%template(vectorul) std::vector<uint32_t>;
%template(vectorull) std::vector<uint64_t>;

%template(vector2f) std::vector<std::vector<float>>;

// Generic prep instantiations for floating-point vectors. 
// This partners with vector_template_init to implement
//  functionality exclusive to vectors with floating-point data 
//  that swig doesn't automatically pick up. 
%define vector_template_prep_float(name, dataType, wrappedName)
%template(_ ## wrappedName ## _length) name::length<dataType>;
%template(_ ## wrappedName ## _normalized) name::normalized<dataType>;
%template(_ ## wrappedName ## _resized) name::resized<dataType>;
%template(_ ## wrappedName ## _projected) name::projected<dataType>;
%template(_ ## wrappedName ## _projectedOntoNormalized) name::projectedOntoNormalized<dataType>;

%extend name<dataType> {
    dataType _length() { return $self->length(); }
    name<dataType> _normalized() { return $self->normalized(); }
    name<dataType> _resized(dataType length) { return $self->resized(length); }
    name<dataType> _projected(const name<dataType> &other) { return $self->projected(other); }
    name<dataType> _projectedOntoNormalized(const name<dataType> &other) { return $self->projectedOntoNormalized(other); }

    %pythoncode %{
        def length(self):
            """length of vector"""
            return self._length()

        def normalized(self):
            """vector normalized"""
            return self._normalized()

        def resized(self, length):
            """resize be a length"""
            return self._resize(length)

        def projected(self, other):
            """project onto another vector"""
            return self._projected(other)

        def projectedOntoNormalized(self, other):
            """project onto a normalized vector"""
            return self._projectedOntoNormalized(other)
    %}
}
%enddef

// Like vector_template_prep_float, but for MxVector2
%define vector2_template_prep_float(dataType, wrappedName)
vector_template_prep_float(mx::type::MxVector2, dataType, wrappedName)

%rename(_ ## wrappedName ## _distance) mx::type::MxVector2::distance<dataType>;

%extend mx::type::MxVector2<dataType> {
    dataType _distance(const mx::type::MxVector2<dataType> &lineStartPt, const mx::type::MxVector2<dataType> &lineEndPt) { 
        return $self->distance(lineStartPt, lineEndPt); 
    }

    %pythoncode %{
        def distance(self, line_start_pt, line_end_pt):
            """distance from a line defined by two points"""
            return self._distance(line_start_pt, line_end_pt)
    %}
}

%enddef

// Like vector_template_prep_float, but for MxVector3
%define vector3_template_prep_float(dataType, wrappedName)
vector_template_prep_float(mx::type::MxVector3, dataType, wrappedName)

%rename(_ ## wrappedName ## _distance) mx::type::MxVector3::distance<dataType>;
%rename(_ ## wrappedName ## _relativeTo) mx::type::MxVector3::relativeTo<dataType>;

%extend mx::type::MxVector3<dataType> {
    dataType _distance(const mx::type::MxVector3<dataType> &lineStartPt, const mx::type::MxVector3<dataType> &lineEndPt) { 
        return $self->distance(lineStartPt, lineEndPt); 
    }
    mx::type::MxVector3<dataType> _relativeTo(const MxVector3<dataType> &origin, const MxVector3<dataType> &dim, const bool &periodic_x, const bool &periodic_y, const bool &periodic_z) {
        return $self->relativeTo(origin, dim, periodic_x, periodic_y, periodic_z);
    }

    %pythoncode %{
        def distance(self, line_start_pt, line_end_pt):
            """distance from a line defined by two points"""
            return self._distance(line_start_pt, line_end_pt)

        def relative_to(self, origin, dim, periodic_x, periodic_y, periodic_z):
            """position relative to an origin in a space with some periodic boundary conditions"""
            return self._relativeTo(origin, dim, periodic_x, periodic_y, periodic_z)
    %}
}
%enddef

// Like vector_template_prep_float, but for MxVector4
%define vector4_template_prep_float(dataType, wrappedName)
vector_template_prep_float(mx::type::MxVector4, dataType, wrappedName)

%rename(_ ## wrappedName ## _distance) mx::type::MxVector4::distance<dataType>;
%rename(_ ## wrappedName ## _distanceScaled) mx::type::MxVector4::distanceScaled<dataType>;
%rename(_ ## wrappedName ## _planeEquation) mx::type::MxVector4::planeEquation<dataType>;

%extend mx::type::MxVector4<dataType> {
    dataType _distance(const mx::type::MxVector3<dataType> &point) { return $self->distance(point); }
    dataType _distanceScaled(const mx::type::MxVector3<dataType> &point) { return $self->distanceScaled(point); }
    static mx::type::MxVector4<dataType> _planeEquation(const mx::type::MxVector3<dataType> &normal, const mx::type::MxVector3<dataType> &point) {
        return mx::type::MxVector4<dataType>::planeEquation(normal, point);
    }
    static mx::type::MxVector4<dataType> _planeEquation(const mx::type::MxVector3<dataType>& p0, 
                                                        const mx::type::MxVector3<dataType>& p1, 
                                                        const mx::type::MxVector3<dataType>& p2) 
    {
        return mx::type::MxVector4<dataType>::planeEquation(p0, p1, p2);
    }

    %pythoncode %{
        def distance(self, point):
            """distance from a point"""
            return self._distance(point)

        def distanceScaled(self, point):
            """scaled distance from a point"""
            return self._distanceScaled(point)

        @classmethod
        def planeEquation(cls, *args):
            """get a plane equation"""
            return cls._planeEquation(*args)
    %}
}
%enddef

// Do the vector template implementation
%define vector_template_init(name, dataType, wrappedName)
%ignore name<dataType>::length;
%ignore name<dataType>::normalized;
%ignore name<dataType>::resized;
%ignore name<dataType>::projected;
%ignore name<dataType>::projectedOntoNormalized;

%template(wrappedName) name<dataType>;
%enddef

// Like vector_template_init, but for MxVector2
%define vector2_template_init(dataType, wrappedName)
%ignore mx::type::MxVector2<dataType>::distance;

vector_template_init(mx::type::MxVector2, dataType, wrappedName)
%enddef

// Like vector_template_init, but for MxVector3
%define vector3_template_init(dataType, wrappedName)
%ignore mx::type::MxVector3<dataType>::distance;

vector_template_init(mx::type::MxVector3, dataType, wrappedName)
%enddef

// Like vector_template_init, but for MxVector4
%define vector4_template_init(dataType, wrappedName)
%ignore mx::type::MxVector4<dataType>::distance;
%ignore mx::type::MxVector4<dataType>::distanceScaled;
%ignore mx::type::MxVector4<dataType>::planeEquation;

vector_template_init(mx::type::MxVector4, dataType, wrappedName)
%enddef

vector2_template_prep_float(double, MxVector2d)
vector2_template_prep_float(float, MxVector2f)
vector2_template_init(double, MxVector2d)
vector2_template_init(float, MxVector2f)
vector2_template_init(int, MxVector2i)

vector3_template_prep_float(double, MxVector3d)
vector3_template_prep_float(float, MxVector3f)
vector3_template_init(double, MxVector3d)
vector3_template_init(float, MxVector3f)
vector3_template_init(int, MxVector3i)

vector4_template_prep_float(double, MxVector4d)
vector4_template_prep_float(float, MxVector4f)
vector4_template_init(double, MxVector4d)
vector4_template_init(float, MxVector4f)
vector4_template_init(int, MxVector4i)

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
            """convert to a python list"""
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
            """convert to a list of python lists"""
            return [list(v) for v in self.asVectors()]
    %}
}

%enddef

matrix_list_cast_add(mx::type::MxMatrix3, double, vectorMxMatrix3d)
matrix_list_cast_add(mx::type::MxMatrix4, double, vectorMxMatrix4d)
matrix_list_cast_add(mx::type::MxMatrix3, float, vectorMxMatrix3f)
matrix_list_cast_add(mx::type::MxMatrix4, float, vectorMxMatrix4f)
