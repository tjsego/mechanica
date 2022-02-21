/**
 * @file mx_types.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines native and wrapped primitive types
 * @date 2021-07-16
 * 
 */
#ifndef _SRC_TYPES_MX_TYPES_H_
#define _SRC_TYPES_MX_TYPES_H_

#include "MxVector.h"
#include "MxVector2.h"
#include "MxVector3.h"
#include "MxVector4.h"
#include "MxMatrix.h"
#include "MxMatrix3.h"
#include "MxMatrix4.h"
#include "MxQuaternion.h"


typedef mx::type::MxVector2<double> MxVector2d;
typedef mx::type::MxVector3<double> MxVector3d;
typedef mx::type::MxVector4<double> MxVector4d;

typedef mx::type::MxVector2<float> MxVector2f;
typedef mx::type::MxVector3<float> MxVector3f;
typedef mx::type::MxVector4<float> MxVector4f;

typedef mx::type::MxVector2<int> MxVector2i;
typedef mx::type::MxVector3<int> MxVector3i;
typedef mx::type::MxVector4<int> MxVector4i;

typedef mx::type::MxVector2<unsigned int> MxVector2ui;
typedef mx::type::MxVector3<unsigned int> MxVector3ui;
typedef mx::type::MxVector4<unsigned int> MxVector4ui;

typedef mx::type::MxMatrix3<double> MxMatrix3d;
typedef mx::type::MxMatrix4<double> MxMatrix4d;

typedef mx::type::MxMatrix3<float> MxMatrix3f;
typedef mx::type::MxMatrix4<float> MxMatrix4f;

typedef mx::type::MxQuaternion<double> MxQuaterniond;
typedef mx::type::MxQuaternion<float> MxQuaternionf;

#endif // _SRC_TYPES_MX_TYPES_H_