/*
 * MxStyle.h
 *
 *  Created on: Jul 30, 2020
 *      Author: andy
 */

#ifndef INCLUDE_MXSTYLE_H_
#define INCLUDE_MXSTYLE_H_

#include <mx_port.h>

CAPI_STRUCT(MxStyle);

enum StyleFlags {
    STYLE_VISIBLE =    1 << 0,
    STYLE_STACKALLOC = 1 << 1
};

#endif /* INCLUDE_MXSTYLE_H_ */
