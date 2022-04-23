/**
 * @file MxCTest.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines support features for testing Mechanica C language support
 * @date 2022-04-08
 */

#ifndef _TESTING_C_MXCTEST_H_
#define _TESTING_C_MXCTEST_H_

#include <stdlib.h>

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <math.h>

#include <mechanica_c.h>
#include <MxCSimulator.h>

#define MXCTEST_CHECK(x) { if(x != S_OK) { return E_FAIL; }; }


HRESULT MxCTest_runQuiet(unsigned int numSteps) {
    double dt;
    MXCTEST_CHECK(MxCUniverse_getDt(&dt));
    MXCTEST_CHECK(MxCUniverse_step(numSteps * dt, dt));
    return S_OK;
}


#endif // _TESTING_C_MXCTEST_H_