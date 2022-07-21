/**
 * @file MxTest.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines support features for testing Mechanica C++ language support
 * @date 2022-07-21
 */

#ifndef _TESTING_CPP_MXTEST_H_
#define _TESTING_CPP_MXTEST_H_

#include <mx_port.h>

#define MXTEST_REPORTERR() { std::cerr << "Error: " << __LINE__ << ", " << MX_FUNCTION << ", " << __FILE__ << std::endl; }
#define MXTEST_CHECK(code) { if((code) != S_OK) { MXTEST_REPORTERR(); return E_FAIL; } }

#endif // _TESTING_CPP_MXTEST_H_