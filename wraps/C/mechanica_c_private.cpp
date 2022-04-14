/**
 * @file mechanica_c_private.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines private support methods for the Mechanica C API
 * @date 2022-03-28
 */

#include "mechanica_c_private.h"


namespace mx { namespace capi {

HRESULT str2Char(const std::string s, char **c, unsigned int *n) {
    MXCPTRCHECK(c);
    MXCPTRCHECK(n);
    *n = s.size() + 1;
    char *cn = new char[*n];
    std::strcpy(cn, s.c_str());
    *c = cn;
    return S_OK;
}

std::vector<std::string> charA2StrV(const char **c, const unsigned int &n) {

    std::vector<std::string> sv;

    for(unsigned int i = 0; i < n; i++) 
        sv.push_back(c[i]);

    return sv;
}

}}
