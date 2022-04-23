/**
 * @file MxMagnum.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines various support features for Magnum. 
 * @date 2022-04-21
 * 
 */

#include <Magnum/GL/GL.h>

#include "mx_cast.h"

#include <cstdint>

namespace mx {

template<> Magnum::GL::Version cast(const std::int32_t &);

template<> std::int32_t cast(const Magnum::GL::Version &);

};
