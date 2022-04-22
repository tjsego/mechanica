/**
 * @file MxMagnum.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines various support features for Magnum. 
 * @date 2022-04-21
 * 
 */

#include "MxMagnum.h"

namespace mx {

template<> Magnum::GL::Version cast(const std::int32_t &v) { return (Magnum::GL::Version)v; }

template<> std::int32_t cast(const Magnum::GL::Version &v) { return (std::int32_t)v; }

};
