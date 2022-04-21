#ifndef SRC_RENDERING_MXIMAGECONVERTERS_H_
#define SRC_RENDERING_MXIMAGECONVERTERS_H_

#include <Corrade/Containers/Array.h>
#include <Magnum/Magnum.h>
#include <Magnum/ImageView.h>


/**
 * jpegQuality shall construct JPEG quantization tables for the given quality setting.
 * The quality value ranges from 0..100.
 */
Corrade::Containers::Array<char> convertImageDataToJpeg(const Magnum::ImageView2D& image, int jpegQuality = 100);

Corrade::Containers::Array<char> convertImageDataToBMP(const Magnum::ImageView2D& image);

Corrade::Containers::Array<char> convertImageDataToHDR(const Magnum::ImageView2D& image);

Corrade::Containers::Array<char> convertImageDataToPNG(const Magnum::ImageView2D& image);

Corrade::Containers::Array<char> convertImageDataToTGA(const Magnum::ImageView2D& image);

#endif // SRC_RENDERING_MXIMAGECONVERTERS_H_