#include <Magnum/Trade/Trade.h>

#include <Corrade/Utility/ConfigurationGroup.h>
#include <Magnum/ImageView.h>
#include <rendering/MxImageConverters.h>
#include <MagnumPlugins/TgaImageConverter/TgaImageConverter.h>
#include <MagnumPlugins/StbImageConverter/StbImageConverter.h>
#include <MxLogger.h>

using namespace Magnum;
using namespace Magnum::Trade;
using namespace Corrade;
using namespace Corrade::Utility;


Corrade::Containers::Array<char> convertImageDataToBMP(const Magnum::ImageView2D& image) {
    StbImageConverter conv(StbImageConverter::Format::Bmp);
    return conv.exportToData(image);
}

Corrade::Containers::Array<char> convertImageDataToHDR(const Magnum::ImageView2D& image) {
    StbImageConverter conv(StbImageConverter::Format::Hdr);
    return conv.exportToData(image);
}

Corrade::Containers::Array<char> convertImageDataToJpeg(const Magnum::ImageView2D& image, int jpegQuality) {
    StbImageConverter conv(StbImageConverter::Format::Jpeg);
    conv.configuration().setValue("jpegQuality", float(jpegQuality) / 100.f);
    return conv.exportToData(image);
}

Corrade::Containers::Array<char> convertImageDataToPNG(const Magnum::ImageView2D& image) {
    StbImageConverter conv(StbImageConverter::Format::Png);
    return conv.exportToData(image);
}

Corrade::Containers::Array<char> convertImageDataToTGA(const Magnum::ImageView2D& image) {
    TgaImageConverter conv;
    return conv.exportToData(image);
}
