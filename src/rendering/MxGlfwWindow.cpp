/*
 * MxWindowNative.cpp
 *
 *  Created on: Apr 13, 2020
 *      Author: andy
 */

#include <rendering/MxGlfwApplication.h>
#include <rendering/MxGlfwWindow.h>
#include <Magnum/GL/DefaultFramebuffer.h>

#include <MxPy.h>
#include <iostream>

using namespace Magnum;

const float &MxGlfwWindow::getFloatField()
{
    return this->f;
}

void MxGlfwWindow::setFloatField(const float &value)
{
    this->f = value;
}




template <class Class, class Result, Result Class::*Member>
struct MyStruct {
    float f;
};

template<typename Klass, typename VarType, VarType Klass::*pm>
void Test(const char* name, const char* doc) {
    std::cout << "foo";
}

MxGlfwWindow::MxGlfwWindow(GLFWwindow *win)
{
    _window = win;
}

//MxGlfwWindow::State MxGlfwWindow::getMouseButtonState(MouseButton mouseButton)
//{
//    return (State)glfwGetMouseButton(_window, (int)mouseButton);
//}



MxVector2i MxGlfwWindow::windowSize() const {
    CORRADE_ASSERT(_window, "Platform::GlfwApplication::windowSize(): no window opened", {});

    MxVector2i size;
    glfwGetWindowSize(_window, &size.x(), &size.y());
    return size;
}

void MxGlfwWindow::redraw() {

    // TODO: get rid of GLFWApplication
    MxSimulator::get()->redraw();
}

Magnum::GL::AbstractFramebuffer &MxGlfwWindow::framebuffer() {
    return Magnum::GL::defaultFramebuffer;
}

void MxGlfwWindow::setTitle(const char* title) {
    
}

