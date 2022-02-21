/*
 * MxUI.cpp
 *
 *  Created on: Oct 6, 2018
 *      Author: andy
 */

#include <rendering/MxUI.h>
#include "MxTestView.h"
#include <iostream>

MxTestView *view = nullptr;

HRESULT MxUI_PollEvents()
{
    glfwPollEvents();
    if(view) {
        view->draw();
    }
    return S_OK;
}

HRESULT MxUI_WaitEvents(double timeout)
{
    glfwWaitEventsTimeout(timeout);
    glfwWaitEvents();
    if(view) {
        view->draw();
    }

    return S_OK;
}

HRESULT MxUI_PostEmptyEvent()
{
    glfwPostEmptyEvent();
    return S_OK;
}

static void error_callback(int error, const char* description)
{
    fprintf(stderr, "Error: %s\n", description);
}


HRESULT MxUI_InitializeGraphics()
{
    std::cout << MX_FUNCTION << std::endl;

    glfwSetErrorCallback(error_callback);

    if (!glfwInit()) {
        return E_FAIL;
    }


    glfwWindowHint(GLFW_SAMPLES, 4); // 4x antialiasing
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4); // We want OpenGL 3.3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // To make MacOS happy; should not be needed
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); // We don't want the old OpenGL

    return S_OK;
}

HRESULT MxUI_CreateTestWindow()
{
    std::cout << MX_FUNCTION << std::endl;

    if(!view) {
        view = new MxTestView(500,500);
    }

    return S_OK;
}

HRESULT MxUI_DestroyTestWindow()
{
    std::cout << MX_FUNCTION << std::endl;

    delete view;
    view = nullptr;
    return S_OK;
}
