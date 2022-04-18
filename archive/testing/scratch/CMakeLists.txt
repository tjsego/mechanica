add_executable(scratch-app 
  $<TARGET_OBJECTS:mechanica_obj>
  $<TARGET_OBJECTS:mdcore_single>
  scratch.cpp
)

target_include_directories(scratch-app
  PRIVATE 
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/mdcore/include>
  ${Python_INCLUDE_DIRS}
  $<INSTALL_INTERFACE:include/mechanica>
)

target_link_libraries(scratch-app 
  mdcore_single
  glfw
  LibSBML::LibSBML 
  Magnum::Application
  Magnum::Magnum
  Magnum::Shaders
  MagnumGlfwApplication
  Magnum::SceneGraph
  Magnum::Primitives
  Magnum::MeshTools
  MagnumWindowlessWglApplication 
  JPEG::JPEG 
  TgaImageConverter
  ${Python_LIBRARY_RELEASE}
)

install(TARGETS scratch-app DESTINATION testing/scratch)
