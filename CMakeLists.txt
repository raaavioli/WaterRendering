cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

project(WaterRendering VERSION 1.0
  DESCRIPTION "Water rendering project for course DH2323"
  LANGUAGES CXX CUDA
)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_FLAGS "-O2 -Wall")

IF(APPLE)
  SET(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "Build architectures for Mac OS X" FORCE)
ENDIF(APPLE)

find_package(CUDA REQUIRED)

list(APPEND CUDA_NVCC_FLAGS "-std=c++11 -rdc=true")
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_SEPARABLE_COMPILATION ON)

add_subdirectory(glfw)
add_subdirectory(glm)
add_subdirectory(glad)

include_directories(
  ${GLFW_INCLUDE_DIRS}
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/imgui
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

file(COPY assets DESTINATION ${CMAKE_BINARY_DIR})
file(GLOB SRCFILES ${CMAKE_SOURCE_DIR}/src/*.cpp)
file(GLOB IMGUI 
  ${CMAKE_SOURCE_DIR}/imgui/imgui.cpp 
  ${CMAKE_SOURCE_DIR}/imgui/imgui_draw.cpp
  ${CMAKE_SOURCE_DIR}/imgui/imgui_tables.cpp
  ${CMAKE_SOURCE_DIR}/imgui/imgui_widgets.cpp
  ${CMAKE_SOURCE_DIR}/imgui/backends/imgui_impl_opengl3.cpp 
  ${CMAKE_SOURCE_DIR}/imgui/backends/imgui_impl_glfw.cpp
)


add_executable(${PROJECT_NAME}
  ${SRCFILES}
  ${IMGUI}
)

target_link_libraries(${PROJECT_NAME} glm glfw glad cuda ${CUDA_cufft_LIBRARY} ${CUDA_LIBRARIES})

