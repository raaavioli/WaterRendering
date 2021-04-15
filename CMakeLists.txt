cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

project(WaterRendering VERSION 1.0
  DESCRIPTION "Water rendering project for course DH2323"
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-O2 -Wall")

IF(APPLE)
  SET(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "Build architectures for Mac OS X" FORCE)
ENDIF(APPLE)

add_subdirectory(glfw)
add_subdirectory(glm)

add_subdirectory("${CMAKE_SOURCE_DIR}/glew-2.2.0/build/cmake"
  "${CMAKE_SOURCE_DIR/glew-2.2.0/}"
)

include_directories(
  ${GLFW_INCLUDE_DIRS}
  ${CMAKE_SOURCE_DIR}/glew-2.2.0/include
  ${CMAKE_SOURCE_DIR}/include
)
add_executable(${PROJECT_NAME}
  ${CMAKE_SOURCE_DIR}/src/main.cpp
)

target_link_libraries(${PROJECT_NAME} glm glfw glew)

