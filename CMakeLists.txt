cmake_minimum_required(VERSION 3.22.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(neural-network LANGUAGES CXX)

add_library(${PROJECT_NAME} SHARED
  ActivationFunction.h
  Layer.h
  Layer.cpp
  Neuron.h
  Neuron.cpp
  )

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic -Werror -O0)

