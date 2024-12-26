cmake_minimum_required(VERSION 3.22.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(neural-network LANGUAGES CXX)

add_subdirectory(tests)

add_library(${PROJECT_NAME} SHARED
  NeuralNetwork.h
  Options.h
  TrainingReport.h
  TrainingSample.h
  src/ActivationFunction.h
  src/ActivationFunction.cpp
  src/ADAMOptimizationAlgorithm.h
  src/ADAMOptimizationAlgorithm.cpp
  src/CostFunction.h
  src/CostFunction.cpp
  src/GradientDescendOptimizationAlgorithm.h
  src/GradientDescendOptimizationAlgorithm.cpp
  src/Layer.h
  src/Layer.cpp
  src/NeuralNetwork.cpp
  src/Neuron.h
  src/Neuron.cpp
  src/OptimizationAlgorithm.h
  src/OptimizationAlgorithm.cpp
  src/SGDOptimizationAlgorithm.h
  src/SGDOptimizationAlgorithm.cpp
  )

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
if (CMAKE_COMPILER_IS_GNUCXX)
  target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic -Werror)
elseif(MSVC)
  target_compile_options(${PROJECT_NAME} PRIVATE -W4)
endif()

target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR})
target_include_directories(${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/src)

