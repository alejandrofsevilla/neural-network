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
  src/OptimizationAlgorithm.h
  src/OptimizationAlgorithm.cpp
  src/SGDOptimizationAlgorithm.h
  src/SGDOptimizationAlgorithm.cpp
  )

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
if (Eigen3_FOUND)
  target_link_libraries(${PROJECT_NAME} PRIVATE Eigen3::Eigen)
endif()

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic -Werror -O2)
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR})
target_include_directories(${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/src)

