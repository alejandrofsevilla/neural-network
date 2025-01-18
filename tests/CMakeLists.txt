cmake_minimum_required(VERSION 3.22.0)

project(neural-network-tests LANGUAGES CXX)

add_executable(${PROJECT_NAME}
  Test.cpp)

target_link_libraries(${PROJECT_NAME} PUBLIC neural-network)

find_package(GTest 1.11.0 REQUIRED)
if (GTest_FOUND)
  target_link_libraries(${PROJECT_NAME} PRIVATE GTest::gtest GTest::gtest_main)
endif()

target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic -Werror)
