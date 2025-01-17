# [![LinuxWorkflow](https://github.com/alejandrofsevilla/neural-network/actions/workflows/Linux.yml/badge.svg)](https://github.com/alejandrofsevilla/neural-network/actions/workflows/Linux.yml) [![MacOsWorkflow](https://github.com/alejandrofsevilla/neural-network/actions/workflows/MacOs.yml/badge.svg)](https://github.com/alejandrofsevilla/neural-networkboost-tcp-server-client/actions/workflows/MacOs.yml)

# neural-network
Implementation of neural network class.

## Requirements
* C++17 compiler.
* CMake 3.22.0
* GoogleTest 1.11.0

## Build and Install
- Clone the repository to your local machine.
   ```terminal
   git clone https://github.com/alejandrofsevilla/neural-network.git
   cd neural-network
   ```
- Build.
   ```terminal
   cmake -S . -B build
   cmake --build build
   ```
- Run tests.
   ```terminal
   ./build/tests/neural-network-tests
   ```
