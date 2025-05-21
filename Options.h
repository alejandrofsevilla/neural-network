#pragma once

#include <cstddef>

namespace options {
enum class ActivationFunction { Step, Linear, Relu, Sigmoid, TanH };
enum class CostFunction { Quadratic, CostEntropy };
enum class Optimization { GradientDescend, ADAM, SGD };

struct LayerConfig {
  std::size_t numberOfInputs;
  std::size_t numberOfNeurons;
  options::ActivationFunction activationFunction;
};

struct TrainingConfig {
  options::Optimization optimization;
  options::CostFunction costFunction;
  std::size_t maxEpoch;
  double learnRate;
  double lossGoal;
};
} // namespace options
