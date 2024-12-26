#pragma once

#include <cstddef>

namespace options {
enum class ActivationFunctionType { Step, Linear, Relu, Sigmoid, TanH };
enum class CostFunctionType { Quadratic, CostEntropy };
enum class OptimizationType { GradientDescend, ADAM, SGD };

struct LayerConfig {
  std::size_t numberOfNeurons;
  options::ActivationFunctionType activationFunction;
};

struct TrainingConfig {
  options::OptimizationType optimization;
  options::CostFunctionType costFunction;
  std::size_t maxEpoch;
  double learnRate;
  double lossGoal;
};
} // namespace options
