#pragma once

#include "OptimizationAlgorithm.h"

#include <map>
#include <memory>
#include <vector>

class Neuron;
class Layer;
class CostFunction;

namespace options {
enum class CostFunctionType;
} // namespace options

class GradientDescendOptimizationAlgorithm : public OptimizationAlgorithm {
public:
  GradientDescendOptimizationAlgorithm(
      options::CostFunctionType costFunction,
      const std::vector<std::unique_ptr<Layer>> &layers);

private:
  void afterSample() override;
  void afterEpoch() override;

  using LayerIdNeuronIdPair = std::pair<std::size_t, std::size_t>;
  std::map<LayerIdNeuronIdPair, std::vector<double>> m_gradients;
};
