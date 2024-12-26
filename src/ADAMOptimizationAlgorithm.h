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

class ADAMOptimizationAlgorithm : public OptimizationAlgorithm {
public:
  ADAMOptimizationAlgorithm(options::CostFunctionType costFunction,
                            const std::vector<std::unique_ptr<Layer>> &layers);

private:
  void afterSample() override;

  std::vector<double> gradients(std::size_t layerId, std::size_t neuronId);

  using ValuePair = std::pair<double, double>;
  using LayerIdNeuronIdPair = std::pair<std::size_t, std::size_t>;
  std::map<LayerIdNeuronIdPair, std::vector<ValuePair>> m_momentEstimates;
};
