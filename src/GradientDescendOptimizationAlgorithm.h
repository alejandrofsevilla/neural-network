#pragma once

#include "OptimizationAlgorithm.h"

#include <Eigen/Dense>
#include <vector>

class Neuron;
class Layer;
class CostFunction;

namespace options {
enum class CostFunctionType;
} // namespace options

class GradientDescendOptimizationAlgorithm : public OptimizationAlgorithm {
public:
  GradientDescendOptimizationAlgorithm(options::CostFunctionType costFunction,
                                       std::vector<Layer> &layers);

private:
  void afterSample() override;
  void afterEpoch() override;

  std::vector<Eigen::MatrixXd> m_gradients;
};
