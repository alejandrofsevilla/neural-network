#pragma once

#include "OptimizationAlgorithm.h"

#include <Eigen/Dense>
#include <vector>

class Layer;
class CostFunction;

namespace options {
enum class CostFunction;
} // namespace options

class GradientDescendOptimizationAlgorithm : public OptimizationAlgorithm {
public:
  GradientDescendOptimizationAlgorithm(options::CostFunction costFunction,
                                       std::vector<Layer> &layers);

private:
  void afterSample() override;
  void afterEpoch() override;

  std::vector<Eigen::MatrixXd> m_averageGradients;
};
