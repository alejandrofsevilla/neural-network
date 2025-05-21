#pragma once

#include "OptimizationAlgorithm.h"

#include <Eigen/Dense>
#include <vector>

class Layer;
class CostFunction;

namespace options {
enum class CostFunction;
} // namespace options

class ADAMOptimizationAlgorithm : public OptimizationAlgorithm {
public:
  ADAMOptimizationAlgorithm(options::CostFunction costFunction,
                            std::vector<Layer> &layers);

private:
  void afterSample() override;

  Eigen::MatrixXd computeGradients(std::size_t layerId);

  std::vector<Eigen::MatrixX<std::pair<double, double>>> m_momentEstimates;
};
