#pragma once

#include "OptimizationAlgorithm.h"

#include <Eigen/Dense>
#include <vector>

class Layer;
class CostFunction;

namespace options {
enum class CostFunctionType;
} // namespace options

class ADAMOptimizationAlgorithm : public OptimizationAlgorithm {
public:
  ADAMOptimizationAlgorithm(options::CostFunctionType costFunction,
                            std::vector<Layer> &layers);

private:
  void afterSample() override;

  Eigen::MatrixXf computeGradients(std::size_t layerId);

  std::vector<Eigen::MatrixX<std::pair<float, float>>> m_momentEstimates;
};
