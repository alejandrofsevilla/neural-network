#pragma once

#include "OptimizationAlgorithm.h"

#include <vector>

class Layer;
class CostFunction;

namespace options {
enum class CostFunctionType;
} // namespace options

class SGDOptimizationAlgorithm : public OptimizationAlgorithm {
public:
  SGDOptimizationAlgorithm(options::CostFunctionType costFunction,
                           std::vector<Layer> &layers);

private:
  void afterSample() override;
};
