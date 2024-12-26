#pragma once

#include "OptimizationAlgorithm.h"

#include <memory>
#include <vector>

class Layer;
class CostFunction;

namespace options {
enum class CostFunctionType;
} // namespace options

class SGDOptimizationAlgorithm : public OptimizationAlgorithm {
public:
  SGDOptimizationAlgorithm(options::CostFunctionType costFunction,
                           const std::vector<std::unique_ptr<Layer>> &layers);

private:
  void afterSample() override;
};
