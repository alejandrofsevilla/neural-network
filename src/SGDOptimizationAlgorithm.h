#pragma once

#include "OptimizationAlgorithm.h"

#include <vector>

class Layer;
class CostFunction;

namespace options {
enum class CostFunction;
} // namespace options

class SGDOptimizationAlgorithm : public OptimizationAlgorithm {
public:
  SGDOptimizationAlgorithm(options::CostFunction costFunction,
                           std::vector<Layer> &layers) noexcept;

private:
  void afterSample() noexcept override;
};
