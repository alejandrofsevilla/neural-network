#include "CostFunction.h"
#include "Options.h"

#include <cmath>

std::unique_ptr<CostFunction>
CostFunction::instance(options::CostFunctionType type) {
  switch (type) {
  case options::CostFunctionType::Quadratic:
    return std::make_unique<QuadraticCostFunction>();
  case options::CostFunctionType::CostEntropy:
    return std::make_unique<CostEntropyCostFunction>();
  default:
    return {};
  }
}

float QuadraticCostFunction::operator()(float value, float target) const {
  return 0.5 * pow((value - target), 2.0);
}

float QuadraticCostFunction::derivative(float value, float target) const {
  return value - target;
}

float CostEntropyCostFunction::operator()(float value, float target) const {
  return -(target * log(value) + (1.0 - target) * log(1.0 - value));
}

float CostEntropyCostFunction::derivative(float value, float target) const {
  return (value - target) / ((1.0 - value) * target);
}
