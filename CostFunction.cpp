#include "CostFunction.h"
#include <cmath>

std::unique_ptr<CostFunction> CostFunction::instance(Type type) {
  switch (type) {
  case CostFunction::Type::Quadratic:
    return std::make_unique<QuadraticCostFunction>();
  case CostFunction::Type::CostEntropy:
    return std::make_unique<CostEntropyCostFunction>();
  default:
    return {};
  }
}

double QuadraticCostFunction::operator()(double value, double target) const {
  return 0.5 * pow((value - target), 2.0);
}

double QuadraticCostFunction::derivative(double value, double target) const {
  return value - target;
}

double CostEntropyCostFunction::operator()(double value, double target) const {
  return -(target * log(value) + (1.0 - target) * log(1.0 - value));
}

double CostEntropyCostFunction::derivative(double value, double target) const {
  return (value - target) / ((1.0 - value) * target);
}
