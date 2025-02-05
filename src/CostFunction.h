#pragma once

#include <memory>

namespace options {
enum class CostFunctionType;
} // namespace options

class CostFunction {
public:
  static std::unique_ptr<CostFunction> instance(options::CostFunctionType type);

  virtual ~CostFunction() = default;

  virtual double operator()(double value, double target) const = 0;
  virtual double derivative(double value, double target) const = 0;
};

class QuadraticCostFunction : public CostFunction {
public:
  double operator()(double value, double target) const override;
  double derivative(double value, double target) const override;
};

class CostEntropyCostFunction : public CostFunction {
public:
  double operator()(double value, double target) const override;
  double derivative(double value, double target) const override;
};
