#ifndef NEURAL_NETWORK_COST_FUNCTION_H
#define NEURAL_NETWORK_COST_FUNCTION_H

#include <memory>

class CostFunction {
public:
  enum class Type { Quadratic, CostEntropy };

  static std::unique_ptr<CostFunction> instance(Type type);

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

#endif
