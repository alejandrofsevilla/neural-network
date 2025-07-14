#pragma once

#include <memory>

namespace options {
enum class CostFunction;
} // namespace options

class CostFunction {
public:
  static std::unique_ptr<CostFunction>
  instance(options::CostFunction type) noexcept;

  virtual ~CostFunction() noexcept = default;

  virtual double operator()(double value, double target) const noexcept = 0;
  virtual double derivative(double value, double target) const noexcept = 0;
};

class QuadraticCostFunction : public CostFunction {
public:
  double operator()(double value, double target) const noexcept override;
  double derivative(double value, double target) const noexcept override;
};

class CostEntropyCostFunction : public CostFunction {
public:
  double operator()(double value, double target) const noexcept override;
  double derivative(double value, double target) const noexcept override;
};
