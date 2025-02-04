#pragma once

#include <memory>

namespace options {
enum class CostFunctionType;
} // namespace options

class CostFunction {
public:
  static std::unique_ptr<CostFunction> instance(options::CostFunctionType type);

  virtual ~CostFunction() = default;

  virtual float operator()(float value, float target) const = 0;
  virtual float derivative(float value, float target) const = 0;
};

class QuadraticCostFunction : public CostFunction {
public:
  float operator()(float value, float target) const override;
  float derivative(float value, float target) const override;
};

class CostEntropyCostFunction : public CostFunction {
public:
  float operator()(float value, float target) const override;
  float derivative(float value, float target) const override;
};
