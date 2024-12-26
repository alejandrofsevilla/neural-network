#pragma once

#include <memory>

namespace options {
enum class ActivationFunctionType;
} // namespace options

class ActivationFunction {
public:
  static std::unique_ptr<ActivationFunction>
  instance(options::ActivationFunctionType type);

  virtual double operator()(double input) const = 0;
  virtual double derivative(double input) const = 0;
};

class StepActivationFunction : public ActivationFunction {
public:
  double operator()(double input) const override;
  double derivative(double input) const override;
};

class LinearActivationFunction : public ActivationFunction {
public:
  double operator()(double input) const override;
  double derivative(double input) const override;
};

class ReluActivationFunction : public ActivationFunction {
public:
  double operator()(double input) const override;
  double derivative(double input) const override;
};

class SigmoidActivationFunction : public ActivationFunction {
public:
  double operator()(double input) const override;
  double derivative(double input) const override;
};

class TanHActivationFunction : public ActivationFunction {
public:
  double operator()(double input) const override;
  double derivative(double input) const override;
};
