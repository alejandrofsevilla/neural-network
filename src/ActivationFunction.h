#pragma once

#include <memory>

namespace options {
enum class ActivationFunction;
} // namespace options

class ActivationFunction {
public:
  static std::unique_ptr<ActivationFunction>
  instance(options::ActivationFunction type) noexcept;

  virtual ~ActivationFunction() noexcept = default;

  virtual double operator()(double input) const noexcept = 0;
  virtual double derivative(double input) const noexcept = 0;
};

class StepActivationFunction : public ActivationFunction {
public:
  double operator()(double input) const noexcept override;
  double derivative(double input) const noexcept override;
};

class LinearActivationFunction : public ActivationFunction {
public:
  double operator()(double input) const noexcept override;
  double derivative(double input) const noexcept override;
};

class ReluActivationFunction : public ActivationFunction {
public:
  double operator()(double input) const noexcept override;
  double derivative(double input) const noexcept override;
};

class SigmoidActivationFunction : public ActivationFunction {
public:
  double operator()(double input) const noexcept override;
  double derivative(double input) const noexcept override;
};

class TanHActivationFunction : public ActivationFunction {
public:
  double operator()(double input) const noexcept override;
  double derivative(double input) const noexcept override;
};
