#pragma once

#include <memory>

namespace options {
enum class ActivationFunctionType;
} // namespace options

class ActivationFunction {
public:
  static std::unique_ptr<ActivationFunction>
  instance(options::ActivationFunctionType type);

  virtual ~ActivationFunction() = default;

  virtual float operator()(float input) const = 0;
  virtual float derivative(float input) const = 0;
};

class StepActivationFunction : public ActivationFunction {
public:
  float operator()(float input) const override;
  float derivative(float input) const override;
};

class LinearActivationFunction : public ActivationFunction {
public:
  float operator()(float input) const override;
  float derivative(float input) const override;
};

class ReluActivationFunction : public ActivationFunction {
public:
  float operator()(float input) const override;
  float derivative(float input) const override;
};

class SigmoidActivationFunction : public ActivationFunction {
public:
  float operator()(float input) const override;
  float derivative(float input) const override;
};

class TanHActivationFunction : public ActivationFunction {
public:
  float operator()(float input) const override;
  float derivative(float input) const override;
};
