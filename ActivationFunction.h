#ifndef NEURAL_NETWORK_ACTIVATION_FUNCTION_H
#define NEURAL_NETWORK_ACTIVATION_FUNCTION_H

#include <memory>

class ActivationFunction {
public:
  enum class Type { Step, Linear, Relu, Sigmoid, TanH };

  static std::unique_ptr<ActivationFunction> instance(Type type);

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

// TODO: LinearActivationFunction, Sigmoid...

#endif
