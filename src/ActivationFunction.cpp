#include "ActivationFunction.h"
#include "Options.h"

#include <cmath>

std::unique_ptr<ActivationFunction>
ActivationFunction::instance(options::ActivationFunctionType type) {
  switch (type) {
  case options::ActivationFunctionType::Step:
    return std::make_unique<StepActivationFunction>();
  case options::ActivationFunctionType::Linear:
    return std::make_unique<LinearActivationFunction>();
  case options::ActivationFunctionType::Relu:
    return std::make_unique<ReluActivationFunction>();
  case options::ActivationFunctionType::Sigmoid:
    return std::make_unique<SigmoidActivationFunction>();
  case options::ActivationFunctionType::TanH:
    return std::make_unique<TanHActivationFunction>();
  default:
    return {};
  }
}

double StepActivationFunction::operator()(double input) const {
  return input >= 0.0 ? 1.0 : 0.0;
}

double StepActivationFunction::derivative(double) const {
  return 0.0;
}

double LinearActivationFunction::operator()(double input) const {
  return input;
}

double
LinearActivationFunction::derivative(double) const {
  return 1.0;
}

double ReluActivationFunction::operator()(double input) const {
  return input >= 0.0 ? input : 0.0;
}

double ReluActivationFunction::derivative(double input) const {
  return input >= 0.0 ? 1.0 : 0.0;
}

double SigmoidActivationFunction::operator()(double input) const {
  return 1.0 / (1 + exp(-input));
}

double SigmoidActivationFunction::derivative(double input) const {
  auto c{this->operator()(input)};
  return c * (1.0 - c);
}

double TanHActivationFunction::operator()(double input) const {
  auto c1{exp(input)};
  auto c2{exp(-input)};
  return (c1 - c2) / (c1 + c2);
}

double TanHActivationFunction::derivative(double input) const {
  return 1.0 - std::pow((this->operator()(input)), 2.0);
}
