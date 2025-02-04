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

float StepActivationFunction::operator()(float input) const {
  return input >= 0.0 ? 1.0 : 0.0;
}

float StepActivationFunction::derivative(float) const { return 0.0; }

float LinearActivationFunction::operator()(float input) const { return input; }

float LinearActivationFunction::derivative(float) const { return 1.0; }

float ReluActivationFunction::operator()(float input) const {
  return input >= 0.0 ? input : 0.0;
}

float ReluActivationFunction::derivative(float input) const {
  return input >= 0.0 ? 1.0 : 0.0;
}

float SigmoidActivationFunction::operator()(float input) const {
  return 1.0 / (1 + exp(-input));
}

float SigmoidActivationFunction::derivative(float input) const {
  auto c{this->operator()(input)};
  return c * (1.0 - c);
}

float TanHActivationFunction::operator()(float input) const {
  auto c1{exp(input)};
  auto c2{exp(-input)};
  return (c1 - c2) / (c1 + c2);
}

float TanHActivationFunction::derivative(float input) const {
  return 1.0 - std::pow((this->operator()(input)), 2.0);
}
