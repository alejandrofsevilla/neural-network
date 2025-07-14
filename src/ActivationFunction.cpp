#include "ActivationFunction.h"
#include "Options.h"

#include <cmath>

std::unique_ptr<ActivationFunction>
ActivationFunction::instance(options::ActivationFunction type) noexcept {
  switch (type) {
  case options::ActivationFunction::Step:
    return std::make_unique<StepActivationFunction>();
  case options::ActivationFunction::Linear:
    return std::make_unique<LinearActivationFunction>();
  case options::ActivationFunction::Relu:
    return std::make_unique<ReluActivationFunction>();
  case options::ActivationFunction::Sigmoid:
    return std::make_unique<SigmoidActivationFunction>();
  case options::ActivationFunction::TanH:
    return std::make_unique<TanHActivationFunction>();
  default:
    return {};
  }
}

double StepActivationFunction::operator()(double input) const noexcept {
  return input >= 0.0 ? 1.0 : 0.0;
}

double StepActivationFunction::derivative(double) const noexcept { return 0.0; }

double LinearActivationFunction::operator()(double input) const noexcept {
  return input;
}

double LinearActivationFunction::derivative(double) const noexcept {
  return 1.0;
}

double ReluActivationFunction::operator()(double input) const noexcept {
  return input >= 0.0 ? input : 0.0;
}

double ReluActivationFunction::derivative(double input) const noexcept {
  return input >= 0.0 ? 1.0 : 0.0;
}

double SigmoidActivationFunction::operator()(double input) const noexcept {
  return 1.0 / (1 + exp(-input));
}

double SigmoidActivationFunction::derivative(double input) const noexcept {
  auto c{this->operator()(input)};
  return c * (1.0 - c);
}

double TanHActivationFunction::operator()(double input) const noexcept {
  auto c1{exp(input)};
  auto c2{exp(-input)};
  return (c1 - c2) / (c1 + c2);
}

double TanHActivationFunction::derivative(double input) const noexcept {
  return 1.0 - std::pow((this->operator()(input)), 2.0);
}
