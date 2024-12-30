#pragma once

#include "ActivationFunction.h"

#include <vector>

struct Neuron {
public:
  Neuron(std::size_t numberOfInputs,
         const ActivationFunction *activationFunction);

  double &weight(std::size_t pos);
  double output(const std::vector<double> &inputs) const;
  double outputDerivative(const std::vector<double> &inputs) const;

private:
  double intermediateQuantity(const std::vector<double> &inputs) const;

  std::vector<double> m_weights;
  const ActivationFunction *m_activationFunction;
};
