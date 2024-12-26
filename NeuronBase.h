#ifndef NEURAL_NETWORK_NEURON_BASE_H
#define NEURAL_NETWORK_NEURON_BASE_H

#include "ActivationFunction.h"

#include <array>
#include <numeric>

template <std::size_t> class Neuron;

class NeuronBase {
public:
  virtual double weight(std::size_t index) const = 0;
  virtual void setWeight(std::size_t index, double value) = 0;

  template <std::size_t N>
  double output(const std::array<double, N> &inputs,
                const ActivationFunction &activationFunction) const {
    return dynamic_cast<const Neuron<N> &>(this)->output(inputs,
                                                         activationFunction);
  }

  template <std::size_t N>
  double outputDerivative(const std::array<double, N> &inputs,
                          const ActivationFunction &activationFunction) const {
    return dynamic_cast<const Neuron<N> &>(this)->outputDerivative(
        inputs, activationFunction);
  }
};

#endif
