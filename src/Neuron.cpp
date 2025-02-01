#include "Neuron.h"

#include "ActivationFunction.h"
#include "CostFunction.h"
#include "Layer.h"

#include <algorithm>
#include <random>

namespace {
constexpr auto f_bias{1};
constexpr auto f_weightInitMinValue{-1.0};
constexpr auto f_weightInitMaxValue{+1.0};

inline auto randomValue(double min, double max) {
  std::random_device rd;
  std::mt19937 eng{rd()};
  std::uniform_real_distribution<double> dist{min, max};
  return dist(eng);
}

inline auto generateWeights(std::size_t numberOfWeights) {
  std::vector<double> weights(numberOfWeights);
  std::generate(weights.begin(), weights.end(), []() {
    return randomValue(f_weightInitMinValue, f_weightInitMaxValue);
  });
  return weights;
}
} // namespace

Neuron::Neuron(const Layer &owner, std::size_t id)
    : m_activationFunction{owner.activationFunction()},
      m_inputs{owner.inputs()}, m_layerId{owner.id()}, m_id{id},
      m_weights{generateWeights(owner.numberOfInputs() + f_bias)},
      m_gradients(m_weights.size(), 0.0), m_intermediateQty{}, m_loss{} {}

std::size_t Neuron::id() const { return m_id; }

std::size_t Neuron::layerId() const { return m_layerId; }

const std::vector<double> &Neuron::weights() const { return m_weights; }

const std::vector<double> &Neuron::gradients() const { return m_gradients; }

double Neuron::computeLoss(double target,
                           const CostFunction &costFunction) const {
  return costFunction(m_activationFunction(m_intermediateQty), target);
}

double Neuron::computeOutput() {
  updateIntermediateQuantity();
  return m_activationFunction(m_intermediateQty);
}

double Neuron::computeError(double target, const CostFunction &costFunction) {
  auto output{computeOutput()};
  auto error{m_activationFunction.derivative(m_intermediateQty) *
             costFunction.derivative(output, target)};
  updateGradients(error);
  return error;
}

double Neuron::computeError(const Layer &nextLayer,
                            const std::vector<double> &nextLayerErrors) {
  auto &nextLayerNeurons{nextLayer.neurons()};
  auto error{m_activationFunction.derivative(m_intermediateQty) *
             std::accumulate(nextLayerNeurons.cbegin(), nextLayerNeurons.cend(),
                             0.0, [this, &nextLayerErrors](auto val, auto &n) {
                               return val + n.weights().at(m_id) *
                                                nextLayerErrors.at(n.id());
                             })};
  updateGradients(error);
  return error;
}

void Neuron::updateWeights(const std::vector<double> &gradients,
                           double learnRate) {
  std::transform(m_weights.cbegin(), m_weights.cend(), gradients.cbegin(),
                 m_weights.begin(),
                 [learnRate](auto w, auto g) { return w - g * learnRate; });
}

void Neuron::updateWeights(double learnRate) {
  updateWeights(m_gradients, learnRate);
}

void Neuron::updateIntermediateQuantity() {
  m_intermediateQty =
      std::inner_product(m_inputs.cbegin(), m_inputs.cend(), m_weights.cbegin(),
                         f_bias * m_weights.back());
}

void Neuron::updateGradients(double error) {
  std::transform(m_inputs.cbegin(), m_inputs.cend(), m_gradients.begin(),
                 [error](auto i) { return i * error; });
  m_gradients.back() = f_bias * error;
}
