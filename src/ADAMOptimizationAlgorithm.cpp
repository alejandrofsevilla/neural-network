#include "ADAMOptimizationAlgorithm.h"

#include "CostFunction.h"
#include "Layer.h"
#include "Neuron.h"
#include "Options.h"

#include <algorithm>
#include <math.h>

namespace {
constexpr auto f_beta1{0.9};
constexpr auto f_beta2{0.999};
constexpr auto f_epsilon{0.00000001};
} // namespace

ADAMOptimizationAlgorithm::ADAMOptimizationAlgorithm(
    options::CostFunctionType costFunction,
    const std::vector<std::unique_ptr<Layer>> &layers)
    : OptimizationAlgorithm(costFunction, layers), m_momentEstimates{} {}

void ADAMOptimizationAlgorithm::afterSample() {
  std::for_each(m_layers.begin(), m_layers.end(), [this](auto &l) {
    auto &neurons{l->neurons()};
    std::for_each(neurons.begin(), neurons.end(), [this, &l](auto &n) {
      auto neuronId{n.id()};
      l->updateNeuronWeights(neuronId, gradients(l->id(), neuronId),
                             m_learnRate);
    });
  });
}

std::vector<double> ADAMOptimizationAlgorithm::gradients(std::size_t layerId,
                                                         std::size_t neuronId) {
  auto &momentEstimate{m_momentEstimates[{layerId, neuronId}]};
  auto &neuron{m_layers.at(layerId)->neurons().at(neuronId)};
  auto gradients{neuron.gradients()};
  momentEstimate.resize(neuron.weights().size());
  std::transform(gradients.cbegin(), gradients.cend(), momentEstimate.begin(),
                 gradients.begin(), [this](auto g, auto &m) {
                   auto &m1{m.first};
                   auto &v1{m.second};
                   m1 = f_beta1 * m1 + (1 - f_beta1) * g;
                   v1 = f_beta2 * v1 + (1 - f_beta2) * pow(g, 2.0);
                   auto m2{m1 / (1 - pow(f_beta1, m_samplesCount))};
                   auto v2{v1 / (1 - pow(f_beta2, m_samplesCount))};
                   return m2 / (sqrt(v2) + f_epsilon);
                 });
  return gradients;
}
