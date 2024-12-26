#include "GradientDescendOptimizationAlgorithm.h"

#include "CostFunction.h"
#include "Layer.h"
#include "Neuron.h"
#include "Options.h"
#include "TrainingSample.h"

#include <algorithm>

GradientDescendOptimizationAlgorithm::GradientDescendOptimizationAlgorithm(
    options::CostFunctionType costFunction,
    const std::vector<std::unique_ptr<Layer>> &layers)
    : OptimizationAlgorithm(costFunction, layers), m_gradients{} {}

void GradientDescendOptimizationAlgorithm::afterSample() {
  std::for_each(m_layers.cbegin(), m_layers.cend(), [this](auto &l) {
    auto &neurons{l->neurons()};
    std::for_each(neurons.cbegin(), neurons.cend(), [this](auto &n) {
      auto &gradients{m_gradients[{n.layerId(), n.id()}]};
      auto &temporalGradients{n.gradients()};
      gradients.resize(temporalGradients.size());
      std::transform(
          gradients.cbegin(), gradients.cend(), temporalGradients.cbegin(),
          gradients.begin(),
          [this](auto g, auto tg) { return g + (tg - g) / m_samplesCount; });
    });
  });
}

void GradientDescendOptimizationAlgorithm::afterEpoch() {
  std::for_each(m_gradients.cbegin(), m_gradients.cend(), [this](auto &val) {
    auto layerId{val.first.first};
    auto neuronId{val.first.second};
    auto &gradients{val.second};
    auto &layer{m_layers.at(layerId)};
    layer->updateNeuronWeights(neuronId, gradients, m_learnRate);
  });
}
