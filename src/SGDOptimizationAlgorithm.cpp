#include "SGDOptimizationAlgorithm.h"

#include "CostFunction.h"
#include "Layer.h"
#include "Neuron.h"
#include "Options.h"

#include <algorithm>

SGDOptimizationAlgorithm::SGDOptimizationAlgorithm(
    options::CostFunctionType costFunction,
    const std::vector<std::unique_ptr<Layer>> &layers)
    : OptimizationAlgorithm(costFunction, layers) {}

void SGDOptimizationAlgorithm::afterSample() {
  std::for_each(m_layers.begin(), m_layers.end(), [this](auto &l) {
    auto &neurons{l->neurons()};
    std::for_each(neurons.begin(), neurons.end(), [this, &l](auto &n) {
      l->updateNeuronWeights(n.id(), m_learnRate);
    });
  });
}
