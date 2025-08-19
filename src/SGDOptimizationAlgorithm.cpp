#include "SGDOptimizationAlgorithm.h"

#include "CostFunction.h"
#include "Layer.h"

SGDOptimizationAlgorithm::SGDOptimizationAlgorithm(
    options::CostFunction costFunction, std::vector<Layer> &layers)
    : OptimizationAlgorithm{costFunction, layers} {}

void SGDOptimizationAlgorithm::afterSample() {
  std::for_each(m_layers.begin(), m_layers.end(),
                [this](auto &l) { l.updateWeights(m_learnRate); });
}
