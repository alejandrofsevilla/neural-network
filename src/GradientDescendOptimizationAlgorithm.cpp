#include "GradientDescendOptimizationAlgorithm.h"

#include "CostFunction.h"
#include "Layer.h"
#include "Options.h"

GradientDescendOptimizationAlgorithm::GradientDescendOptimizationAlgorithm(
    options::CostFunctionType costFunction, std::vector<Layer> &layers)
    : OptimizationAlgorithm(costFunction, layers),
      m_gradients(m_layers.size()) {}

void GradientDescendOptimizationAlgorithm::afterSample() {
  std::for_each(m_layers.begin(), m_layers.end(), [this](auto &l) {
    auto &layerGradients{m_gradients.at(l.id())};
    layerGradients.resize(l.weights().rows(), l.weights().cols());
    layerGradients += (l.computeGradients() - layerGradients) / m_samplesCount;
  });
}

void GradientDescendOptimizationAlgorithm::afterEpoch() {
  std::for_each(m_layers.begin(), m_layers.end(), [this](auto &l) {
    l.updateWeights(m_gradients.at(l.id()), m_learnRate);
  });
}
