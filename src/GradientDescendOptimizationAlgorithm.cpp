#include "GradientDescendOptimizationAlgorithm.h"

#include "CostFunction.h"
#include "Layer.h"
#include "Options.h"

GradientDescendOptimizationAlgorithm::GradientDescendOptimizationAlgorithm(
    options::CostFunctionType costFunction, std::vector<Layer> &layers)
    : OptimizationAlgorithm(costFunction, layers),
      m_averageGradients(m_layers.size()) {}

void GradientDescendOptimizationAlgorithm::afterSample() {
  std::for_each(m_layers.begin(), m_layers.end(), [this](auto &l) {
    auto &layerGradients{m_averageGradients.at(l.id())};
    layerGradients.resize(l.weights().rows(), l.weights().cols());
    layerGradients += ((l.inputs() * l.errors().transpose()) - layerGradients) /
                      m_samplesCount;
  });
}

void GradientDescendOptimizationAlgorithm::afterEpoch() {
  std::for_each(m_layers.begin(), m_layers.end(), [this](auto &l) {
    l.updateWeights(m_averageGradients.at(l.id()), m_learnRate);
  });
}
