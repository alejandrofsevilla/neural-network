#include "GradientDescendOptimizationAlgorithm.h"

#include "CostFunction.h"
#include "Layer.h"
#include "Options.h"

namespace {
auto generateGradients(const std::vector<Layer> &layers) {
  std::vector<Eigen::MatrixXd> gradients;
  std::transform(layers.cbegin(), layers.cend(), std::back_inserter(gradients),
                 [](auto &l) {
                   auto &weights{l.weights()};
                   return Eigen::MatrixXd::Zero(weights.rows(), weights.cols());
                 });
  return gradients;
}
} // namespace

GradientDescendOptimizationAlgorithm::GradientDescendOptimizationAlgorithm(
    options::CostFunction costFunction, std::vector<Layer> &layers) noexcept
    : OptimizationAlgorithm{costFunction, layers},
      m_averageGradients{generateGradients(layers)} {}

void GradientDescendOptimizationAlgorithm::afterSample() noexcept {
  std::for_each(m_layers.begin(), m_layers.end(), [this](auto &l) {
    auto &gradients{m_averageGradients.at(l.id())};
    gradients += ((l.inputs() * l.errors().transpose()) - gradients) /
                 (m_sampleCount + 1);
  });
}

void GradientDescendOptimizationAlgorithm::afterEpoch() noexcept {
  std::for_each(m_layers.begin(), m_layers.end(), [this](auto &l) {
    l.updateWeights(m_averageGradients.at(l.id()), m_learnRate);
  });
}
