#include "ADAMOptimizationAlgorithm.h"

#include "CostFunction.h"
#include "Layer.h"
#include "Options.h"

#include <algorithm>
#include <iterator>
#include <math.h>

namespace {
constexpr auto f_beta1{0.9};
constexpr auto f_beta2{0.999};
constexpr auto f_epsilon{0.00000001};

auto generateGradients(const std::vector<Layer> &layers) {
  std::vector<Eigen::MatrixX<std::pair<float, float>>> gradients;
  std::transform(layers.cbegin(), layers.cend(), std::back_inserter(gradients),
                 [](auto &l) {
                   return Eigen::MatrixX<std::pair<float, float>>(
                       l.weights().cols(), l.weights().rows());
                 });
  return gradients;
}
} // namespace

ADAMOptimizationAlgorithm::ADAMOptimizationAlgorithm(
    options::CostFunctionType costFunction, std::vector<Layer> &layers)
    : OptimizationAlgorithm(costFunction, layers),
      m_momentEstimates{generateGradients(layers)} {}

void ADAMOptimizationAlgorithm::afterSample() {
  std::for_each(m_layers.begin(), m_layers.end(), [this](auto &l) {
    l.updateWeights(computeGradients(l.id()), m_learnRate);
  });
}

Eigen::MatrixXf
ADAMOptimizationAlgorithm::computeGradients(std::size_t layerId) {
  auto &layer{m_layers.at(layerId)};
  Eigen::MatrixXf gradients{layer.inputs() * layer.errors().transpose()};
  auto gradientsView{gradients.reshaped()};
  auto momentEstimatesView{m_momentEstimates.at(layerId).reshaped()};
  for (auto i = 0; i < gradientsView.size(); i++) {
    auto &m{momentEstimatesView[i]};
    auto &g{gradientsView[i]};
    auto &m1{m.first};
    auto &v1{m.second};
    m1 = f_beta1 * m1 + (1 - f_beta1) * g;
    v1 = f_beta2 * v1 + (1 - f_beta2) * pow(g, 2.0);
    auto m2{m1 / (1 - pow(f_beta1, m_samplesCount))};
    auto v2{v1 / (1 - pow(f_beta2, m_samplesCount))};
    g = m2 / (sqrt(v2) + f_epsilon);
  }
  return gradients;
}
