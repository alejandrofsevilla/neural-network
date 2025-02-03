#include "ADAMOptimizationAlgorithm.h"

#include "CostFunction.h"
#include "Layer.h"
#include "Options.h"

#include <algorithm>
#include <math.h>

namespace {
constexpr auto f_beta1{0.9};
constexpr auto f_beta2{0.999};
constexpr auto f_epsilon{0.00000001};
} // namespace

ADAMOptimizationAlgorithm::ADAMOptimizationAlgorithm(
    options::CostFunctionType costFunction, std::vector<Layer> &layers)
    : OptimizationAlgorithm(costFunction, layers),
      m_momentEstimates(layers.size()) {}

void ADAMOptimizationAlgorithm::afterSample() {
  std::for_each(m_layers.begin(), m_layers.end(), [this](auto &l) {
    l.updateWeights(computeGradients(l.id()), m_learnRate);
  });
}

Eigen::MatrixXd
ADAMOptimizationAlgorithm::computeGradients(std::size_t layerId) {
  auto &momentEstimates{m_momentEstimates.at(layerId)};
  auto &layer{m_layers.at(layerId)};
  momentEstimates.resize(layer.weights().rows(), layer.weights().cols());
  auto gradients{layer.computeGradients()};
  auto gradientsView{gradients.reshaped()};
  std::transform(gradientsView.cbegin(), gradientsView.cend(),
                 momentEstimates.reshaped().begin(), gradientsView.begin(),
                 [this](auto &g, auto &m) {
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
