#include "Layer.h"

#include "ActivationFunction.h"
#include "CostFunction.h"

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
} // namespace

Layer::Layer(std::size_t id, std::size_t numberOfInputs,
             std::size_t numberOfNeurons,
             options::ActivationFunctionType activationFunction)
    : m_id{id}, m_numberOfInputs{numberOfInputs},
      m_numberOfNeurons{numberOfNeurons},
      m_activationFunction{ActivationFunction::instance(activationFunction)},
      m_intermediateQtys{Eigen::VectorXd::Zero(numberOfNeurons)},
      m_inputs{Eigen::VectorXd::Ones(numberOfInputs + f_bias)},
      m_outputs{Eigen::VectorXd::Zero(numberOfNeurons)},
      m_errors{Eigen::VectorXd::Zero(numberOfNeurons)},
      m_weights{Eigen::MatrixXd::NullaryExpr(
          numberOfInputs + f_bias, numberOfNeurons,
          []() {
            return randomValue(f_weightInitMinValue, f_weightInitMaxValue);
          })},
      m_loss{} {}

std::size_t Layer::id() const { return m_id; }

std::size_t Layer::numberOfInputs() const { return m_numberOfInputs; }

std::size_t Layer::numberOfNeurons() const { return m_numberOfNeurons; }

double Layer::loss() const { return m_loss; }

const Eigen::MatrixXd &Layer::weights() const { return m_weights; }

const Eigen::VectorXd &Layer::outputs() const { return m_outputs; }

const Eigen::VectorXd &Layer::errors() const { return m_errors; }

Eigen::MatrixXd Layer::computeGradients() const {
  return m_inputs * m_errors.transpose();
}

void Layer::updateOutputs(const Eigen::VectorXd &inputs) {
  m_inputs.head(inputs.size()) = inputs;
  m_intermediateQtys = m_weights.transpose() * m_inputs;
  std::transform(
      m_intermediateQtys.cbegin(), m_intermediateQtys.cend(), m_outputs.begin(),
      [this](auto q) { return m_activationFunction->operator()(q); });
}

void Layer::updateErrors(const Layer &nextLayer) {
  m_errors = nextLayer.weights().topRows(m_outputs.size()) * nextLayer.errors();
  std::transform(m_errors.cbegin(), m_errors.cend(),
                 m_intermediateQtys.cbegin(), m_errors.begin(),
                 [this](auto e, auto q) {
                   return e * m_activationFunction->derivative(q);
                 });
}

void Layer::updateErrors(const Eigen::VectorXd &targets,
                         const CostFunction &costFunction) {
  std::transform(m_outputs.cbegin(), m_outputs.cend(), targets.cbegin(),
                 m_errors.begin(), [&costFunction](auto o, auto t) {
                   return costFunction.derivative(o, t);
                 });
  auto losses{m_outputs.binaryExpr(
      targets, [&costFunction](auto o, auto t) { return costFunction(o, t); })};
  m_loss = losses.sum() / m_numberOfNeurons;
}

void Layer::updateWeights(double learnRate) {
  updateWeights(computeGradients(), learnRate);
}

void Layer::updateWeights(const Eigen::MatrixXd &gradients, double learnRate) {
  m_weights -= learnRate * gradients;
}
