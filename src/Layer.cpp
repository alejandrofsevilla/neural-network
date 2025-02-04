#include "Layer.h"

#include "ActivationFunction.h"
#include "CostFunction.h"

#include <random>

namespace {
constexpr auto f_bias{1};
constexpr auto f_weightInitMinValue{-1.0};
constexpr auto f_weightInitMaxValue{+1.0};

inline auto randomValue(float min, float max) {
  std::random_device rd;
  std::mt19937 eng{rd()};
  std::uniform_real_distribution<float> dist{min, max};
  return dist(eng);
}
} // namespace

Layer::Layer(std::size_t id, std::size_t numberOfInputs,
             std::size_t numberOfNeurons,
             options::ActivationFunctionType activationFunction)
    : m_id{id}, m_numberOfInputs{numberOfInputs},
      m_numberOfNeurons{numberOfNeurons},
      m_activationFunction{ActivationFunction::instance(activationFunction)},
      m_outputDerivatives{Eigen::VectorXf::Zero(numberOfNeurons)},
      m_outputs{Eigen::VectorXf::Zero(numberOfNeurons)},
      m_inputs{Eigen::VectorXf::Ones(numberOfInputs + f_bias)},
      m_errors{Eigen::VectorXf::Zero(numberOfNeurons)},
      m_weights{Eigen::MatrixXf::NullaryExpr(
          numberOfInputs + f_bias, numberOfNeurons,
          []() {
            return randomValue(f_weightInitMinValue, f_weightInitMaxValue);
          })},
      m_loss{} {}

std::size_t Layer::id() const { return m_id; }

std::size_t Layer::numberOfInputs() const { return m_numberOfInputs; }

std::size_t Layer::numberOfNeurons() const { return m_numberOfNeurons; }

float Layer::loss() const { return m_loss; }

const Eigen::MatrixXf &Layer::weights() const { return m_weights; }

const Eigen::VectorXf &Layer::inputs() const { return m_inputs; }

const Eigen::VectorXf &Layer::outputs() const { return m_outputs; }

const Eigen::VectorXf &Layer::errors() const { return m_errors; }

void Layer::updateOutputs(const Eigen::VectorXf &inputs) {
  m_inputs.head(inputs.size()) = inputs;
  Eigen::VectorXf intermediateQtys = m_weights.transpose() * m_inputs;
  for (auto i = 0; i < intermediateQtys.size(); i++) {
    auto q{intermediateQtys[i]};
    m_outputs[i] = m_activationFunction->operator()(q);
    m_outputDerivatives[i] = m_activationFunction->derivative(q);
  } // TODO: Nullaryexpr
}

void Layer::updateErrors(const Layer &nextLayer) {
  auto &nextLayerWeights{nextLayer.weights()};
  m_errors = (nextLayerWeights.topRows(nextLayerWeights.rows() - 1) *
              nextLayer.errors())
                 .cwiseProduct(m_outputDerivatives);
}

void Layer::updateErrorsAndLoss(const Eigen::VectorXf &targets,
                                const CostFunction &costFunction) {
  m_loss = 0;
  for (auto i = 0; i < m_outputs.size(); i++) {
    m_errors[i] = costFunction.derivative(m_outputs[i], targets[i]);
    m_loss += costFunction(m_outputs[i], targets[i]) / m_numberOfNeurons;
  } // TODO: Nullaryexpr
}

void Layer::updateWeights(float learnRate) {
  m_weights -= m_inputs * m_errors.transpose() * learnRate;
}

void Layer::updateWeights(const Eigen::MatrixXf &gradients, float learnRate) {
  m_weights -= learnRate * gradients;
}
