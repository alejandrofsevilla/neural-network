#include "Layer.h"

#include "ActivationFunction.h"
#include "CostFunction.h"

#include <random>

namespace {
constexpr auto f_bias{1};
constexpr auto f_weightInitMinValue{-1.0};
constexpr auto f_weightInitMaxValue{+1.0};

inline auto randomValue(double min, double max) {
  std::mt19937 eng{std::random_device{}()};
  std::uniform_real_distribution<double> dist{min, max};
  return dist(eng);
}
} // namespace

Layer::Layer(std::size_t id, std::size_t numberOfInputs,
             std::size_t numberOfNeurons,
             options::ActivationFunction activationFunction)
    : m_id{id}, m_numberOfInputs{numberOfInputs},
      m_numberOfNeurons{numberOfNeurons},
      m_activationFunction{ActivationFunction::instance(activationFunction)},
      m_outputDerivatives{Eigen::VectorXd::Zero(numberOfNeurons)},
      m_outputs{Eigen::VectorXd::Zero(numberOfNeurons)},
      m_inputs{Eigen::VectorXd::Ones(numberOfInputs + f_bias)},
      m_errors{Eigen::VectorXd::Zero(numberOfNeurons)},
      m_weights{Eigen::MatrixXd::NullaryExpr(
          numberOfInputs + f_bias, numberOfNeurons,
          std::bind(randomValue, f_weightInitMinValue, f_weightInitMaxValue))},
      m_loss{} {}

std::size_t Layer::id() const { return m_id; }

std::size_t Layer::numberOfInputs() const { return m_numberOfInputs; }

std::size_t Layer::numberOfNeurons() const { return m_numberOfNeurons; }

double Layer::loss() const { return m_loss; }

const Eigen::MatrixXd &Layer::weights() const { return m_weights; }

const Eigen::VectorXd &Layer::inputs() const { return m_inputs; }

const Eigen::VectorXd &Layer::outputs() const { return m_outputs; }

const Eigen::VectorXd &Layer::errors() const { return m_errors; }

void Layer::updateWeights(double learnRate) {
  m_weights -= m_inputs * m_errors.transpose() * learnRate;
}

void Layer::updateWeights(const Eigen::MatrixXd &gradients, double learnRate) {
  m_weights -= learnRate * gradients;
}

void Layer::updateOutputs(const Eigen::VectorXd &inputs) {
  m_inputs.head(inputs.size()) = inputs;
  Eigen::VectorXd intermediateQuantities{m_inputs.transpose() * m_weights};
  m_outputs = intermediateQuantities.unaryExpr(
      [this](auto q) { return m_activationFunction->operator()(q); });
  m_outputDerivatives = intermediateQuantities.unaryExpr(
      [this](auto q) { return m_activationFunction->derivative(q); });
}

void Layer::updateErrors(const Layer &nextLayer) {
  auto &nextLayerWeights{nextLayer.weights()};
  m_errors = (nextLayerWeights.topRows(nextLayerWeights.rows() - 1) *
              nextLayer.errors())
                 .cwiseProduct(m_outputDerivatives);
}

void Layer::updateErrorsAndLoss(const Eigen::VectorXd &targets,
                                const CostFunction &costFunction) {
  m_loss = 0;
  m_errors =
      m_outputs.binaryExpr(targets, [this, &costFunction](auto o, auto t) {
        m_loss += costFunction(o, t) / m_numberOfNeurons;
        return costFunction.derivative(o, t);
      });
}
