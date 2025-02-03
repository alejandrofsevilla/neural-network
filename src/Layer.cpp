#include "Layer.h"

#include "ActivationFunction.h"
#include "CostFunction.h"
#include "Options.h"

#include <Eigen/Core>
#include <Eigen/src/Core/util/Constants.h>
#include <algorithm>
#include <iterator>
#include <numeric>
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

#include <iostream>
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
          numberOfInputs + f_bias, numberOfNeurons, []() {
            return randomValue(f_weightInitMinValue, f_weightInitMaxValue);
          })} {}

std::size_t Layer::id() const { return m_id; }

std::size_t Layer::numberOfInputs() const { return m_numberOfInputs; }

std::size_t Layer::numberOfNeurons() const { return m_numberOfNeurons; }

Eigen::MatrixXd Layer::computeGradients() const {
  return m_inputs * m_errors.transpose();
}

double Layer::computeLoss(const Eigen::VectorXd &targets,
                          const CostFunction &costFunction) const {
  auto losses{targets.binaryExpr(m_outputs, [&costFunction](auto t, auto o) {
    return costFunction(o, t);
  })};
  return losses.sum() / m_numberOfNeurons;
}

const Eigen::MatrixXd &Layer::weights() const { return m_weights; }

const Eigen::VectorXd &Layer::computeOutputs(const Eigen::VectorXd &inputs) {
  m_inputs.head(inputs.size()) = inputs;
  m_intermediateQtys = m_weights.transpose() * m_inputs;
  m_outputs = m_intermediateQtys.unaryExpr(
      [this](auto q) { return m_activationFunction->operator()(q); });
  return m_outputs;
}

const Eigen::VectorXd &
Layer::computeErrors(const Layer &nextLayer,
                     const Eigen::VectorXd &nextLayerErrors) {
  m_errors = nextLayer.weights().topRows(m_outputs.size()) * nextLayerErrors;
  m_errors = m_errors.binaryExpr(m_intermediateQtys, [this](auto e, auto q) {
    return e * m_activationFunction->derivative(q);
  });
  return m_errors;
}

const Eigen::VectorXd &Layer::computeErrors(const Eigen::VectorXd &targets,
                                            const CostFunction &costFunction) {
  m_errors = m_intermediateQtys.binaryExpr(
      targets, [this, &costFunction](auto q, auto t) {
        return m_activationFunction->derivative(q) *
               costFunction.derivative(m_activationFunction->operator()(q), t);
      });
  return m_errors;
}

void Layer::updateWeights(double learnRate) {
  updateWeights(computeGradients(), learnRate);
}

void Layer::updateWeights(const Eigen::MatrixXd &gradients, double learnRate) {
  m_weights -= learnRate * gradients;
}
