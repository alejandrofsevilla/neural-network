#include "Optimization.h"

#include <cmath>

namespace {
constexpr auto f_adamBetaOne{0.9};
constexpr auto f_adamBetaTwo{0.999};
constexpr auto f_adamEpsilon{0.00000001};
} // namespace

std::unique_ptr<Optimization> Optimization::instance(Type type) {
  switch (type) {
  case Optimization::Type::GradientDescend:
    return std::make_unique<GradientDescendOptimization>();
  case Optimization::Type::ADAM:
    return std::make_unique<ADAMOptimization>();
  default:
    return {};
  }
}

void Optimization::addGradient(double gradient) {
  m_totalGradient += gradient;
  m_numberOfSamples++;
}

void Optimization::reset() {
  m_totalGradient = 0.0;
  m_numberOfSamples = 0;
}

double GradientDescendOptimization::correction() {
  return m_totalGradient / m_numberOfSamples;
}

double ADAMOptimization::correction() {
  auto averageGradient{m_totalGradient / m_numberOfSamples};
  m_momentEstimates.first = m_momentEstimates.first * f_adamBetaOne +
                            (1.0 + f_adamBetaOne) * averageGradient;
  m_momentEstimates.second = m_momentEstimates.second * f_adamBetaTwo +
                             (1.0 + f_adamBetaTwo) * averageGradient;
  return m_momentEstimates.first /
         (pow(m_momentEstimates.second, 0.5) + f_adamEpsilon);
}
