#include "Optimization.h"

#include <cassert>
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

double GradientDescendOptimization::correction(double gradient,
                                               std::size_t numberOfSamples) {
  assert(numberOfSamples > 0);
  return gradient / numberOfSamples;
}

double ADAMOptimization::correction(double gradient,
                                    std::size_t numberOfSamples) {
  assert(numberOfSamples > 0);
  m_firstMomentEstimate = m_firstMomentEstimate * f_adamBetaOne +
                          (1. + f_adamBetaOne) * (gradient / numberOfSamples);
  m_secondMomentEstimate = m_secondMomentEstimate * f_adamBetaTwo +
                           (1. + f_adamBetaTwo) * (gradient / numberOfSamples);
  return m_firstMomentEstimate /
         (pow(m_secondMomentEstimate, 0.5) + f_adamEpsilon);
}
