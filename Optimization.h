#pragma once

#include <memory>

class Optimization {
public:
  enum class Type { GradientDescend, ADAM };

  static std::unique_ptr<Optimization> instance(Type type);

  virtual double correction() = 0;

  void addGradient(double gradient);
  void reset();

protected:
  double m_totalGradient{0.0};
  std::size_t m_numberOfSamples{0};
};

class GradientDescendOptimization : public Optimization {
  double correction() override;
};

class ADAMOptimization : public Optimization {
  double correction() override;

private:
  std::pair<double, double> m_momentEstimates{0.0, 0.0};
};
