#include <NeuralNetwork.h>
#include <Options.h>
#include <TrainingSample.h>
#include <gtest/gtest.h>
#include <iostream>
#include <math.h>

namespace {
template <class F> auto generateBatch(F f, double min, double max) {
  TrainingBatch batch;
  auto numberOfSamples{100};
  auto step{(max - min) / (numberOfSamples - 1)};
  for (auto i = 0; i < numberOfSamples; i++) {
    auto x{min + step * i};
    auto y{f(x)};
    batch.samples.push_back(TrainingSample{{x}, {y}});
  }
  return batch;
}

template <class F>
auto generateBatch(F f, double min1, double min2, double max1, double max2) {
  TrainingBatch batch;
  auto numberOfSamples{20};
  auto step1{(max1 - min1) / (numberOfSamples - 1)};
  auto step2{(max2 - min2) / (numberOfSamples - 1)};
  for (auto i = 0; i < numberOfSamples; i++) {
    auto x{min1 + step1 * i};
    for (auto j = 0; j < numberOfSamples; j++) {
      auto y{min2 + step2 * j};
      auto z{f(x, y)};
      batch.samples.push_back(TrainingSample{{x, y}, {z}});
    }
  }
  return batch;
}

template <class F>
inline auto validate(NeuralNetwork &nn, const F &f, double min, double max,
                     double tol) {
  auto maxError{0.0};
  auto numberOfSamples{200};
  auto step{(max - min) / (numberOfSamples - 1)};
  for (auto i = 0; i < numberOfSamples; i++) {
    auto x{min + i * step};
    auto y{nn.computeOutputs({x}).back()};
    auto t{f(x)};
    maxError = std::isnan(y) ? tol : std::max(maxError, abs(y - t));
  }
  EXPECT_LT(maxError, tol);
}

template <class F>
inline auto validate(NeuralNetwork &nn, const F &f, double min1, double min2,
                     double max1, double max2, double tol) {
  auto maxError{0.0};
  auto numberOfSamples{100};
  auto step1{(max1 - min1) / (numberOfSamples - 1)};
  auto step2{(max2 - min2) / (numberOfSamples - 1)};
  for (auto i = 0; i < numberOfSamples; i++) {
    for (auto j = 0; j < numberOfSamples; j++) {
      auto x{min1 + i * step1};
      auto y{min2 + j * step2};
      auto z{nn.computeOutputs({x, y}).back()};
      auto t{f(x, y)};
      maxError = std::isnan(z) ? tol : std::max(maxError, abs(z - t));
    }
  }
  EXPECT_LT(maxError, tol);
}

inline auto print(TrainingReport r) {
  std::cout << "TrainingTime: " << r.trainingTime.count() << "ms" << std::endl;
  std::cout << "MinLoss: " << r.loss << std::endl;
  std::cout << "Epochs: " << r.epochs << std::endl;
}
} // namespace

TEST(NeuralNetworkTest, FunctionCosineWith50NeuronsTanHAndSGDOptimization) {
  auto min{-1.5 * M_PI}, max{2.5 * M_PI};
  NeuralNetwork nn{1};
  nn.addLayer({50, options::ActivationFunctionType::TanH});
  nn.addLayer({1, options::ActivationFunctionType::Linear});
  auto cosineFunction{cosf};
  auto batch{generateBatch(cosineFunction, min, max)};
  print(nn.train({options::OptimizationType::SGD,
                  options::CostFunctionType::Quadratic, 20000, 0.01, 0.0001},
                 batch));
  auto tolerance{0.1};
  validate(nn, cosineFunction, min, max, tolerance);
}

TEST(NeuralNetworkTest, FunctionCubeWith50NeuronsSigmoidAndADAMOptimization) {
  auto min{-2.0}, max{2.0};
  NeuralNetwork nn{1};
  nn.addLayer({50, options::ActivationFunctionType::Sigmoid});
  nn.addLayer({1, options::ActivationFunctionType::Linear});
  auto cubeFunction{std::bind(powf, std::placeholders::_1, 3.0)};
  auto batch{generateBatch(cubeFunction, min, max)};
  print(nn.train({options::OptimizationType::ADAM,
                  options::CostFunctionType::Quadratic, 20000, 0.005, 0.0001},
                 batch));
  auto tolerance{0.1};
  validate(nn, cubeFunction, min, max, tolerance);
}

TEST(NeuralNetworkTest,
     FunctionSqrWith50NeuronsReluAndGradientDescendOptimization) {
  auto min{-2.0}, max{2.0};
  NeuralNetwork nn{1};
  nn.addLayer({50, options::ActivationFunctionType::Relu});
  nn.addLayer({1, options::ActivationFunctionType::Linear});
  auto sqrFunction{std::bind(powf, std::placeholders::_1, 2.0)};
  auto batch{generateBatch(sqrFunction, min, max)};
  print(nn.train({options::OptimizationType::GradientDescend,
                  options::CostFunctionType::Quadratic, 20000, 0.05, 0.0001},
                 batch));
  auto tolerance{0.1};
  validate(nn, sqrFunction, min, max, tolerance);
}

TEST(NeuralNetworkTest,
     FunctionSphereWith25NeuronsTanH25NeuronsTanHAndSGDOptimization) {
  auto min1{-1.0}, min2{-1.0}, max1{1.0}, max2{1.0};
  NeuralNetwork nn{2};
  nn.addLayer({25, options::ActivationFunctionType::TanH});
  nn.addLayer({25, options::ActivationFunctionType::TanH});
  nn.addLayer({1, options::ActivationFunctionType::Linear});
  auto sphereFunction{std::bind(
      [](auto x, auto y) { return sqrt(2. - pow(x, 2.0) - pow(y, 2.0)); },
      std::placeholders::_1, std::placeholders::_2)};
  auto batch{generateBatch(sphereFunction, min1, min2, max1, max2)};
  print(nn.train({options::OptimizationType::SGD,
                  options::CostFunctionType::Quadratic, 20000, 0.05, 0.00001},
                 batch));
  auto tolerance{0.1};
  validate(nn, sphereFunction, min1, min2, max1, max2, tolerance);
}
