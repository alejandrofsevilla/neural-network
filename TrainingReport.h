#pragma once

#include <chrono>
#include <cstddef>

struct TrainingReport {
  std::chrono::milliseconds trainingTime;
  std::size_t epochs;
  double loss;
};
