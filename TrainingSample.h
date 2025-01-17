#pragma once

#include <vector>

struct TrainingSample {
  std::vector<double> inputs;
  std::vector<double> outputs;
};

struct TrainingBatch {
  std::vector<TrainingSample> samples;
};