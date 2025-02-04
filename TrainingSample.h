#pragma once

#include <vector>

struct TrainingSample {
  std::vector<float> inputs;
  std::vector<float> outputs;
};

struct TrainingBatch {
  std::vector<TrainingSample> samples;
};
