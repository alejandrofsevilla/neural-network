#ifndef NEURAL_NETWORK_NEURON_TRAINING_DATA_H
#define NEURAL_NETWORK_NEURON_TRAINING_DATA_H

#include <vector>

struct NeuronTrainingData {
  std::vector<double> outputs;
  std::vector<double> outputDerivatives;
  std::vector<double> targets;
};

#endif
