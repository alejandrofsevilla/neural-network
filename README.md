# [![LinuxWorkflow](https://github.com/alejandrofsevilla/neural-network/actions/workflows/Linux.yml/badge.svg)](https://github.com/alejandrofsevilla/neural-network/actions/workflows/Linux.yml) [![MacOsWorkflow](https://github.com/alejandrofsevilla/neural-network/actions/workflows/MacOs.yml/badge.svg)](https://github.com/alejandrofsevilla/neural-networkboost-tcp-server-client/actions/workflows/MacOs.yml)
# neural-network
Implementation of neural network class.

## Requirements
* C++17 compiler.
* CMake 3.22.0
* GoogleTest 1.11.0
  
## Interface
```cpp
class NeuralNetwork {
public:
  explicit NeuralNetwork(std::size_t numberOfInputs);

  ~NeuralNetwork();

  std::vector<double> computeOutputs(const std::vector<double> &inputs);

  void addLayer(options::LayerConfig config);

  TrainingReport train(options::TrainingConfig config,
                       const TrainingBatch &batch);
```
## Options
```cpp
namespace options {
enum class ActivationFunctionType { Step, Linear, Relu, Sigmoid, TanH };
enum class CostFunctionType { Quadratic, CostEntropy };
enum class OptimizationType { GradientDescend, ADAM, SGD };

struct LayerConfig {
  std::size_t numberOfNeurons;
  options::ActivationFunctionType activationFunction;
};

struct TrainingConfig {
  options::OptimizationType optimization;
  options::CostFunctionType costFunction;
  std::size_t maxEpoch;
  double learnRate;
  double lossGoal;
};
```

## Build and Test
- Clone the repository to your local machine.
   ```terminal
   git clone https://github.com/alejandrofsevilla/neural-network.git
   cd neural-network
   ```
- Build.
   ```terminal
   cmake -S . -B build
   cmake --build build
   ```
- Run tests.
   ```terminal
   ./build/tests/neural-network-tests
   ```
 
## Description
## List of Symbols
$\large s$ *= sample*\
$\large S$ *= number of samples in training batch*\
$\large l$ *= layer*\
$\large L$ *= number of layers*\
$\large n_l$ *= neuron at layer l*\
$\large N_l$ *= number of neurons in layer l*\
$\large w_{n_{l-1}n_l}$ *= weight between neurons* $n_{l-1}$ *and* $n_l$\
$\large b_{n_l}$ *= bias of neuron* $n_l$\
$\large z_{n_l}$ *= intermediate quantity of neuron* $n_l$\
$\large y_{n_l}$ *= output of neuron* $n_l$\
$\large \hat y_{n_l}$ = *target output of neuron* $n_l$\
$\large E_{n_l}$ *= error at neuron* $n_l$\
$\large A_{n_l}$ *= activation function at neuron* $n_l$ *{Binary Step, Linear, ReLU, Sigmoid, Tanh...}*\
$\large C$ *= cost function {MSE, SSE, WSE, NSE...}*\
$\large O$ *= optimization Algorithm {Gradient Descend, ADAM, Quasi Newton Method...}*\
$\large α$ *= learning rate*

## Neuron Equations
<p align="center">
  <img src="https://github.com/user-attachments/assets/fe1d5008-b3ec-4791-8453-9bca7dad3007" />
</p>

### Neuron Intermediate Quantity
$$ \large 
z_{n_l} = \sum_{n_{l-1}}^{N_{l-1}}(w_{n_{l-1}n_l} \cdot y_{n_{l-1}} + b_{n_l}) 
$$
### Neuron Output
$$ \large
y_{n_l} = A_{n_l}\big(z_{n_l}\big)
$$

## Training 
<p align="justify">
Errors of the network are reduced by an optimization algorithm $O$ that uses the derivatives of the cost function ${\partial C}/{\partial {w_{n_{l-1}n_l}}}$ and ${\partial C}/{\partial {b_{n_l}}}$ to periodically update the network weights and biases.
</p>

$$ \large
\Delta w_{n_{l-1}n_l} = - α \cdot O\big(\frac {\partial C}{\partial {w_{n_{l-1}n_l}}}\big)
$$

$$ \large
\Delta b_{n_l} = - α \cdot O\big(\frac {\partial C}{\partial {b_{n_l}}}\big)
$$

### Chain Rule

$$ \large
\frac {\partial C}{\partial {w_{n_{l-1}n_l}}} 
= \frac{\partial C}{\partial z_{n_l}} \cdot \frac{\partial z_{n_l}}{\partial {w_{n_{l-1}n_l}}}
= \frac{\partial C}{\partial z_{n_l}} \cdot y_{n_{l-1}}
= \frac{\partial C}{\partial y_{n_l}} \cdot \frac{\partial y_{n_l}}{\partial z_{n_l}} \cdot y_{n_{l-1}}
= \dot C\big(y_{n_l}, \hat y_{n_l}\big) \cdot \dot A_{n_l}\big(z_{n_l}\big) \cdot y_{n_{l-1}}
$$

$$ \large
\frac {\partial C}{\partial {b_{n_l}}} 
= \frac{\partial C}{\partial z_{n_l}}
= \frac{\partial C}{\partial y_{n_l}} \cdot \frac{\partial y_{n_l}}{\partial z_{n_l}}
= \dot C\big(y_{n_l}, \hat y_{n_l}\big) \cdot \dot A_{n_l}\big(z_{n_l}\big)
$$

### Backpropagation
<p align="justify">
The terms $\dot C (y_{n_l} \hat y_{n_l})$ depend on the output target value for each neuron $\hat y_{n_l}$. Training data set only counts on the value of $\hat y_{n_l}$ for the last layer $l = L$. For all previous layers $l < L$, components $\dot C ( y_{n_l}, \hat y_{n_l})$ are computed as a weighted sum of the neuron errors previously calculated at the next layer $E_{n_{l+1}}$ :
</p>

$$ \large
\dot C \big( y_{n_l}, \hat y_{n_l} \big) = \sum_{n_{l+1}}^{N_{l+1}} w_{n_{l}n_{l+1}} \cdot E_{n_{l+1}}
$$

Neuron error $E_{n_l}$ is defined as:

$$ \large
E_{n_l} = \dot C\big(y_{n_l}, \hat y_{n_l}\big) \cdot \dot A_{n_l}\big(z_{n_l}\big)
$$

## Activation Functions
### Binary Step
<p align="center">
  <img src="https://github.com/user-attachments/assets/e46372d3-e7db-41c5-9229-a773b17a1d9b" alt="drawing" width="500"/>
</p>

$$ \large
\begin{split}A \big(z\big) = \begin{Bmatrix} 1 & z ≥ 0 \\
 0 & z < 0 \end{Bmatrix}\end{split}
$$

$$ \large 
\dot A \big(z\big) = 0
$$

### Linear
<p align="center">
  <img src="https://github.com/user-attachments/assets/f4dceb90-73c4-4e40-83ac-0271ac412cff" alt="drawing" width="500"/>
</p>

$$ \large
A \big(z\big) = z
$$

$$ \large
\dot A \big(z\big) = 1
$$

### ReLU (Rectified Linear Unit)
<p align="center">
  <img src="https://github.com/user-attachments/assets/fc453862-5fd8-43a0-ac5d-57bdf318eee6" alt="drawing" width="500"/>
</p>

$$ \large
\begin{split}A \big(z\big) = \begin{Bmatrix} z & z > 0 \\
 0 & z ≤ 0 \end{Bmatrix}\end{split}
$$

$$ \large
\begin{split}\dot A \big(z\big) = \begin{Bmatrix} 1 & z > 0 \\
 0 & z ≤ 0 \end{Bmatrix}\end{split}
$$

### Leaky ReLU
<p align="center">
  <img src="https://github.com/user-attachments/assets/13983fa3-bfe7-41b7-9c70-f6021eef33a2" alt="drawing" width="500"/>
</p>

$$ \large
\begin{split}A \big(z, \tau \big) = \begin{Bmatrix} z & z > 0 \\
\tau \cdot z & z ≤ 0 \end{Bmatrix}\end{split}
$$

$$ \large
\begin{split}\dot A \big(z, \tau \big) = \begin{Bmatrix} 1 & z > 0 \\
\tau & z ≤ 0 \end{Bmatrix}\end{split}
$$

where typically:

$$ \large \tau=0.01 $$

### Sigmoid
<p align="center">
  <img src="https://github.com/user-attachments/assets/8758a47d-2494-4dde-a513-ae87a4f63d64" alt="drawing" width="500"/>
</p>

$$ \large
A \big(z\big) = \frac{1} {1 + e^{-z}}
$$

$$ \large
\dot A \big(z\big) = A(z) \cdot (1-A(z))
$$

### Tanh
<p align="center">
  <img src="https://github.com/user-attachments/assets/483cf2be-54d2-4a0f-9ca7-45d7252789b5" alt="drawing" width="500"/>
</p>

$$ \large
A \big(z\big) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}
$$

$$ \large
\dot A \big(z\big) = 1 - {A(z)}^2 
$$

## Cost Function

### Quadratic Cost
$$\large C\big(y, \hat y\big) = 1/2 \cdot {\big(y - \hat y\big)^{\small 2}}$$

$$\large\dot C\big(y, \hat y\big) = \big(y - \hat y\big)$$

### Cross Entropy Cost
$$ \large
C\big(y, \hat y\big) = -\big({\hat y} \text{ ln } y + (1 - {\hat y}) \cdot \text{ ln }(1-y)\big)
$$

$$ \large
\dot C\big(y, \hat y\big) = \frac{y - \hat y}{(1-y) \cdot y}
$$

## Optimization Algorithm
### Gradient Descend
Network parameters are updated after every epoch.

$$ \large
O \big( \frac{\partial C}{\partial {w_{n_{l-1}n_l}}} \big) = \frac{1}{S} \cdot \sum_{s}^S{\frac{\partial C}{\partial {w_{n_{l-1}n_l}}}}
$$

### Stochastic Gradient Descend
Network parameters updated after every sample.

$$ \large
O \big( \frac{\partial C}{\partial {w_{n_{l-1}n_l}}} \big) = \frac{\partial C}{\partial {w_{n_{l-1}n_l}}}
$$

### Adaptive Moment Estimation
Network parameters updated after every sample.

$$ \large
O \big( \frac{\partial C}{\partial {w_{n_{l-1}n_l}}} \big) = \frac{m_t}{\sqrt{v_t}+\epsilon}
$$

where:

$$ \large
m_t = \beta_1 \cdot m_{t-1} + (1+\beta_1) \cdot \big( \frac{1}{S} \cdot \sum_{s}^S{\frac{\partial C}{\partial {w_{n_{l-1}n_l}}}} \big)
$$

$$ \large
v_t = \beta_2 \cdot v_{t-1} + (1+\beta_2) \cdot \big( \frac{1}{S} \cdot \sum_{s}^S{\frac{\partial C}{\partial {w_{n_{l-1}n_l}}}} \big)
$$

where typically:

$$ \large m_0 = 0 $$

$$ \large v_0 = 0 $$

$$ \large \epsilon = 1/10^{\small 8} $$

$$ \large \beta_1 = 0.9 $$

$$ \large \beta_2 = 0.999 $$


## References
- http://neuralnetworksanddeeplearning.com/
- https://www.analyticsvidhya.com/blog/2020/04/feature-scaling-machine-learning-normalization-standardization/
- https://comsm0045-applied-deep-learning.github.io/Slides/COMSM0045_05.pdf
- https://towardsdatascience.com/optimizers-for-training-neural-network-59450d71caf6
- https://stats.stackexchange.com/questions/154879/a-list-of-cost-functions-used-in-neural-networks-alongside-applications
- https://en.wikipedia.org/wiki/Activation_function#Table_of_activation_functions
- https://arxiv.org/abs/1502.03167
- https://www.samyzaf.com/ML/rl/qmaze.html


