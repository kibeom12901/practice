# Perception: Egocentric Aligned Accumulation

## Objective

The goal here is to create a spatiotemporal Bird's Eye View (BEV) feature from multi-view camera inputs over several time steps.

## Challenges Addressed

- **Alignment Issues**: Direct concatenation methods suffer from alignment problems.
- **Height Information Loss**: Some existing methods like FIERY lose important height information during the process.

## Proposed Method

The framework introduces an accumulative ego-centric alignment method, which includes two key steps:

### 1. Spatial Fusion

- **Transformation**: Multi-view images are transformed into a common 3D frame using depth predictions.
- **Feature Extraction**: Features from each camera image are lifted into 3D space based on depth estimations, using the equation:

  <img width="143" alt="Screenshot 2024-08-27 at 4 48 26 PM" src="https://github.com/user-attachments/assets/8e8f7582-b79f-4739-9087-6b23d8e3ef05">

  Here, <img width="26" alt="Screenshot 2024-08-27 at 4 46 46 PM" src="https://github.com/user-attachments/assets/ecafd953-3fb4-400f-bdd7-0046e00b4686"> represents the 3D features, <img width="24" alt="Screenshot 2024-08-27 at 4 47 12 PM" src="https://github.com/user-attachments/assets/59193035-3c51-486d-894d-0a3c998c6629"> is the feature map, and <img width="22" alt="Screenshot 2024-08-27 at 4 47 32 PM" src="https://github.com/user-attachments/assets/e0a28c4c-8e12-4f7c-bb4a-9cd55f14b3a2"> is the depth map.

- **Alignment**: These features are then aligned to the current view using the vehicle’s ego-motion and pooled into BEV features.

### 2. Temporal Fusion

- **Enhancement of Static Object Perception**: A temporal fusion technique is applied to enhance the perception of static objects by using a self-attention mechanism that boosts the importance of features from previous time steps.
  
  Equation:
  
  ![equation](https://latex.codecogs.com/png.latex?\tilde{x}_t=b_t+%5Csum%20%5Climits_%7Bi=1%7D%5E%7Bt-1%7D%20%5Calpha_i%20%5Ctimes%20%5Ctilde{x}_{t-i})

  Here, ![equation](https://latex.codecogs.com/png.latex?\tilde{x}_t) represents the accumulated feature, and ![equation](https://latex.codecogs.com/png.latex?b_t) is the BEV feature map.

- **3D Convolutions**: These fused features are then processed with 3D convolutions to improve the perception of dynamic objects, using the equation:

  ![equation](https://latex.codecogs.com/png.latex?\tilde{x}_{1%5Esim%20t}=C(%5Ctilde{x}_{1%5Esim%20t},m_{1%5Esim%20t}))

  Where ![equation](https://latex.codecogs.com/png.latex?m_{1\sim%20t}) is the ego-motion matrix, and ![equation](https://latex.codecogs.com/png.latex?C) represents the 3D convolution network.

## Prediction: Dual Pathway Probabilistic Future Modeling

### Overview

In dynamic driving environments, predicting future trajectories is challenging due to the uncertainty and stochastic nature of the future.

### Objective

The aim is to model the uncertainty in future predictions by considering the stochastic nature of the driving environment.

### Methodology

#### 1. Uncertainty Modeling

- **Gaussian Distribution**: The future uncertainty is modeled as diagonal Gaussians with a mean (\( \mu \)) and variance (\( \sigma^2 \)):

  ![equation](https://latex.codecogs.com/png.latex?\eta_t%5Csim%20N(\mu_t,%20\sigma_t^2))

  During inference, the system samples from:

  ![equation](https://latex.codecogs.com/png.latex?\eta_t%5Csim%20N(\mu_t,%200))

#### 2. Dual Pathway Architecture

- **Pathway a**: Integrates BEV features up to the current timestamp with the uncertainty distribution.
- **Pathway b**: Uses the sampled Gaussian distribution ![equation](https://latex.codecogs.com/png.latex?\eta_t) as input to a GRU, with the current feature ![equation](https://latex.codecogs.com/png.latex?x_t) as the initial hidden state.

#### 3. Prediction Combination

- **Equation**:

  ![equation](https://latex.codecogs.com/png.latex?\hat{x}_{t+1}=G(x_t,\eta_t)\oplus%20G(x_{0:t}))

  Where ![equation](https://latex.codecogs.com/png.latex?G) represents the GRU process, and ![equation](https://latex.codecogs.com/png.latex?\oplus) denotes the combination of these predictions.

### Future State Predictions

This combined prediction serves as the base for future state predictions (up to \( H \) horizons).

#### 4. Decoding

- **Multi-Head Decoder**: The combined features from both pathways are fed into a decoder, which has multiple output heads generating different interpretable intermediate representations.

## Planning Module

### Objective of the Planner

The main goal is to plan a safe and comfortable trajectory that will guide the SDV towards a target point while avoiding obstacles and considering traffic rules.

### Trajectory Sampling and Cost Function

- **Sampling**: The system generates a set of possible trajectories using a simplified vehicle model (the bicycle model) and evaluates each trajectory using a cost function:

  ![equation](https://latex.codecogs.com/png.latex?f(\tau,%20o,%20m;%20w)%20=%20f_o(\tau,%20o,%20m;%20w_o)%20+%20f_v(\tau;%20w_v)%20+%20f_r(\tau;%20w_r))

  This equation describes the total cost function ![equation](https://latex.codecogs.com/png.latex?f(\tau,o,m;w)), which is a sum of three sub-costs:
  - ![equation](https://latex.codecogs.com/png.latex?f_o): Safety and traffic compliance cost.
  - ![equation](https://latex.codecogs.com/png.latex?f_v): Prediction module cost based on learned features.
  - ![equation](https://latex.codecogs.com/png.latex?f_r): Performance cost including comfort and progress.

### Selecting the Optimal Trajectory

- **Equation**:

  ![equation](https://latex.codecogs.com/png.latex?\tau^*=arg%20min%20_{\tau_h}%20f(\tau_h,%20o,%20m;%20w))

  This equation identifies the optimal trajectory ![equation](https://latex.codecogs.com/png.latex?\tau^*) by minimizing the cost function.

### GRU-Based Refinement

- **Post-Selection Refinement**: After selecting the optimal trajectory, the system further refines it using a GRU network.

## Breakdown of the Loss Function

### Overall Loss Function (Equation 7)

- **Equation**:

  ![equation](https://latex.codecogs.com/png.latex?L=L_{per}+\alpha%20L_{pre}+\beta%20L_{pla})

  Where:
  - ![equation](https://latex.codecogs.com/png.latex?L_{per}): Perception loss.
  - ![equation](https://latex.codecogs.com/png.latex?\alpha%20L_{pre}): Prediction loss, scaled by a learnable weight ![equation](https://latex.codecogs.com/png.latex?\alpha).
  - ![equation](https://latex.codecogs.com/png.latex?\beta%20L_{pla}): Planning loss, scaled by a learnable weight ![equation](https://latex.codecogs.com/png.latex?\beta).

### Perception Loss

- **Components**: Segmentation loss, top-k cross-entropy loss, L2/L1 losses, and depth loss.

### Prediction Loss

- **Components**: Semantic and instance segmentation, discounting future losses.

### Planning Loss

- **Equation (8)**:

  ![equation](https://latex.codecogs.com/png.latex?L_{pla}=%5Cmax%5Climits_{\tau}%20%5Bf(\tau_h,%20c)-f(\tau,%20c)+d(\tau_h,%20\tau)%5D^{+}+d(\tau_h,%20\tau_o^*))

  The loss ensures that the selected trajectory is optimal and aligned with expert behavior.

---

This README provides an overview of the methodology, challenges addressed, proposed solutions, and detailed equations for the system. For further details, please refer to the full documentation and source code.
