# Perception: Egocentric Aligned Accumulation

## Objective
The goal is to create a spatiotemporal Bird's Eye View (BEV) feature from multi-view camera inputs over several time steps.

## Challenges Addressed
- **Alignment Issues:** Direct concatenation methods suffer from alignment problems.
- **Height Information Loss:** Some existing methods like FIERY lose important height information during the process.

## Proposed Method
The framework introduces an accumulative ego-centric alignment method, which includes two key steps:

### 1. Spatial Fusion
- **Transformation:** Multi-view images are transformed into a common 3D frame using depth predictions.
- **Feature Extraction:** Features from each camera image are lifted into 3D space based on depth estimations, using the equation:

  ![equation](https://latex.codecogs.com/png.latex?u_i^k%20%3D%20f_i^k%20%5Cotimes%20d_i^k)

  Here, $u_i^k$ represents the 3D features, $f_i^k$ is the feature map, and $d_i^k$ is the depth map.

- **Alignment:** These features are then aligned to the current view using the vehicle’s ego-motion and pooled into BEV features.

### 2. Temporal Fusion
- **Enhancement of Static Object Perception:** A temporal fusion technique is applied to enhance the perception of static objects by using a self-attention mechanism that boosts the importance of features from previous time steps.

  ![equation](https://latex.codecogs.com/png.latex?%5Ctilde%7Bx%7D_t%20%3D%20b_t%20%2B%20%5Csum_%7Bi%3D1%7D%5E%7Bt-1%7D%20%5Calpha_i%20%5Ctimes%20%5Ctilde%7Bx%7D_%7Bt-i%7D)

  Here, $\tilde{x}_t$ represents the accumulated feature, and $b_t$ is the BEV feature map.

- **3D Convolutions:** These fused features are then processed with 3D convolutions to improve the perception of dynamic objects, using the equation:

  ![equation](https://latex.codecogs.com/png.latex?x_%7B1%5Csim%20t%7D%20%3D%20C(%5Ctilde%7Bx%7D_%7B1%5Csim%20t%7D%2C%20m_%7B1%5Csim%20t%7D))

  Where $m_{1\sim t}$ is the ego-motion matrix, and $C$ represents the 3D convolution network.

---

## Prediction: Dual Pathway Probabilistic Future Modeling

### Overview
In dynamic driving environments, predicting future trajectories is challenging due to the uncertainty and stochastic nature of the future. The aim is to model the uncertainty in future predictions by considering the stochastic nature of the driving environment.

### Methodology

#### 1. Uncertainty Modeling
- **Gaussian Distribution:** The future uncertainty is modeled as diagonal Gaussians with a mean ($\mu$) and variance ($\sigma^2$). Here, $\mu$ and $\sigma^2$ represent the latent channels in the model.
- **Sampling During Training:** During training, the system samples from a Gaussian distribution ![equation](https://latex.codecogs.com/png.latex?%5Ceta_t%20%5Csim%20%5Cmathcal%7BN%7D(%5Cmu_t%2C%20%5Csigma_t%5E2)), but during inference, it samples from ![equation](https://latex.codecogs.com/png.latex?%5Ceta_t%20%5Csim%20%5Cmathcal%7BN%7D(%5Cmu_t%2C%200)).

#### 2. Dual Pathway Architecture
- **Pathway A:** Integrates BEV features up to the current timestamp with the uncertainty distribution. This pathway uses historical features as input to a GRU (Gated Recurrent Unit), where the first feature $x_1$ is used as the initial hidden state.
- **Pathway B:** Uses the sampled Gaussian distribution $\eta_t$ as input to a GRU, with the current feature $x_t$ as the initial hidden state.

#### 3. Prediction Combination

  ![equation](https://latex.codecogs.com/png.latex?%5Chat%7Bx%7D_%7Bt%2B1%7D%20%3D%20G(x_t%2C%20%5Ceta_t)%20%5Coplus%20G(x_%7B0%3At%7D))

  Here, $G$ represents the GRU process, and $\oplus$ denotes the combination of these predictions.

- **Future State Predictions:** This combined prediction serves as the base for future state predictions (up to $H$ horizons).

#### 4. Decoding
- **Multi-Head Decoder:** The combined features from both pathways are fed into a decoder, which has multiple output heads. These heads generate different interpretable intermediate representations such as:
  - **Instance Segmentation:** Outputs instance centerness, offset, and future flow for identifying objects like vehicles and pedestrians.
  - **Semantic Segmentation:** Focuses on key actors like vehicles and pedestrians.
  - **HD Map Elements:** Generates interpretable map elements like drivable areas and lane boundaries, which are crucial for autonomous driving.
  - **Cost Volume:** A specific head is designed to represent the cost associated with each possible trajectory within the planning horizon.

---

## Planning Module

### Objective of the Planner
The main goal is to plan a safe and comfortable trajectory that will guide the SDV towards a target point while avoiding obstacles and considering traffic rules.

### Trajectory Sampling and Cost Function

#### Sampling
The system generates a set of possible trajectories using a simplified vehicle model (the bicycle model) and evaluates each trajectory using a cost function.

  ![equation](https://latex.codecogs.com/png.latex?f(%5Ctau%2C%20o%2C%20m%3B%20w)%20%3D%20f_o(%5Ctau%2C%20o%2C%20m%3B%20w_o)%20%2B%20f_v(%5Ctau%3B%20w_v)%20%2B%20f_r(%5Ctau%3B%20w_r))

  This equation describes the total cost function $f(\tau, o, m; w)$, which is a sum of three sub-costs:
  - **$f_o$:** Evaluates the trajectory based on occupancy predictions and map representations, considering safety and compliance with traffic rules.
  - **$f_v$:** Comes from the prediction module and is based on the learned features (e.g., the predicted future states of the environment).
  - **$f_r$:** Considers the overall performance of the trajectory, including comfort (e.g., minimizing sudden jerks or sharp turns) and progress towards the destination.

#### Sub-Costs
- **Safety Cost:** Ensures the SDV avoids collisions with other objects and maintains a safe distance from obstacles, particularly at high speeds.
- **Cost Volume:** A learned representation generated by the prediction module that reflects the complexity of the environment. It is clipped to ensure it doesn't dominate the evaluation of trajectories.
- **Comfort and Progress:** Penalizes trajectories that involve excessive lateral acceleration, jerk, or curvature, and rewards trajectories that efficiently move towards the destination.

### High-Level Commands and Target Information
- **Command-Based Planning:** The cost function does not inherently include target information (e.g., the final destination). Instead, the planner uses high-level commands (like "Go Straight" or "Turn Left") to evaluate and select trajectories that align with the desired action.

### Selecting the Optimal Trajectory

  ![equation](https://latex.codecogs.com/png.latex?%5Ctau%5E*%20%3D%20%5Cunderset%7B%5Ctau_h%7D%7B%5Ctext%7Barg%20min%7D%7D%20f(%5Ctau_h%2C%20o%2C%20m%3B%20w))

  This equation identifies the optimal trajectory $\tau^*$ from the set of possible trajectories $\tau_h$ by minimizing the cost function $f(\tau_h, o, m; w)$.

#### GRU-Based Refinement
- **Post-Selection Refinement:** After selecting the optimal trajectory, the system further refines it using a GRU network. This step integrates information from the front-view camera (such as the status of traffic lights) to ensure the trajectory is safe and appropriate given the current traffic conditions.
- **Adjustment of Trajectory:** The GRU refines the trajectory by processing the trajectory points $\tau^*$ and adjusting them based on real-time visual information from the cameras.

---

## Breakdown of the Loss Function

### Overall Loss Function (Equation 7)

  ![equation](https://latex.codecogs.com/png.latex?L%20%3D%20L_%7Bper%7D%20%2B%20%5Calpha%20L_%7Bpre%7D%20%2B%20%5Cbeta%20L_%7Bpla%7D)

  Components:
  - **$L_{per}$:** Perception loss.
  - **$\alpha L_{pre}$:** Prediction loss, scaled by a learnable weight $\alpha$.
  - **$\beta L_{pla}$:** Planning loss, scaled by a learnable weight $\beta$.

  **Learnable Weights:** $\alpha$ and $\beta$ are not fixed constants but are learnable parameters, allowing the model to dynamically balance the contribution of each loss component based on the gradients during training.

### Perception Loss ($L_{per}$)
- **Segmentation Loss:** Includes the loss for segmenting both current and past frames, as well as mapping losses (e.g., lane and drivable area prediction) and depth prediction.
- **Top-k Cross-Entropy Loss:** Used for semantic segmentation, focusing on the most relevant classes.
- **L2 and L1 Losses:** Used for instance segmentation tasks like centerness supervision and offset/flow prediction.
- **Depth Loss:** Uses a pre-generated depth value from another network for direct supervision.

### Prediction Loss ($L_{pre}$)
- **Semantic and Instance Segmentation:** The prediction module also infers future semantic and instance segmentation, using a similar top-k cross-entropy loss as in the perception task.
- **Discounting Future Losses:** Future predictions are more uncertain, so losses for future timestamps are exponentially discounted to account for this uncertainty.

### Planning Loss ($L_{pla}$)

#### Components
- **Max-Margin Loss:** The model treats expert behavior $\tau_h$ as a positive example and trajectories sampled from the set $\tau$ as negative examples. The max-margin loss helps ensure that the expert behavior is preferred over sampled trajectories.
- **L1 Distance Loss:** Measures the distance between the planned trajectory and the expert trajectory. This loss is used to refine the selected trajectory and bring it closer to what a human expert might choose.

#### Equation (8)

  ![equation](https://latex.codecogs.com/png.latex?L_%7Bpla%7D%20%3D%20%5Cunderset%7B%5Ctau%7D%7B%5Ctext%7Bmax%7D%7D%20%5Bf(%5Ctau_h%2C%20c)%20-%20f(%5Ctau%2C%20c)%20%2B%20d(%5Ctau_h%2C%20%5Ctau)%5D_%2B%20%2B%20d(%5Ctau_h%2C%20%5Ctau_o%5E*))

  **ReLU Function $[\cdot]_+$:** Ensures that the loss is non-negative.

  **Distance $d(\tau_h, \tau)$:** Measures how far the sampled trajectory $\tau$ is from the expert trajectory $\tau_h$. The goal is to minimize this distance for the selected trajectory.
