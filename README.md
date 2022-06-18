# c^2:concurrent-networks-for-co-design
This is the official repository for the paper: C^2: Co-design of Robots via Concurrent Networks Coupling Online and Offline Reinforcement Learning
## Archirecture
![image](https://user-images.githubusercontent.com/48233618/174423615-4c291d95-e4de-4868-a3c1-9f8b02955a38.png)

With the rise of computing power, using data-driven approaches for automatic robot design has become a feasible way, e.g., co-design robots’ morphology and controller. Nevertheless, evaluating the fitness of the controller under each morphology is time-consuming. As a pioneering data-driven method, Co-adaptation utilizes a double-network mechanism with the aim of obtaining a trained Q function to replace the traditional evaluation of a diverse set of candidates, thereby speeding up optimization. In this paper, we find that Co-adaptation ignores the existence of exploration error during training and state-action distribution shift during parameter transmitting, which hurt the performance of Coadaptation. We propose the framework of concurrent network that couples online and offline RL methods. By leveraging the behavior cloning term flexibly, we mitigate the impact of the above issues on the optimized performance. Simulation and physical experiments are performed to demonstrate that our proposed method outperforms baseline algorithms, which illustrates that the proposed method is an effective way of discovering the optimal combination of morphology and controller.
## Requirements
+ gym
+ numpy
+ mujoco
+ torch
+ torchvision
+ xmltodict
+ rlkit
+ bayesian-optimization
## Training
```python main.py```  
The hyperparameters can be modified in __experiment_configs.py__.
## Implementation
Our code is mainly based on the official version of Coadaptation：[Coadaptation](https://github.com/ksluck/Coadaptation)  
Part of offline RL is from: [TD3BC](https://github.com/sfujim/TD3_BC)
## Reference
[Data-efficient Co-Adaptation of Morphology and Behaviour with Deep Reinforcement Learning](https://arxiv.org/pdf/1911.06832.pdf)  
[Adaptive Behavior Cloning Regularization for Stable Offline-to-Online Reinforcement Learning](https://offline-rl-neurips.github.io/2021/pdf/30.pdf)
