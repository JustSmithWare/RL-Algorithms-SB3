# Reinforcement Learning Concepts and SB3 Implementations

An in-depth exploration of Reinforcement Learning (RL) concepts and a detailed explanation of their implementations using Stable Baselines3 (SB3). This repository delves into three fundamental RL algorithms:

* __Advantage Actor-Critic (A2C)__

* __Proximal Policy Optimization (PPO)__

* __Deep Q-Networks (DQN)__



Included are comprehensive explanations, code walkthroughs, and diagrams to help the reader understand both the theoretical and practical aspects of these algorithms.

Each RL algorithm's foundational concepts as well as implementation is explained in its own markdown file.   
Starting with a summary of the core idea of the algorithm, to later delve into the reward collection and training phases, until reaching the loss computation and gradient backpropagation.

## Contents
* `A2C.md`: Detailed analysis of the A2C algorithm, reviewing Actor-Critic methods and advantage estimation, and its implementation in SB3.   

* `PPO.md`: Comprehensive guide to PPO, including TRPO theoretical foundations and SB3 code structure.    

* `DQN.md`: Exploration of DQN, review of how off-policy, model-free algorithms learn from past experiences and it's implementation.  

* `diagrams/:` Visual representations supporting the concepts discussed.  

## Overview

Reinforcement learning is a pivotal area in machine learning, focusing on how agents can learn to maximize cumulative rewards by exploring an environment while developing creative and novel strategies. This repository serves as a resource for understanding:

* The mathematical underpinnings of A2C, PPO, and DQN.  

* Implementation specifics within the SB3 framework.  

* Practical considerations when applying these algorithms to real-world problems.  

## Getting Started

To make the most of this repository:

1. Prerequisites: The reader should be familiar with Python and basic machine learning concepts like gradients and backpropagation.
2. Environment Setup:
    * Install Python 3.7 or higher.
    * Install the required packages:

        `pip install stable-baselines3`

3. Exploration:
    * Read the content of each file, take notes, and feel free to return whenever you need a review of core RL concepts for your next project.

## Contribution
Contributions are welcome! If you have suggestions, improvements, or additional resources that could enhance this repository, feel free to open an issue or submit a pull request.


## License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

## Acknowledgements
[Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/en/master/)  
[OpenAI Spinning Up](https://spinningup.openai.com/en/latest/)

