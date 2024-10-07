# Advantage Actor-Critic (A2C)
![Actor-Critic Diagram](diagrams/A2C.png)

# Introduction

In reinforcement learning, the concept of Actor-Critic methods refers to a distinct kind of paradigm where 2 interdependent estimators (one for the policy distribution, and another for the value function) converge at the same time towards a parameter configuration that maximizes the Return.

The return is defined by the following equation, and directly depends on the rewards collected during the **collection phase** in a **Rollout Buffer**, which will be stored together with the **value function** and **advantage function** values for all transitions in the current rollout period.
$$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots$$

The Value function $V(s_t)$ returns the value of each state, which is defined by the difference between the total reward from a given episode $e$ and the compound reward at a given state $s$ in that same episode $e$. We use a Multi-Layer-Perceptron-Based estimator that directly depends on the weights $\theta_v$ to approximate the value function.

$$G_t = r_t + \gamma V_\theta(s_{t+1})$$

In advantage-based methods, an advantage function is employed to compute the Policy Estimator (Actor) Loss' Gradient, while the Value Function Estimator (Critic) Loss' Gradient is computed using the Mean Squared Error between estimated and actual return values. A typical Advantage function should represent how a given action $a$ compares to the average action from a given state $s$, which is formally defined by:
$$A(s, a) = Q(s, a) - V(s)$$
But can be estimated using the TD-Error (as explained in the Collection Phase section) for further control over the Bias-Variance tradeoff during training of the Value function estimate.

## Collection Phase
Conceptually, in the collection phase, **experience tuples** are collected while the model interacts with the environment. These 4-tuples consist of: (state, action, reward, next_state) and are employed during the TD-Error computation that will end up training the "Critic" Value function estimate part of the model.

As seen in the `add` method of the `RolloutBuffer` class in `buffers.py` (stable baselines 3 implementation), on each model-environment interaction, a state observation, an action and the resulting reward are stored, as well as the Value function estimate for each state:  

```python
def add(
        self,
        ...
    ) -> None:
        ...

        self.observations[self.pos] = np.array(obs)
        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        ...
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        ...
```

In order to compute the policy's loss and train the "Actor" part of the network, an Advantage values are computed using the rewards and state value estimates.
Conceptually, it is most common to find the advantage function definition in terms of the **Action-Value function:** $Q(s, a)$:

$$A(s_t, a_t)=Q(s_t, a_t)−V(s_t)$$

However in practice, when using a model-based approach where state-value estiamtes are available (for example by forwarding observations through the critic network), the Advantage values can be extracted using the **Temporal Difference Error** from 2 consecutive states $\delta_t$.  
This value only depends on state value estimates $V(s)$ as well as environment rewards $R_t$. When computing the Advantage, the state value estimate $V(s_t)$ provides a **baseline** for the Advantage values, improving training stability and speeding up learning, especially with imbalanced rewards.  
These values are computed in the `compute_returns_and_advantage` method as part of the collection phase.
   
A more rigurous derivation of this equation from the previous form is available for the reader in appendix A.

$$A(s_t) = \delta_t = (R_{t+1}+\gamma V(s_{t+1}))−V(s_t)$$

```python
def compute_returns_and_advantage(self, ...) -> None:
        ...
        for step in reversed(range(self.buffer_size)):
            ...
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]

            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
            
        self.returns = self.advantages + self.values
```
In Stable-Baselines 3's implementation of the `RolloutBuffer` class (`buffers.py`), Generalized Advantage Estimation (GAE) is employed so the user can specify a `gae_lambda` value which allows for a smooth transition between a high Variance *Monte-Carlo* approach ($\lambda = 1$) and a high Bias *1-step Temporal-Difference* ($\lambda = 0$) approach when computing the **Advantage values**. The latter of which uses only 1-step rewards while the former *Monte-Carlo* approach uses all future rewards in the episode to compute the advantage values.

Instead of computing the advantage directly using only the *1-step TD-error*: ($\delta_t = (r_t+\gamma V(s_{t+1}))−V(s_t)$), $\delta_t$ may be extracted for multiple consecutive timesteps, weighted by the previously mentioned `gae_lambda` value and discounted using the `gamma` factor before being compounded in a weighted sum that computes the Generalized Advantage value for each state:
$$A_t = \sum_{l=0}^{T-t} (\gamma \lambda)^l \delta_{t+l}$$ 

## Training Phase
In Stable Baselines 3's `A2C` implementation, a single update step is performed per rollout over all parallel environments. This means that after `n_steps` are completed per environemnt, the `compute_returns_and_advantage` method is called a single time. 
The model's `train` method will use the gathered experience tuples in order to train the actor and the critic on the rollout data. This code can be found in `a2c.py`'s `A2C` class as well as in `policy.py`'s `ActorCriticPolicy` class in the stable baselines 3 implementation. 

### Policy Evaluation
```python
def train(self) -> None:
    ...
    for rollout_data in self.rollout_buffer.get(batch_size=None):
        actions = rollout_data.actions
        ...

        values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
```
First, the **Policy Evaluation** process takes place, which consist of computing the value estimates for the states (`rollout_data.observations`) by forwarding the states' observations through the critic's network, which goal is to estimate the value function of each state $s$ following the policy $\pi$. These values will be leveraged later during the computation for the Critic's (Value Function Estimator's) loss function's gradients. Conceptually, the policy evaluation process extracts the values estimates by following the expected value of the discounted reward following the policy $\pi$:

$$V^\pi(s) = \mathbb{E}_\pi \left[ \sum_{t=0}^\infty \gamma^t R_{t+1} \mid S_0 = s \right]$$

### Computing Policy Loss
After policy evaluation, the Advantage values collected during the rollout phase are normalized and multiplied by the log-probabilities of the policy distribution in order to extract the policy loss' gradients.
```python
    ...
    # Normalize advantage (not present in the original implementation)
    advantages = rollout_data.advantages
    if self.normalize_advantage:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Policy gradient loss
    policy_loss = -(advantages * log_prob).mean()
```

Which corresponds to:
$$J(\theta_\pi) = -\mathbb{E} \left[\log \pi_\theta(a_t | s_t; \theta_\pi) A_t\right]$$

Where $J(\theta_\pi)$ is the policy loss value, and is computed in practice by averaging the products of the advantages values and the logarithmic probabilities of taking their corresponding actions over a given **batch**.  
The size of a batch is the number of parallelized environments times the number of steps in a rollout, and in the case of A2C's implementation, there is a single batch per update. This means that the policy is evaluated and improved once every *n_steps* timesteps (on all parallel environments).

### Computing Value Loss
The value loss is computed using the MSE between the actual returns and the Policy-Evaluation predicted values:
```python
    # Value loss using the TD(gae_lambda) target
    value_loss = F.mse_loss(rollout_data.returns, values)
```

Conceptually:
$$L(\theta_v) = \frac{1}{2} \mathbb{E} \left[ \left( G_t - V(s_t; \theta_v) \right)^2 \right]$$

Finally, the entropy loss is computed from the policy distribution's Shannon entropy and added to the loss in order to provide control to the user over the exploration vs exploitation tradeoff during training. This control is provided via the ent_coef parameter, which multiplies the entropy loss before adding it to the total loss.

```python
    ...
    # Entropy loss favor exploration
    if entropy is None:
        # Approximate entropy when no analytical form
        entropy_loss = -th.mean(-log_prob)
    else:
        entropy_loss = -th.mean(entropy)

    loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

```

### Backpropagation and gradient step
After the total loss is computed, the backwards method is called in order to back-propagate the loss-function's gradients through the computational graph's tensors until the Actor's and Critic's network weights are reached.  
The gradients are clipped to `max_grad_norm` hyperparameter value with the goal of stabilizing training, before adding the learning-rate-weighted values to the networks' weights during the self.policy.optimizer.step() method call in order to perform a single gradient step.

```python

    # Optimization step
    self.policy.optimizer.zero_grad()
    loss.backward()

    Clip grad norm
    th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
    self.policy.optimizer.step()
```