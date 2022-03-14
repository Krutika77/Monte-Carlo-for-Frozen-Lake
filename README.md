# Monte-Carlo-for-Frozen-Lake
Reinforcement Learning 
Based on Monte Carlo Method
Problem:
Here we will use techniques based on Monte Carlo (MC) estimators to solve reinforcement learning problems in which we don't know the environmental behavior. Here the algorithm based on MC will learn based on an episode by episode strategy and estimate the state-action values over many episodes to find an optimal/good policy employing First Visit Monte Carlo Control. Here we will use similar approach to implementing the MC but on our FrozenLake8x8 (both slippery and none slippery) and shows the performance of these policies. 


Proposed Solution:
Using the gym toolkit and numpy library, the parameters:
o	env is the initialized Open AI gym environment object with the FrozenLake8x8-v0 and FrozenLake8x8NotSlippery-v0 in the two programs.
o	S is the agent’s state
 
o	V is a vector, the value to be uses for estimation
o	Gamma is the discount factor
o	Policy is a matrix where each cell has the probability of performing an action a in state s.
o	Theta is the threshold
The program employs a value iteration algorithm that computes the optimal state value function by iteratively improving the estimate of V(s).
Following the prescribed algorithm,
•	def calc_values(env, s, V, gamma) – is used to calculate the state-value. It returns a vector containing the estimated value of each action. It computes the values by:
action_values[action] = action_values[action] + probability * (reward + discount_factor_gamma * V[next_state])
•	def create_policy(env, gamma=1.0, theta=0.001) – is employed to create a policy after evaluation. It selects the best action, updates the vector (value function) if there is a change in value and updates the state and policy for optimum performance.

•	def apply_policy(env, n_episodes, policy) – This function applies the updated policy to the real environment by implementing the best actions possible in the current state, executing an episode. It keeps tabs of the goals scored and displays (renders) the steps taken in taken in/for execution of an ongoing episode. It keeps tabs of the goals scored and number of episodes executed in total for final efficiency evaluation.
