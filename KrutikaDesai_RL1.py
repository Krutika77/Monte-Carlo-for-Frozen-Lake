import gym
import numpy as np

def calc_values(env, s, V, gamma):
    a_values = np.zeros(env.nA)
    for a in range(env.nA):
        for P, next_s, r, end in env.P[s][a]:
            a_values[a] += P * (r + gamma * V[next_s])
    return a_values

def create_policy(env, gamma=1.0, theta=0.001):
    # Initialize state-value function V
    V = np.zeros(env.nS)
    for i in range(100000):
        # Stopping condition
        flag = 0
        # Update each state
        for s in range(env.nS):
            a_value = calc_values(env, s, V, gamma)
            # Select best action based on the highest state-action value
            best_a = np.max(a_value)
            # Find the change in value and update the value function
            flag = max(flag, np.abs(V[s] - best_a))
            V[s] = best_a
    # Create a policy 
    policy = np.zeros([env.nS, env.nA])
    for s in range(env.nS):
        # Get best action for this state
        a_value = calc_values(env, s, V, gamma)
        # Select best a based on the highest s-a value
        best_a = np.argmax(a_value)
        # Update the policy for optimal performance
        policy[s, best_a] = 1.0
    return policy, V

def apply_policy(env, n_episodes, policy):
    goals = 0
    total_r = 0
    env.render()
    for episode in range(n_episodes):

        end = False
        s = env.reset()

        while not end:
            env.render()
            # Select best action in current state
            a = np.argmax(policy[s])

            # Responce
            next_s, r, end, info = env.step(a)

            total_r += r
            # Updates state
            s = next_s
            # Number of goals over episodes
            if end and r == 1.0:
                goals += 1
                env.render()
                
    avg_r = total_r / n_episodes

    return goals, total_r, avg_r

# Number of episodes 
n_episodes = 10000
env = gym.make('FrozenLake8x8-v0')
policy, V = create_policy(env.env)
goals, total_r, avg_r = apply_policy(env, n_episodes, policy)
print(f'Total number of goals achieved over {n_episodes} episodes = {goals}')
print(f'Avg reward earned = {avg_r} \n\n')