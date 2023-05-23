import gymnasium as gym
import numpy as np


# state_value_of_current_policy or policy evaluation
# Generate the value for each state under the policy pi
# Input:
# + transition_table of the environment
#        (like given current state what is the probabilities to go to next state if taking action a)
#        In gymnasium transition table is a 2D array where columns are actions, rows are states
#        Each cell in the table stores a tuple (probability, next state, reward, terminate or done)
# + pi is the policy that is needed to evaluate. This is a function.
#        This function accept an integer which present a state as input
#        and the output will be an integer which is a recommend action.
# + gamma is the discount value (which is an integer)
# + theta is the stop value reference to Sutton and Barto RL Book.
def state_value_of_current_policy(transition_table, pi, gamma, theta=1e-10):
    # Initalize V value for all state at start
    # Later on, this prev_v will save the values for all states of previous round
    prev_v = np.zeros(len(transition_table))
    # try to be as close as the algorithm that is suggested in Sutton and Barto Book
    k_is_converge = False
    # Looping rounds. While the condition is satisfied run another evaluation round
    while not k_is_converge:
        # Initialize V value for this evaluation round
        current_v = np.zeros(len(transition_table))
        # looping for every state
        for state in range(len(transition_table)):
            # get all possibilities of the next state of transition state based on current state and action
            for prob, next_state, reward, done in transition_table[state][pi(state)]:
                # not done is there because if reach termination state
                current_v[state] += prob * (reward + gamma * prev_v[next_state] * (not done))

        # Stopping condition
        if np.max(np.abs(current_v - prev_v)) < theta:
            k_is_converge = True
        # save the current_v for next round iteration (if the while loop condition still satisfy)
        prev_v = current_v.copy()
    return prev_v


# new_policy_pi_from_state_value or Policy Iteration
# Input:
# + transition_table same as above
# + v this is the state value array which contain the evaluation value of all state under a policy
#        It is a 1D array where the index represent state and
#        each cell contain a float which present the value of that state
# + gamma same as above
# Output:
# new_policy_pi which is a function which is similar to pi as describe above
def new_policy_pi_from_state_value(transition_table, v, gamma):
    # Some implementation using Q table
    # Q table is a combination of states and actions
    # Q table is a 2D array.
    # In this implementation, I use 1D array because this is what I understand
    # policy is a collection where each state has a recommendation action. So 1D array makes sense to me look at line 46
    # For better understanding
    new_policy = np.zeros(len(transition_table))
    # Looping through all states
    for state in range(len(transition_table)):
        # Create a place to store all actions' values for a state
        all_actions = np.zeros(len(transition_table[0]))
        # Loop through all available actions
        for action in range(len(transition_table[0])):
            # Initialize a action value
            value = 0.0
            # For that action, loop through all possibilities of that action can generate new state
            for prob, next_state, reward, done in transition_table[state][action]:
                # the value function is cumulative
                value += prob * (reward + gamma * v[next_state] * (not done))
            # Save the action value
            all_actions[action] = value
        # Assign the best action that gains highest value to the state
        new_policy[state] = np.argmax(all_actions)
    # Create a new pi function based on the new policy
    new_policy_pi = lambda state: new_policy[state]
    return new_policy_pi

# policy_iteration_algorithm based on Sutton Barto Book
# Input:
# + transition_table: see description above
# + gamma: see description above
# + theta: see description above
# Output:
# A policy function which accept an integer as movement input and output an integer which present as the recommend move
# Value function for each state
def policy_iteration_algorithm(transition_table, gamma, theta=1e-10):
    # Create a random move for each state to form a dummy policy
    random_move_4_each_state = np.random.choice(list(transition_table[0].keys()), len(transition_table))
    # Create a dummy policy based on previous line (Initial policy)
    # Later on pi_function will store the current policy function for comparing with new policy
    pi_function = lambda state: {s: a for s, a in enumerate(random_move_4_each_state)}[state]
    # set the variable to False
    policy_is_stable = False

    # Run until policy is stable
    while not policy_is_stable:
        # save the previous policy or old policy
        old_pi = {state: pi_function(state) for state in range(len(transition_table))}
        # Evaluate the current policy (or initial policy when first start)
        V = state_value_of_current_policy(transition_table, pi_function, gamma, theta)
        # Make new policy from the state value function (this state value function is based on the current policy)
        new_policy_pi_function = new_policy_pi_from_state_value(transition_table, V, gamma)
        # Check if the policy is stable by stable means that the recommended move for each state does not change
        # in new policy as well as old policy
        if old_pi == {state: new_policy_pi_function(state) for state in range(len(transition_table))}:
            policy_is_stable = True
        # Make new policy pi function as current policy for next iteration if the while condition is still satisfy.
        pi_function = new_policy_pi_function

    return pi_function, V


if __name__ == '__main__':
    # Based on gymnasium environment
    # P has to be transition table of the environment

    # Create FrozenLake environment
    # env = gym.make('FrozenLake-v1', is_slippery=False)
    env = gym.make('CliffWalking-v0')
    # Set seeds for reproducible
    np.random.seed(0)
    env.reset(seed=0)
    # Get the transition table
    P = env.P
    # Get the optimal policy and Value table for the environment
    # Changing the discount value between [0, 1] inclusive.
    # The algorithm behavior change accordingly.
    # Discount = 1 can make algorithm run indefinitely (I guess due to exploitation infinite value)
    pi, V = policy_iteration_algorithm(P, 0.95)
    print(V)
