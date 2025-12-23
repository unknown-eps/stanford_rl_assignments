### MDP Value Iteration and Policy Iteration

import numpy as np
from riverswim import RiverSwim

np.set_printoptions(precision=3)


def bellman_backup(state, action, R, T, gamma, V):
    """
    Perform a single Bellman backup.

    Parameters
    ----------
    state: int
    action: int
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)
    gamma: float
    V: np.array (num_states)

    Returns
    -------
    backup_val: float
    """
    backup_val = 0.0
    ############################
    # YOUR IMPLEMENTATION HERE #
    backup_val = R[state, action] + gamma * np.sum(T[state, action] * V)
    ############################

    return backup_val


def policy_evaluation(policy, R, T, gamma, tol=1e-3):
    """
    Compute the value function induced by a given policy for the input MDP
    Parameters
    ----------
    policy: np.array (num_states)
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)
    gamma: float
    tol: float

    Returns
    -------
    value_function: np.array (num_states)
    """
    num_states, num_actions = R.shape
    value_function = np.zeros(num_states)

    ############################
    # YOUR IMPLEMENTATION HERE #
    max_iter = 5000
    break_loop = False
    for itr in range(max_iter):
        value_function_new = np.zeros(num_states)
        for state_idx in range(num_states):
            value_function_new[state_idx] = bellman_backup(
                state_idx, policy[state_idx], R, T, gamma, value_function
            )
        if np.linalg.norm(value_function - value_function_new, ord=np.inf) < tol:
            value_function = value_function_new
            break_loop = True
        value_function = value_function_new
        if break_loop:
            break
    assert break_loop, "policy evaluation failed to converge."
    ############################
    return value_function


def policy_improvement(policy, R, T, V_policy, gamma):
    """
    Given the value function induced by a given policy, perform policy improvement
    Parameters
    ----------
    policy: np.array (num_states)
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)
    V_policy: np.array (num_states)
    gamma: float

    Returns
    -------
    new_policy: np.array (num_states)
    """
    num_states, num_actions = R.shape
    new_policy = np.zeros(num_states, dtype=int)

    ############################
    # YOUR IMPLEMENTATION HERE #
    Q_policy = np.zeros((num_states, num_actions))
    for state_idx in range(num_states):
        for action_idx in range(num_actions):
            Q_policy[state_idx, action_idx] = bellman_backup(
                state_idx, action_idx, R, T, gamma, V_policy
            )
    new_policy = np.argmax(Q_policy, axis=1)
    ############################
    return new_policy


def policy_iteration(R, T, gamma, tol=1e-3):
    """Runs policy iteration.

    You should call the policy_evaluation() and policy_improvement() methods to
    implement this method.
    Parameters
    ----------
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)

    Returns
    -------
    V_policy: np.array (num_states)
    policy: np.array (num_states)
    """
    num_states, num_actions = R.shape
    V_policy = np.zeros(num_states)
    policy = np.zeros(num_states, dtype=int)
    ############################
    # YOUR IMPLEMENTATION HERE #
    max_iter = 5000
    break_iter = False
    for _ in range(max_iter):
        V_cur_policy = policy_evaluation(policy, R, T, gamma, tol)
        new_policy = policy_improvement(policy, R, T, V_cur_policy, gamma)
        V_policy = V_cur_policy
        if np.array_equal(policy, new_policy):
            break_iter = True
        policy = new_policy
        if break_iter:
            break
    assert break_iter, "policy iteration failed to converge."
    ############################
    return V_policy, policy


def value_iteration(R, T, gamma, tol=1e-3):
    """Runs value iteration.
    Parameters
    ----------
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)

    Returns
    -------
    value_function: np.array (num_states)
    policy: np.array (num_states)
    """
    num_states, num_actions = R.shape
    value_function = np.zeros(num_states)
    policy = np.zeros(num_states, dtype=int)
    ############################
    # YOUR IMPLEMENTATION HERE #
    max_iter = 5000
    break_iter = False
    for _ in range(max_iter):
        value_function_new = np.zeros(num_states)
        for state_idx in range(num_states):
            temp_value = -float("inf")
            for action_idx in range(num_actions):
                cur_val = R[state_idx, action_idx] + gamma * np.sum(
                    T[state_idx, action_idx] * value_function
                )
                temp_value = max(temp_value, cur_val)
            value_function_new[state_idx] = temp_value
        if np.linalg.norm(value_function_new - value_function, ord=np.inf) < tol:
            break_iter = True
        value_function = value_function_new
        if break_iter:
            break
    policy = policy_improvement(policy, R, T, value_function, gamma)
    assert break_iter, "Value iteration failed to converge."
    ############################
    return value_function, policy


# Edit below to run policy and value iteration on different configurations
# You may change the parameters in the functions below
if __name__ == "__main__":
    SEED = 1234

    RIVER_CURRENT = "WEAK"
    # RIVER_CURRENT = 'STRONG'
    assert RIVER_CURRENT in ["WEAK", "MEDIUM", "STRONG"]
    env = RiverSwim(RIVER_CURRENT, SEED)

    R, T = env.get_model()
    discount_factor = 0.99
    # discount_factor = 0.67

    print("\n" + "-" * 25 + "\nBeginning Policy Iteration\n" + "-" * 25)

    V_pi, policy_pi = policy_iteration(R, T, gamma=discount_factor, tol=1e-3)
    print(V_pi)
    print([["L", "R"][a] for a in policy_pi])

    print("\n" + "-" * 25 + "\nBeginning Value Iteration\n" + "-" * 25)

    V_vi, policy_vi = value_iteration(R, T, gamma=discount_factor, tol=1e-3)
    print(V_vi)
    print([["L", "R"][a] for a in policy_vi])

    # V = bellman_backup(1, 1, R, T, discount_factor, V_pi)
    # print(f'V={V}')
