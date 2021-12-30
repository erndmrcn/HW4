# Eren Demircan - 2237246
# CEng462 - AI HW4
# Grid problem MDP solver

# copy module for deepcopy on list/dictionary
import copy

# global variables and structures
# height and width of the grid world
M, N = 0, 0

# survival reward
reward = 0.0

# list contains all obstacles' positions
obstacle_states = []

# list contains all goal states' positions
goal_state = []

# discount variable
gamma = 0.0
epsilon = 0.0

# total number of iteration 
iteration = 0

# grid is a dictionary
# terminal states initialized with their values
# obstacle states are indicated by 'obstacle'
grid = dict()

# utility functions

# read into global variables and structures
def readFile(file_name):
    global M, N, reward, action_noise, epsilon, gamma, iteration, obstacle_states, goal_state
    with open(file_name) as f:
        lines = f.readlines()

    lines = [a.replace('\n', '') for a in lines]
    # environment
    if lines[0] == '[environment]': 
        ints = lines[1].split(' ')
        M = int(ints[0])         
        N = int(ints[1])       

    # obstacle states
    if lines[2] == '[obstacle states]':
        obstacle_states = lines[3].split('|')
    
    # goal states
    if lines[4] == '[goal states]':
        goal_state = lines[5].split('|')
    
    # reward
    if lines[6] == '[reward]':
        reward = float(lines[7])

    # action noise
    if lines[8] == '[action noise]':
        action_noise = (float(lines[9]), float(lines[10]), float(lines[11]))

    # gamma
    if lines[12] == '[gamma]':
        gamma = float(lines[13])
    
    # epsilon
    if lines[14] == '[epsilon]':
        epsilon = float(lines[15])

    # iteration
    if lines[16] == '[iteration]':
        iteration = int(lines[17])
    return


# going up helper
def go_up(first_dim, second_dim):
    up = str((first_dim - 1, second_dim))
    # if up state is a valid state (non-obstacle and inside grid), add the action
    if up not in obstacle_states_new and up in states:
        return up
    else:
        return str((first_dim, second_dim))


# going down helper
def go_down(first_dim, second_dim):
    down = str((first_dim + 1, second_dim))
    # if down state is a valid state (non-obstacle and inside grid), add the action
    if down not in obstacle_states_new and down in states:
        return down
    else:
        return str((first_dim, second_dim))


# going left helper
def go_left(first_dim, second_dim):
    left = str((first_dim, second_dim - 1))
    # if left state is a valid state (non-obstacle and inside grid), add the action
    if left not in obstacle_states_new and left in states:
        return left
    else:
        return str((first_dim, second_dim))


# going right helper
def go_right(first_dim, second_dim):
    right = str((first_dim, second_dim + 1))
    # if right state is a valid state (non-obstacle and inside grid), add the action
    if right not in obstacle_states_new and right in states:
        return right
    else:
        return str((first_dim, second_dim))


# from given state take given action and return the resultant state's reward
def go(state, action):
    # 0.8 probability take selected action, up    , down , left, right
    # 0.1 counter clockwise 90 degree,      left  , right, down, up
    # 0.1 clockwise 90 degree,              right , left , up  , down

    # state -> (a, b)
    temp = state.split(',')
    first_dim = int(temp[0][1:])  # -> firstDim
    second_dim = int(temp[1][:-1])  # -> secondDim

    if action == 'up':
        new_state_1 = go_up(first_dim, second_dim)      # selected action
        new_state_2 = go_left(first_dim, second_dim)    # counter-cw action
        new_state_3 = go_right(first_dim, second_dim)   # clockwise action
    elif action == 'down':
        new_state_1 = go_down(first_dim, second_dim)    # selected action
        new_state_2 = go_right(first_dim, second_dim)   # counter-cw action
        new_state_3 = go_left(first_dim, second_dim)    # clockwise action
    elif action == 'left':
        new_state_1 = go_left(first_dim, second_dim)    # selected action
        new_state_2 = go_down(first_dim, second_dim)    # counter-cw action
        new_state_3 = go_up(first_dim, second_dim)      # clockwise action
    elif action == 'right':
        new_state_1 = go_right(first_dim, second_dim)   # selected action
        new_state_2 = go_up(first_dim, second_dim)      # counter-cw action
        new_state_3 = go_down(first_dim, second_dim)    # clockwise action
    # for exit, obstacle and other type of attacks
    else:
        # obstacle state
        # do nothing
        return [(float(0), state)]

    if len(action_noise) > 1:
        return [(action_noise[0], new_state_1), (action_noise[1], new_state_2), (action_noise[2], new_state_3)]
    elif len(action_noise) == 1:
        return [(action_noise[0], new_state_1)]
    else:
        exit('incorrect action noise')


# clear all global variables
def clear():
    M, N = 0, 0
    reward = 0.0
    obstacle_states = []
    obstacle_states_new = []
    goal_state = []
    goal_state_new = []
    gamma = 0.0
    epsilon = 0.0
    iteration = 0
    grid = dict()
    return


# update grid values
# update obstacle and goal state values
def update_grid():
    global obstacle_states_new, goal_state_new, states
    
    # update grid variable
    for i in range(M):
        for j in range(N):
            grid[str((i,j))] = float(reward)
    
    # update goal states
    goal_state_new = []
    for state in goal_state:
        s = state.split(':')
        s[0] = s[0].replace(',', ', ')
        goal_state_new.append(s[0])
        grid[s[0]] = float(s[1])

    # update obstacle states
    obstacle_states_new = []
    for state in obstacle_states:
        state = state.replace(',', ', ')
        obstacle_states_new.append(state)
        grid[state] = 'Obstacle'
    
    states = list(grid.keys())
    
    return


# change actions from strings to symbols
def symbol(pi):
    temp_dict = {}
    for s in states:
        l = s.split(',')
        
        f = int(l[0][1:])
        ss = int(l[1][:-1])
        if pi[s] == 'up':
            temp_dict.update({(f, ss): '^'})
        elif pi[s] == 'down':
            temp_dict.update({(f, ss): 'V'})
        elif pi[s] == 'right':
            temp_dict.update({(f, ss): '>'})
        elif pi[s] == 'left':
            temp_dict.update({(f, ss): '<'})
        else:
            continue

    temp_dict = dict(sorted(temp_dict.items()))
    return temp_dict


# transition function
# function T -> inputs: s, a -> current state and action taken
#               returns: p, s1 -> probability of state gone to and the state gone to
def T(state, action):
    return go(state, action) 


# reward function
# function R -> inputs: s -> current state
#               returns: reward for that state, -0.04 for non terminal states
def R(state):
    rew = grid[state]
    if rew == 'Obstacle':
        return float(0)
    else:
        return float(rew)


# Action function
# function A -> inputs: s -> current state
#               returns: a -> all possible actions from the current state
def A(state):
    if state in goal_state_new:
        # only exit action possible
        return ['exit']
    if state in obstacle_states_new:
        # cannot enter this state but check anyway
        # no possible actions from this state
        return ['Obstacle']
    else:
        l = state.split(',')
        m = int(l[0][1:])
        n = int(l[1][:-1])
        actions = []

        up = str((m - 1, n))
        down = str((m + 1, n))
        right = str((m, n + 1))
        left = str((m, n - 1))

        if up in states:
            actions.append('up')
        if right in states:
            actions.append('right') 
        if down in states:
            actions.append('down')
        if left in states:
            actions.append('left')

        return actions


def roundValues(u):
    result = {}
    for s in states:
        l = s.split(',')

        f = int(l[0][1:])
        ss = int(l[1][:-1])
        u[s] = round(u[s], 2)
        result.update({(f, ss): u[s]})

    result = dict(sorted(result.items()))
    return result


def expected_utility(a, s, U):
    return sum(p * U[s1] for (p, s1) in T(s, a))


def policy_evaluation(pi, U, k):
    for i in range(k):
        for s in states:
            U[s] = R(s) + gamma * sum(p * U[s1] for (p, s1) in T(s, pi[s]))     
    return U


# given the U, find the optimal policy
def best_policy(U):
    pi = {}
    for s in states:
        pi[s] = max(A(s), key=lambda a: expected_utility(a, s, U))
    return pi


# implement value iteration algorithm
def valueIteration():
    update_grid()

    # initialize to 0
    U1 = {s: 0 for s in states}

    # main loop
    while True:
        U = U1.copy()
        delta = 0
        for s in states:
            U1[s] = R(s) + gamma * max(sum(p * U[s1] for (p, s1) in T(s, a)) 
                                                for a in A(s))

            delta = max(delta, abs(U1[s] - U[s]))
        
        if delta <= epsilon * (1 - gamma) / gamma:
            return U, best_policy(U)

    return


def policyIteration():
    update_grid()
    # initialze to zero
    U = {s: 0 for s in grid.keys()}

    # random-like policy initialization
    l = states
    pi = {}
    for i in range(len(l)):
        s = l[i]
        actions = A(s)
        remainder = i % len(actions)
        pi.update({s: actions[remainder]})

    while True:
        U = policy_evaluation(pi, U, 30)
        unchanged = True
        for s in l:
            a = max(A(s), key=lambda a: expected_utility(a, s, U))
            if a != pi[s]:
                pi[s] = a
                unchanged = False
        if unchanged:
            return pi, U


# function to be called
def SolveMDP(method_name, problem_file_name):
    readFile(problem_file_name)

    # call value iteration method
    if method_name == 'ValueIteration':
        result = valueIteration()

        # clear all globals
        clear()
        return roundValues(result[0]), symbol(result[1])

    # call policy iteration method
    elif method_name == 'PolicyIteration':
        result = policyIteration()
        # clear all globals
        clear()
        return roundValues(result[1]), symbol(result[0])


# print(SolveMDP('PolicyIteration', 'mdp1.txt'))