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
# total numbe rof iteration 
iteration = 0

first = False
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

def go_up(first_dim, second_dim):
    up = str((first_dim - 1, second_dim))
    if up not in obstacle_states_new and up in list(grid.keys()):
        return up
    else:
        return str((first_dim, second_dim))

def go_down(first_dim, second_dim):
    down = str((first_dim + 1, second_dim))
    if down not in obstacle_states_new and down in list(grid.keys()):
        return down
    else:
        return str((first_dim, second_dim))
    # if first_dim >= M - 1:
    #     return str((first_dim, second_dim))
    # else:
    #     return str((first_dim + 1, second_dim))

def go_left(first_dim, second_dim):
    left = str((first_dim, second_dim - 1))
    if left not in obstacle_states_new and left in list(grid.keys()):
        return left
    else:
        return str((first_dim, second_dim))
    # if second_dim <= 0:
    #     return str((first_dim, second_dim))
    # else:
    #     return str((first_dim, second_dim - 1))

def go_right(first_dim, second_dim):
    right = str((first_dim, second_dim + 1))
    if right not in obstacle_states_new and right in list(grid.keys()):
        return right
    else:
        return str((first_dim, second_dim))
    # if second_dim >= N - 1:
    #     return str((first_dim, second_dim))
    # else:
    #     return str((first_dim, second_dim + 1))

# from given state take given action and return the resultant state's reward
def go(state, action):
    global first
    # 0.8 probability take selected action, up    , down , left, right
    # 0.1 counter clockwise 90 degree,      left  , right, down, up
    # 0.1 clockwise 90 degree,              right , left , up  , down

    # state -> (a, b)
    first_dim = int(state[1:2])  # -> firstDim
    second_dim = int(state[4:5])  # -> secondDim
    curr = (first_dim, second_dim)

    if action == 'up':
        new_state_1 = go_up(first_dim, second_dim)
        new_state_2 = go_left(first_dim, second_dim)
        new_state_3 = go_right(first_dim, second_dim)
    elif action == 'down':
        new_state_1 = go_down(first_dim, second_dim)
        new_state_2 = go_right(first_dim, second_dim)
        new_state_3 = go_left(first_dim, second_dim)
    elif action == 'left':
        new_state_1 = go_left(first_dim, second_dim)
        new_state_2 = go_down(first_dim, second_dim)
        new_state_3 = go_up(first_dim, second_dim)
    elif action == 'right':
        new_state_1 = go_right(first_dim, second_dim)
        new_state_2 = go_up(first_dim, second_dim)
        new_state_3 = go_down(first_dim, second_dim)
    elif action == 'exit':
        # terminal states
        if not first:
            first = True
            return [(float(grid[state]), state)]
        return [(0.0, state)]
    elif action == 'Obstacle':
        return [(0.0, state)]
    else:
        # obstacle state
        # do nothing
        print('Wrong action is taken. Returning empty list.')
        return [(0.0, state)]

    if len(action_noise) > 1:
        return [(action_noise[0], new_state_1), (action_noise[1], new_state_2), (action_noise[2], new_state_3)]
    elif len(action_noise) == 1:
        return [(action_noise[0], new_state_1)]
    else:
        print('action noise problem. Returning \'None\'.')
        return []

# function T -> inputs: s, a -> current state and action taken
#               returns: p, s1 -> probability of state gone to and the state gone to
def T(state, action):
    return go(state, action) 

# function R -> inputs: s -> current state
#               returns: reward for that state, -0.04 for non terminal states
def R(state):
    rew = grid[state]
    if rew == 'Obstacle':
        return float(0)
    else:
        return float(rew)

# function A -> inputs: s -> current state
#               returns: a -> all possible actions from the current state
def A(state):
    if state in goal_state_new:
        # only exit action possible
        return ['exit']
    elif state in obstacle_states_new:
        # cannot enter this state but check anyway
        # no possible actions from this state
        return ['Obstacle']
    else:
        m = int(state[1:2])
        n = int(state[4:5])
        actions = []

        up = str((m - 1, n))
        down = str((m + 1, n))
        right = str((m, n + 1))
        left = str((m, n - 1))

        states = list(grid.keys())

        if up in states and up not in obstacle_states_new:
            actions.append('up')
        if down in states and down not in obstacle_states_new:
            actions.append('down')
        if left in states and left not in obstacle_states_new:
            actions.append('left')
        if right in states and right not in obstacle_states_new:
            actions.append('right') 

        print(actions)
        return actions

def expected_utility(a, s, U):
    return sum(p * U[s1] for (p, s1) in T(s, a))

def policy_evaluation(pi, U, k):
    for i in range(k):
        for s in list(grid.keys()):
            U[s] = R(s) + gamma * sum(p * U[s1] for (p, s1) in T(s, pi[s]))
        
    return U

# implement value iteration algorithm
def valueIteration():
    global obstacle_states_new, goal_state_new, grid
    
    for i in range(M):
        for j in range(N):
            grid[str((i,j))] = float(reward)
    
    goal_state_new = []
    for state in goal_state:
        s = state.split(':')
        s[0] = s[0].replace(',', ', ')
        goal_state_new.append(s[0])
        grid[s[0]] = float(s[1])

    obstacle_states_new = []
    for state in obstacle_states:
        state = state.replace(',', ', ')
        obstacle_states_new.append(state)
        grid[state] = 'Obstacle'

    U1 = {s: 0 for s in list(grid.keys())}

    while True:
        U = U1.copy()
        delta = 0
        for s in list(grid.keys()):
            U1[s] = R(s) + gamma * max(sum(p * U[s1] for (p, s1) in T(s, a)) 
                                                for a in A(s))

            delta = max(delta, abs(U1[s] - U[s]))
        if delta <= epsilon * (1 - gamma) / gamma:
            return U

    # then extract policies
    
    return

def policyIteration():
    global goal_state_new, obstacle_states_new

    for i in range(M):
        for j in range(N):
            grid[str((i,j))] = float(reward)
    
    goal_state_new = []
    for state in goal_state:
        s = state.split(':')
        s[0] = s[0].replace(',', ', ')
        goal_state_new.append(s[0])
        grid[s[0]] = float(s[1])

    obstacle_states_new = []
    for state in obstacle_states:
        state = state.replace(',', ', ')
        obstacle_states_new.append(state)
        grid[state] = 'Obstacle'

    U = {s: 0 for s in grid.keys()}

    l = list(grid.keys())
    pi = {}
    for i in range(len(l)):
        s = l[i]
        actions = A(s)
        remainder = i % len(actions)
        pi.update({s: actions[remainder]})

    print(pi)
    
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

def SolveMDP(method_name, problem_file_name):
    # move(state) # if |action noise| > 1 then random else determinant
    # how to randomize???
    #
    # reward(state)
    readFile(problem_file_name)

    if method_name == 'ValueIteration':
        # call value iteration method
        return valueIteration()

    elif method_name == 'PolicyIteration':
        # call policy iteration method
        return policyIteration()

print(SolveMDP('ValueIteration', 'mdp1.txt'))