# Eren Demircan - 2237246
# CEng462 - AI HW5
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


def SolveMDP(method_name, problem_file_name):
    return 

readFile("mdp1.txt")
print(M, N, reward, action_noise, gamma, epsilon, iteration, obstacle_states, goal_state)