import random


reward = -0.01
discount = 0.99
max_error = 10**(-3)

num_actions = 4
actions = [(1,0),(0,1),(-1,0),(0,1)] #Down, Left, Up, Right
num_rows = 3
num_cols = 4
utility = [[0, 0, 0, 1], [0, 0, 0, -1], [0, 0, 0, 0]]

random.seed(1)
policy = [[random.randint(0, 3) for j in range(num_cols)] for i in range(num_rows)] # Random initial policy

# Visualization
def printEnvironment(arr, policy=False):
    actionStr = ["Down", "Left", "Up", "Right"]
    res = ""
    for r in range(num_rows):
        res += "|"
        for c in range(num_cols):
            if r == c == 1:
                val = "WALL"
            elif r <= 1 and c == 3:
                val = "+1" if r == 0 else "-1"
            else:
                if policy:
                    val = actionStr[arr[r][c]]
                else:
                    val = str(arr[r][c])
            #Correct formatting of grid                        
            res += " " + val[:5].ljust(5) + " |" 
        res += "\n"
    print(res)

# Get the utility of the state reached by performing the given action from the given state
def getUtility(utility, r, c, action):
    rowChange, colChange = actions[action]
    newRow, newCol = r+rowChange, c+colChange
        #Action not possible
    if newRow < 0 or newCol < 0 or newRow >= num_rows or newCol >= num_cols or (newRow == newCol == 1): 
        return utility[r][c]
    else:
        return utility[newRow][newCol]

# Calculate the utility of a state given an action
#Utility = Reward(s) + Sum over s' { Probability(s'|s,a)U(s') }
def calculateUtility(utility, r, c, action):
    u = reward
    u += 0.1 * discount * getUtility(utility, r, c, (action-1)%4)
    u += 0.8 * discount * getUtility(utility, r, c, action)
    u += 0.1 * discount * getUtility(utility, r, c, (action+1)%4)
    return u

#Check convergence
#max over all s { V(t)-V(t-1) < e }
def valueIteration(utility):
    print("During the value iteration:\n")
    i=1        
    while True:
        nextUtility = [[0, 0, 0, 1], [0, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0]]
        error = 0
        for r in range(num_rows):
            for c in range(num_cols):
                if (r <= 1 and c == 3) or (r == c == 1):
                    continue
                nextUtility[r][c] = max([calculateUtility(utility, r, c, action) for action in range(num_actions)]) # Bellman update
                error = max(error, abs(nextUtility[r][c]-utility[r][c]))
        utility = nextUtility
        print("Iteration {}".format(i))        
        printEnvironment(utility)
        i+=1
        if error < max_error:
            break
    return utility

# Get the optimal policy from utility
def getOptimalPolicy(utility):
    policy = [[-1, -1, -1, -1] for i in range(num_rows)]
    for r in range(num_rows):
        for c in range(num_cols):
            if (r <= 1 and c == 3) or (r == c == 1):
                continue
            # Choose the action that maximizes the utility
            maxAction, maxUtility = None, -float("inf")
            for action in range(num_actions):
                u = calculateUtility(utility, r, c, action)
                if u > maxUtility:
                    maxAction, maxUtility = action, u
            policy[r][c] = maxAction
    return policy

# Print the initial environment
print("The initial utility is:\n")
printEnvironment(utility)

# Value iteration
# Get the optimal utility and policy and print them
utility = valueIteration(utility)

#Get optimum policy for the utility
policy = getOptimalPolicy(utility)

print("The optimal policy is:\n")
printEnvironment(policy, True)
