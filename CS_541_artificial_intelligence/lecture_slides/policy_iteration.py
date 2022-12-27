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
                val = actionStr[arr[r][c]]
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
def policyEvaluation(policy, utility):
    while True:
        nextUtility = [[0, 0, 0, 1], [0, 0, 0, -1], [0, 0, 0, 0]]
        error = 0
        for r in range(num_rows):
            for c in range(num_cols):
                if (r <= 1 and c == 3) or (r == c == 1):
                    continue
                nextUtility[r][c] = calculateUtility(utility, r, c, policy[r][c]) 
                error = max(error, abs(nextUtility[r][c]-utility[r][c]))
        utility = nextUtility
        if error < max_error:
            break
    return utility


def policyIteration(policy, utility):
    print("During the policy iteration:\n")
    i=1
    while True:
        utility = policyEvaluation(policy, utility)
        unchanged = True
        for r in range(num_rows):
            for c in range(num_cols):
                if (r <= 1 and c == 3) or (r == c == 1):
                    continue
                maxAction, maxUtility = None, -float("inf")
                for action in range(num_actions):
                    u = calculateUtility(utility, r, c, action)
                    if u > maxUtility:
                        maxAction, maxUtility = action, u
				#Action which maximizes the utility
                if maxUtility > calculateUtility(utility, r, c, policy[r][c]):
                    policy[r][c] = maxAction 
                    unchanged = False
        if unchanged:
            break
        print("Iteration {}".format(i))			
        printEnvironment(policy, True)
        i+=1
    return policy

# Print the initial environment
print("The initial random policy is:\n")
printEnvironment(policy, True)

# Policy iteration
policy = policyIteration(policy, utility)

# Print the optimal policy
print("The optimal policy is:\n")
printEnvironment(policy, True)
