#%% VS Code Notebook
import functools
import numpy as np
import pandas as pd

from ortools.algorithms import pywrapknapsack_solver
# %% Cell based on docs: https://developers.google.com/optimization/bin/knapsack
solver = pywrapknapsack_solver.KnapsackSolver(
    pywrapknapsack_solver.KnapsackSolver.KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER, 
    'OnlineInfnitesimalKnapsackSolver',
)

values = [
    360.1, 83.5, 59, 130.7, 431, 67, 231, 52, 93, 125
]
weights = [[
    0.1, 0.2, 0.2, 0.1, 0.2, 0.1, 0.3, 0.1, 0.1, 0.2
]]
capacities = [1]

solver.Init(values, weights, capacities)
best_value = solver.Solve()
best_value
# %%
# transform [0,1) random to (0,epsilon] and return
def sample_infnitesimal_weight(epsilon,size=None):
    return ((np.random.random(size=size)*-1)+1)*epsilon

sample_infnitesimal_weight(0.1)
# %%
sample_value = np.random.uniform
sample_value(5,10,size=5)
#%%
@functools.lru_cache(maxsize=100)
def beta(p_min,p_max):
    return 1/(1+np.log(p_max/p_min))

def phi(y,p_min,p_max):
    beta_star = beta(p_min,p_max)
    if y<beta_star:
        return p_min
    else:
        return p_min*np.exp(y/beta_star-1)

def threshold_algorithm(p_min,p_max,weights,values):
    total_value = 0
    y = 0
    chosen = []
    for value, weight,i in zip(values,weights,range(len(weights))):
        pt = phi(y,p_min,p_max)
        density = value/weight
        if density>=pt and weight+y<=1: # accept
            total_value += value
            y += weight
            chosen.append(i)

    return total_value,chosen
# %%
# Parameters:
epsilon = 1
trials = 10000
items = 50

p_max = 101
p_min = 100

solver = pywrapknapsack_solver.KnapsackSolver(
    pywrapknapsack_solver.KnapsackSolver.KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER, 
    'OnlineInfnitesimalKnapsackSolver',
)

rows = []
# Solver doesn't do well with small numbers for values since it casts them to int64_t
scale_factor = 10_000_000
for i in range(trials):

    weights = sample_infnitesimal_weight(epsilon,size=items)
    values = sample_value(p_min,p_max,size=items)*weights
    density = values/weights
    capacities = np.array([1])

    solver.Init(
        (values*scale_factor).tolist(), 
        [(weights*scale_factor).tolist()], 
        (capacities*scale_factor).tolist(),
    )
    best_value = solver.Solve()/scale_factor
    best_chosen = [i for i in range(len(weights)) if solver.BestSolutionContains(i)]

    online_value,online_chosen = threshold_algorithm(p_min,p_max,weights,values)

    rows.append({
        'online_value':online_value,
        'best_value':best_value,
    })

df = pd.DataFrame(rows)
imperical_ratio = df['best_value'].mean()/df['online_value'].mean()
print('Empirical Ratio:',imperical_ratio)
print('Theoretical Ratio:',1 + np.log(p_max/p_min))
# %%

# %%
