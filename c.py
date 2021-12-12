#%% VS Code Notebook
import functools
import numpy as np
import pandas as pd

from ortools.algorithms import pywrapknapsack_solver
# %%
solver = pywrapknapsack_solver.KnapsackSolver(
    pywrapknapsack_solver.KnapsackSolver.KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER, 
    'OnlineInfnitesimalKnapsackSolver',
)

values = [
    360.1, 83.5, 59, 130.7, 431, 67, 231, 52, 93, 125
]
weights = [
    [0.1, 0.2, 0.2, 0.1, 0.2, 0.1, 0.3, 0.1, 0.1, 0.2],
    [0.2, 0.1, 0.3, 0.2, 0.3, 0.2, 0.05,0.1, 0.1, 0.1],
]
capacities = [1,1]

solver.Init(values, weights, capacities)
best_value = solver.Solve()
print(best_value)
# %%
# transform [0,1) random to (0,epsilon] and return
def sample_infnitesimal_weight(epsilon,size=None):
    return ((np.random.random(size=size)*-1)+1)*epsilon

sample_infnitesimal_weight(0.1)
# %%
# transform [0,1) random to (0,epsilon] and return
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

agg = np.max
# agg = np.min
# agg = np.mean
# agg = np.sum
def threshold_algorithm(p_min,p_max,all_weights,values):
    total_value = 0
    y = np.zeros(len(all_weights))
    chosen = []
    for value, weights,i in zip(values,all_weights,range(len(all_weights))):
        pt = phi(y,p_min,p_max)
        density = value/agg(weights)
        if density>=pt and weights+y<=1: # accept
            total_value += value
            y += weights
            chosen.append(i)

    return total_value,chosen
# %%
# Parameters:
epsilon = 0.001
trials = 100
items = 25000
d = 2

p_max = 10
p_min = 1

solver = pywrapknapsack_solver.KnapsackSolver(
    pywrapknapsack_solver.KnapsackSolver.KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER, 
    'OnlineInfnitesimalKnapsackSolver',
)

rows = []
# Solver doesn't do well with small numbers for values since it casts them to int64_t
scale_factor = 10_000_000
for i in range(trials):

    weights = np.array([sample_infnitesimal_weight(epsilon,size=items) for _ in range(d)])
    values = sample_value(p_min,p_max,size=items)*weights
    density = values/weights
    capacities = np.ones(d)

    solver.Init(
        (values*scale_factor).tolist(), 
        (weights*scale_factor).tolist(), 
        (capacities*scale_factor).tolist(),
    )
    best_value = solver.Solve()/scale_factor

    online_value,_ = threshold_algorithm(p_min,p_max,weights,values)

    rows.append({
        'online_value':online_value,
        'best_value':best_value,
    })

df = pd.DataFrame(rows)
imperical_ratio = df['best_value'].mean()/df['online_value'].mean()
print('Imperical Ratio:',imperical_ratio)
print('Theoretical Ratio:',1 + np.log(p_max/p_min))
# %%
