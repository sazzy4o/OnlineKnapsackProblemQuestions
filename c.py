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
# %% Cell based on https://stackoverflow.com/a/18444710/3558475
import scipy.stats as stats
def trunc_normal(mean, sd, low, upp, size=None):
    if size is not None:
        return np.array(stats.truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd).rvs(size))
    return stats.truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd).rvs(1)[0]
# %%
# transform [0,1) random to (0,epsilon] and return
def sample_infnitesimal_weight(epsilon,size=None):
    return ((np.random.random(size=size)*-1)+1)*epsilon
# def sample_infnitesimal_weight(epsilon,size=None):
#     return trunc_normal(epsilon,epsilon/2,0,epsilon,size=size)

sample_infnitesimal_weight(0.1)
# %%
# sample_value = np.random.uniform
sample_value = lambda p_min,p_max,size=None: trunc_normal(np.mean([p_min,p_max]),(p_max-p_min)/2,p_min,p_max,size=size)
sample_value(5,10,size=5)
#%%
@functools.lru_cache(maxsize=100)
def beta(p_min,p_max):
    return 1/(1+np.log(p_max/p_min))

def phi(y,p_min,p_max):
    beta_star = beta(p_min,p_max)
    if np.max(y)<beta_star:
        return p_min
    else:
        return p_min*np.exp(np.max(y)/beta_star-1)

agg = np.max
# agg = np.min
# agg = np.mean
# agg = np.sum
def threshold_algorithm(p_min,p_max,all_weights,values):
    total_value = 0
    y = np.zeros(len(all_weights))
    chosen = []
    for value, weights,i in zip(values,all_weights.T,range(all_weights.shape[1])):
        pt = phi(y,p_min,p_max)
        density = value/agg(weights)
        if density>=pt and np.max(weights+y)<=1: # accept
            total_value += value
            y += weights
            chosen.append(i)

    return total_value,chosen
# %%
# Parameters:
epsilon = 0.001
trials = 100
items = 10000
d = 4

p_max = 250
p_min = 10

solver = pywrapknapsack_solver.KnapsackSolver(
    pywrapknapsack_solver.KnapsackSolver.KNAPSACK_MULTIDIMENSION_CBC_MIP_SOLVER, 
    'OnlineInfnitesimalKnapsackSolver',
)

rows = []
# Solver doesn't do well with small numbers for values since it casts them to int64_t
scale_factor = 10_000_000
from tqdm import tqdm
for i in tqdm(range(trials)):

    weights = np.array([sample_infnitesimal_weight(epsilon,size=items) for _ in range(d)])
    values = sample_value(p_min,p_max,size=items)*np.max(weights,axis=0)
    density = values/weights
    capacities = np.ones(d)

    solver.Init(
        (values*scale_factor).tolist(), 
        (weights*scale_factor).tolist(), 
        (capacities*scale_factor).tolist(),
    )
    best_value = solver.Solve()/scale_factor

    online_value,res = threshold_algorithm(p_min,p_max,weights,values)

    rows.append({
        'online_value':online_value,
        'best_value':best_value,
    })

df = pd.DataFrame(rows)
imperical_ratio = df['best_value'].mean()/df['online_value'].mean()
print('Empirical Ratio:',imperical_ratio)
print('Theoretical Ratio:',1 + np.log(p_max/p_min))
# %%
