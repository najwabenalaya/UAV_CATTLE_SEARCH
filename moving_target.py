#!/usr/bin/env python
# coding: utf-8

# %%
import math
import random
from itertools import combinations
import gurobipy as grp
from gurobipy import GRB
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import argparse
import sys
import random
import json
import time
import psutil
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# %%

M = 100
DEFAULT_SIZE_X = 3
DEFAULT_SIZE_Y = 3
DEFAULT_SEED = 1
DEFAULT_max_TIME = 5
DEFAULT_SPEED = 10
DEFAULT_MATRIX = np.array([[0, 1, 0], [0.5, 0, 0.5], [0, 1, 0]])
DEFAULT_INITIAL_DRONE_X = None
DEFAULT_INITIAL_DRONE_Y = None
DEFAULT_INITIAL_DIST = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0.001, 0.999]])
DEFAULT_MOBILITY = None
DEFAULT_MIN_PD = 0.99


transition_matrix = np.array([[0, 1, 0], [1, 0, 0], [0, 1, 0]])
initial_dist = np.array([0.5, 0.4, 0.1])

if False:
    p = initial_dist
    for i in range(0, 10):
        print(p)
        p = p @ transition_matrix.T

prefix = "ucssmt"


def get_args():
    if "pydevconsole" in sys.argv[0]:
        arg_list = sys.argv[3:]
    else:
        arg_list = sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("--x-size", type=int, default=DEFAULT_SIZE_X)
    parser.add_argument("--y-size", type=int, default=DEFAULT_SIZE_Y)
    parser.add_argument("--initial-drone-x", type=int, default=DEFAULT_INITIAL_DRONE_X)
    parser.add_argument("--initial-drone-y", type=int, default=DEFAULT_INITIAL_DRONE_Y)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--mobility", type=float, default=DEFAULT_MOBILITY)
    parser.add_argument("--initial-dist", default=DEFAULT_INITIAL_DIST)
    parser.add_argument("--speed", type=float, default=DEFAULT_SPEED)
    parser.add_argument("--no-show", action="store_true", default=False)
    parser.add_argument("--max-time", type=int, default=DEFAULT_max_TIME)
    parser.add_argument("--save-problem", action="store_true", default=False)
    parser.add_argument("--uniform-poc", action="store_true", default=False)
    parser.add_argument("--nb-poc", type=int, default=None)
    parser.add_argument("--extra-constraint", action="store_true", default=False)
    parser.add_argument("--min-pd", type=float, default=DEFAULT_MIN_PD)
    parser.add_argument("--objective-pd", action="store_true", default=False)
    args = parser.parse_args(arg_list)
    return args


args = get_args()
SIZE_X = args.x_size
SIZE_Y = args.y_size
SEED = args.seed
NO_SHOW = args.no_show
MAX_TIME = args.max_time
SAVE_PROB = args.save_problem
SPEED = args.speed
INITIAL_DIST = args.initial_dist
UNIFORM_POC = args.uniform_poc
NB_POC = args.nb_poc
EXTRA_CONSTRAINT = args.extra_constraint
DBG = True
INITIAL_DRONE_X = args.initial_drone_x
INITIAL_DRONE_Y = args.initial_drone_y
MOBILITY = args.mobility
MIN_PD = args.min_pd
OBJECTIVE_PD = args.objective_pd


def get_dir_name():
    global SIZE_X, SIZE_Y, SEED
    dir_name = prefix + f"-{SIZE_X}x{SIZE_Y}"
    if MAX_TIME != DEFAULT_max_TIME or True:
        if int(MAX_TIME) == MAX_TIME:
            dir_name += f"xT{int(MAX_TIME)}"
        else:
            dir_name += f"xT{MAX_TIME}"
    if SPEED != DEFAULT_SPEED:
        dir_name += f"-sp{int(SPEED)}" if (int(SPEED) == SPEED) else f"-sp{SPEED}"
    if UNIFORM_POC:
        dir_name += "-uniform"
    if NB_POC is not None:
        dir_name += f"-poc{NB_POC}"
    if SEED != DEFAULT_SEED:
        dir_name += f"-s{SEED}"
    if MOBILITY is not None:
        dir_name += f"-m{MOBILITY}"
    if MIN_PD is not None and MIN_PD != DEFAULT_MIN_PD:
        dir_name += f"-pd{MIN_PD}"
    return dir_name


# %%

def pos_to_index(x, y):
    return x + y * SIZE_X + 1


def index_to_pos(idx):
    assert 1 <= idx and idx < pos_to_index(SIZE_X, SIZE_Y)
    return (idx - 1) % SIZE_X, (idx - 1) // SIZE_X


def get_index_neighbors(idx):
    x, y = index_to_pos(idx)
    result = []
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)]:
        xx = x + dx
        yy = y + dy
        if (0 <= xx and xx < SIZE_X and 0 <= yy and yy < SIZE_Y):
            result.append(pos_to_index(xx, yy))
    return result


set_X = range(1, SIZE_X + 1)
set_Y = range(1, SIZE_Y + 1)
set_J = list(range(1, SIZE_X * SIZE_Y + 1))
print(set_J)

time_start = time.time()
process = psutil.Process()
_ignored = process.cpu_percent()

np.random.seed(SEED)
random.seed(SEED)

dir_name = get_dir_name()
if not os.path.exists(dir_name):
    os.mkdir(dir_name)

if SIZE_X == 5 and SIZE_Y == 5:
    initial_dist = np.array([[0, 0.5, 0, 0, 0], [0, 0, 0.5, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
    initial_dist = initial_dist.flatten()
else:
    if UNIFORM_POC:
        initial_dist = np.ones(shape=(SIZE_Y * SIZE_X)) / (SIZE_Y * SIZE_X)
    else:
        initial_dist = np.random.random(size=(SIZE_Y * SIZE_X))

    if NB_POC is not None:
        pos_list = [(x, y) for x in range(SIZE_X) for y in range(SIZE_Y)]
        unselected_idx = np.random.choice(range(len(pos_list)), size=SIZE_X * SIZE_Y - NB_POC, replace=False)  # ?
        print(pos_list)
        print(unselected_idx)
        for i in unselected_idx:
            initial_dist[pos_to_index(*pos_list[i]) - 1] = 0
    initial_dist = initial_dist / np.sum(initial_dist)


if MOBILITY is not None:
    assert 0 <= MOBILITY and MOBILITY <= 1
    transition_matrix = np.zeros(shape=(len(set_J), len(set_J)))
    for j in set_J:
        neigh_list_j = get_index_neighbors(j)
        assert j in neigh_list_j
        neigh_list_j.remove(j)
        transition_matrix[j - 1, j - 1] = 1 - MOBILITY
        for neigh_j in neigh_list_j:
            transition_matrix[neigh_j - 1, j - 1] = MOBILITY / len(neigh_list_j)

print(transition_matrix.sum(axis=0))
print(transition_matrix.sum(axis=1))
transition_matrix = transition_matrix.T

if True:
    p = initial_dist
    print("----")
    for i in range(0, 10):
        print(p)
        p = p @ transition_matrix.T

# %%

m = grp.Model(name="USC Model")

if DBG:
    for i in range(1, 4 + 1):
        print(i, get_index_neighbors(i))

if DBG:
    i = 12
    # print("x"+str(i)+"+1")
    # print("x%d+1" % i)
    # print("x{0}+1 but not x{1}".format(i,i+3))
    print(f"x{i}+1")

# Adding the Variables


time_horizon = [x for x in range(1, MAX_TIME + 1)]  # t
v = {}
q = {}
w = {}
Pd = {}
print(time_horizon, set_J)

for j in set_J:
    for t in time_horizon:
        v[j, t] = m.addVar(vtype=GRB.BINARY, name=f"v({j},{t})")
        q[j, t] = m.addVar(vtype=GRB.CONTINUOUS, name=f"q({j},{t})")
        w[j, t] = m.addVar(vtype=GRB.CONTINUOUS, name=f"w({j},{t})")
        m.update();

# creating the model and setting the objective

if OBJECTIVE_PD:
    objective = sum(w[j, t] for t in time_horizon for j in set_J)
    m.setObjective(objective, GRB.MAXIMIZE)
else:
    objective = sum(w[j, t] * t for t in time_horizon for j in set_J)
    m.setObjective(objective, GRB.MINIMIZE)

print(objective)

# Adding the constraint

### the big M method for linearisation
for j in set_J:
    for t in time_horizon:
        m.addConstr(w[j, t] <= q[j, t])
        m.addConstr(w[j, t] <= v[j, t] * M)
        m.addConstr(w[j, t] >= 0)
        m.addConstr(w[j, t] >= q[j, t] - M * (1 - v[j, t]))

### Flow Conservation constraint
for j in set_J:  #
    for t in time_horizon[:-1]:
        v_neigh = [v[k, t + 1] for k in get_index_neighbors(j)]
        m.addConstr(v[j, t] <= sum(v_neigh))

for t in time_horizon:
    cell = [v[j, t] for j in set_J]
    m.addConstr(sum(cell) == 1)

print(transition_matrix)

if INITIAL_DRONE_X is not None:
    assert INITIAL_DRONE_Y is not None
    pos = pos_to_index(INITIAL_DRONE_X, INITIAL_DRONE_Y)
    raise RuntimeError(" Not implemented: set initial position of the drone in v(.,.)",
                       (pos, INITIAL_DRONE_X, INITIAL_DRONE_Y))

### the undetected Probability q(j,t) update in t+1 constraint
for i in set_J:
    for t in time_horizon:
        if t == MAX_TIME:
            continue
        z = [transition_matrix[j - 1, i - 1] * (q[j, t] - w[j, t]) for j in set_J]
        m.addConstr(q[i, t + 1] == sum(z))

### the intial distribution constraint
for j in set_J:
    m.addConstr(q[j, 1] == initial_dist[j - 1]);  # Math-index for q, Python-index for initial_dis

# Penalty constraint

Pd = sum(w.values())
m.addConstr(Pd >= MIN_PD)

# Running the Optimization
m.update()
print(m.display())
m.optimize()
print(m.getJSONSolution())

# Printing out the solution

for t in time_horizon:
    for j in set_J:
        print(v[j, t], end=" ")
    print()

print()
for t in time_horizon:
    for j in set_J:
        print(q[j, t], end=" ")
    print()

print()
for t in time_horizon:
    for j in set_J:
        print(w[j, t], end=" ")
    print()

print("markov evolution")
density_dist = initial_dist
for t in time_horizon:
    print(density_dist)
    density_dist = density_dist @ transition_matrix

if SAVE_PROB:
    m.write(dir_name + "/problem.mps")
    m.write(dir_name + "/problem.lp")


# %%


def swap(xy):
    return xy[1], xy[0]


time_end = time.time()

if False:
    solution_edges = []
    for edge, selected in v.items():
        # x[(1,2)].X
        if selected.v >= 1:
            print(edge)
            solution_edges.append(edge)

epsilon = 1e-3

drone_pos_list = []
drone_index_pos_list = []
drone_prob_detect_list = []
residual_undetected_prob_list = []
for t in range(1, MAX_TIME + 1):
    drone_pos = None
    detect_prob = None
    residual_undetected_probs = []
    for j in set_J:
        value = v[j, t].X
        if abs(value) < epsilon:
            v_val = 0
        elif abs(1 - value) < epsilon:
            v_val = 1
        else:
            raise ValueError("INvalid boolean value", value)
        if v_val == 1:
            assert drone_pos is None
            drone_pos = j
            detect_prob = w[j, t].X
        residual_undetected_probs.append(q[j, t].X)
    assert drone_pos is not None
    drone_pos_list.append(index_to_pos(drone_pos))
    drone_index_pos_list.append(drone_pos)
    drone_prob_detect_list.append(detect_prob)
    residual_undetected_prob_list.append(residual_undetected_probs)

print(drone_pos_list)
print(drone_prob_detect_list)
print(residual_undetected_prob_list)

# Save the solution in file(s)
m.write(dir_name + "/solution.sol")
with open(dir_name + "/solution.json", "w") as f:
    f.write(m.getJSONSolution())

simple_solution = {
    "size_x": SIZE_X,
    "size_y": SIZE_Y,
    "max_time": MAX_TIME,
    "drone_pos_list": drone_pos_list,
    "drone_prob_detect_list": drone_prob_detect_list,
    "residual_undetected_prob_list": residual_undetected_prob_list,
    "objective": m.ObjVal
}

sol_file_name = dir_name + "/solution-short.json"
with open(sol_file_name, "w") as f:
    json.dump(simple_solution, f)

print("solution saved in ", sol_file_name)

# Time information about solving
solving_info = {
    # "time_start": time_start, "time_end": time_end,
    "search-time": time_end - time_start,
    "cpu_times": process.cpu_times(), "cpu_percent": process.cpu_percent(), "times": os.times()
}
with open(dir_name + "/solving-info.json", "w") as f:
    json.dump(solving_info, f)

import plot_trajectory

prob_map = np.zeros((SIZE_Y, SIZE_X))
print(initial_dist)

for x in range(SIZE_X):
    for y in range(SIZE_Y):
        prob_map[y, x] = initial_dist[pos_to_index(x, y) - 1]
plot_trajectory.plot_trajectory(dir_name, drone_index_pos_list, SIZE_X, SIZE_Y, prob_map=prob_map)
plt.show()

if m.status == grp.GRB.OPTIMAL:
    print("Optimal solution found.")
else:
    print("No optimal solution found.")

start_time = time.time()
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")







