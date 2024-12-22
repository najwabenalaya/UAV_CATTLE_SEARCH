#!/usr/bin/env python
# coding: utf-8

#%%
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
import plot_trajectory
import time
import psutil
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import warnings
import pprint


#%%

M = 10
DEFAULT_SIZE_X = 1
DEFAULT_SIZE_Y = 3
DEFAULT_SEED = 1
DEFAULT_max_TIME = 6
DEFAULT_SPEED = 10
DEFAULT_INITIAL_DRONE_X = None
DEFAULT_INITIAL_DRONE_Y = None
DEFAULT_DIST_C1 = np.array([0.2, 0.3, 0.5])
DEFAULT_DIST_C2 = np.array([0.2, 0.3, 0.5])
DEFAULT_MOBILITY = None
DEFAULT_MOBILITY_C1= None
DEFAULT_MOBILITY_C2= None
DEFAULT_MIN_PD = 0.8




prefix = "ucstmt"

def get_args():
    if "pydevconsole" in sys.argv[0]: # test if the script running within pydevconsole
        arg_list = sys.argv[3:]
    else:
        arg_list = sys.argv [1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("--x-size", type=int, default=DEFAULT_SIZE_X)
    parser.add_argument("--y-size", type=int, default=DEFAULT_SIZE_Y)
    parser.add_argument("--initial-drone-x", type=int, default=DEFAULT_INITIAL_DRONE_X)
    parser.add_argument("--initial-drone-y", type=int, default=DEFAULT_INITIAL_DRONE_Y)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--mobility", type=float, default=DEFAULT_MOBILITY)
    parser.add_argument("--mobility-c1", type=float, default=DEFAULT_MOBILITY_C1)
    parser.add_argument("--mobility-c2", type=float, default=DEFAULT_MOBILITY_C2)
    parser.add_argument("--speed", type=float, default=DEFAULT_SPEED)
    parser.add_argument("--no-show", action="store_true", default=False)
    parser.add_argument("--max-time", type=int, default=DEFAULT_max_TIME)
    parser.add_argument("--save-problem", action="store_true", default=False)
    parser.add_argument("--uniform-poc", action="store_true", default=False)
    parser.add_argument("--nb-poc-c1", type=int, default=None)
    parser.add_argument("--nb-poc-c2", type=int, default=None)
    parser.add_argument("--extra-constraint", action="store_true", default=False)
    parser.add_argument("--min-pd", type=float, default=DEFAULT_MIN_PD)
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
UNIFORM_POC = args.uniform_poc
#NB_POC = args.nb_poc
EXTRA_CONSTRAINT = args.extra_constraint
DBG = True
INITIAL_DRONE_X = args.initial_drone_x
INITIAL_DRONE_Y = args.initial_drone_y
MOBILITY = args.mobility
MOBILITY_C1 = args.mobility_c1
MOBILITY_C2 = args.mobility_c2
MIN_PD = args.min_pd
NB_POC_C1 = args.nb_poc_c1
NB_POC_C2 = args.nb_poc_c2

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
    # if NB_POC is not None:
    #     dir_name += f"-poc{NB_POC}
    if SEED != DEFAULT_SEED:
        dir_name += f"-s{SEED}"
    if MOBILITY_C1 is not None:
        dir_name += f"-m1{MOBILITY_C1}"
    if MOBILITY_C2 is not None:
        dir_name += f"-m2{MOBILITY_C2}"
    if MIN_PD is not None and MIN_PD != DEFAULT_MIN_PD:
        dir_name += f"-pd{MIN_PD}"
    return dir_name
    if NB_POC_C1 is not None:
        dir_name += f"-poc-c1{NB_POC_C1}"
    return dir_name
    if NB_POC_C2 is not None:
        dir_name += f"-poc-c2{NB_POC_C2}"
    return dir_name

#%%

def joint_pos(j1, j2):
    return (j2-1)*len(set_J)+(j1-1)+1


def pos_to_index(x, y):
    return x + y * SIZE_X + 1


def index_to_pos(idx):
    assert 1 <= idx and idx < pos_to_index(SIZE_X, SIZE_Y)
    return (idx - 1) % SIZE_X, (idx - 1) // SIZE_X


def flatten_matrix(mat):
    y_size,x_size  = mat.shape
    assert x_size == SIZE_X and y_size == SIZE_Y
    res = np.zeros(shape = x_size*y_size)
    for x in range(x_size):
        for y in range(y_size):
            idx_plus_1 = pos_to_index(x,y)
            res[idx_plus_1-1] = mat[y,x]
    return res


def get_index_neighbors(idx):
    x, y = index_to_pos(idx)
    result = []
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)]:
        xx = x + dx
        yy = y + dy
        if (0 <= xx and xx < SIZE_X and 0 <= yy and yy < SIZE_Y):
            result.append(pos_to_index(xx, yy))
    return result

set_X = range(1, SIZE_X+1)
set_Y = range(1, SIZE_Y+1)
set_J = list(range(1, SIZE_X*SIZE_Y+1))
print(set_J)


time_start = time.time()
process = psutil.Process()
_ignored = process.cpu_percent()

np.random.seed(SEED)
random.seed(SEED)


dir_name = get_dir_name()
if not os.path.exists(dir_name):
    os.mkdir(dir_name)

if SIZE_X == 1 and SIZE_Y == 4:
    dist_C1 = np.array([0, 0, 0, 1])
    dist_C2 = np.array([1, 0, 0, 0])
elif SIZE_X == 2 and SIZE_Y == 3:
    dist_C1 = flatten_matrix(np.array([[0,0],[1,0],[0,0]]))
    dist_C2 = flatten_matrix(np.array([[0,0],[0,1],[0,0]]))
else:
    if UNIFORM_POC:
        dist_C1 =  np.ones(shape=(SIZE_Y*SIZE_X))/(SIZE_Y*SIZE_X)
        dist_C2 = np.ones(shape=(SIZE_Y*SIZE_X))/(SIZE_Y*SIZE_X)
    else:
        dist_C1 = np.random.random(size=(SIZE_Y*SIZE_X))
        dist_C2 = np.random.random(size=(SIZE_Y*SIZE_X))
        dist_C1 = dist_C1 / np.sum(dist_C1)
        dist_C2 = dist_C2 / np.sum(dist_C2)

    if NB_POC_C1 is not None:
        pos_list = [(x, y) for x in range(SIZE_X) for y in range(SIZE_Y)]
        unselected_idx = np.random.choice(range(len(pos_list)), size=SIZE_X * SIZE_Y - NB_POC_C1, replace=False)  # ?
        print(pos_list)
        print(unselected_idx)
        for i in unselected_idx:
            dist_C1[pos_to_index(*pos_list[i]) - 1] = 0
    dist_C1 = dist_C1 / np.sum(dist_C1)

    if NB_POC_C2 is not None:
        pos_list = [(x,y) for x in range(SIZE_X) for y in range(SIZE_Y)]
        unselected_idx = np.random.choice(range(len(pos_list)), size=SIZE_X*SIZE_Y-NB_POC_C2, replace=False) #?
        print(pos_list)
        print(unselected_idx)
        for i in unselected_idx:
            dist_C2[pos_to_index(*pos_list[i])-1] = 0
    dist_C2 = dist_C2 / np.sum(dist_C2)


list_joint_pos = []
for j1 in set_J:
    for j2 in set_J:
        list_joint_pos.append(joint_pos(j1,j2))
num_joint_pos = len(list_joint_pos)


if MOBILITY_C1 is not None:
    assert 0 <= MOBILITY_C1 <= 1
    transition_matrix_c1 = np.zeros((len(set_J), len(set_J)))

    for i, j in enumerate(set_J):
        transition_matrix_c1[i-1, i-1] = 1 - MOBILITY_C1
        for k, neigh_j in enumerate(set_J):
            if k != i:
                transition_matrix_c1[k, i] = MOBILITY_C1 / ( len(set_J) -1)
else:
    transition_matrix_c1 = np.eye(len(set_J), len(set_J))


if MOBILITY_C2 is not None:
    assert 0 <= MOBILITY_C2 <= 1
    transition_matrix_c2 = np.zeros((len(set_J), len(set_J)))

    for i, j in enumerate(set_J):
        transition_matrix_c2[i-1, i-1] = 1 - MOBILITY_C2
        for k, neigh_j in enumerate(set_J):
            if k != i:
                transition_matrix_c2[k, i] = MOBILITY_C2 / ( len(set_J) -1)
else:
    transition_matrix_c2 = np.eye(len(set_J), len(set_J))


transition_matrix= np.kron(transition_matrix_c1, transition_matrix_c2)
joint_state = np.zeros((len(set_J), len(set_J)))


if SIZE_X is None and SIZE_Y is None:
    for j in set_J:
        for i in set_J:
            joint_state[j-1,i-1 ] = (DEFAULT_DIST_C1[j-1]* DEFAULT_DIST_C2[i-1])
elif SIZE_X is None and SIZE_Y is not None:
    warnings.warn("XXX: code looks incorrect, but maybe works because DEFAULT_X_SIZE==1")
    for j in set_J:
        for i in set_J:
            joint_state[j-1, i-1] = (DEFAULT_DIST_C1[j-1] * dist_C2[i-1])
elif SIZE_X is not None and SIZE_Y is None:
    warnings.warn("XXX: code looks incorrect")
    for j in set_J:
        for i in set_J:
            joint_state[j-1,i-1] = (dist_C1[j-1] * DEFAULT_DIST_C2[i-1])
else:
    prior_mobility_time = 0
    tc1 = np.linalg.matrix_power(transition_matrix_c1, prior_mobility_time)
    tc2 = np.linalg.matrix_power(transition_matrix_c2, prior_mobility_time)
    dist_C1 = tc1 @ dist_C1
    dist_C2 = tc2 @ dist_C2
    for j in set_J:
        for i in set_J:
            joint_state[j-1, i-1] = (dist_C1[j-1] * dist_C2[i-1])


#%%

m = grp.Model(name="ucstmt Model")


time_horizon = [x for x in range(1, MAX_TIME+1)] #t
v = {}
q = {}
Q = {}
w = {}
W1 = {}
W2 = {}
W3 = {}
Pd = {}


for j in set_J:
    for t in time_horizon:
        v[j, t] = m.addVar(vtype=GRB.BINARY, name=f"v({j},{t})")
        q[j, t] = m.addVar(vtype=GRB.CONTINUOUS, name=f"q({j},{t})")
        w[j, t] = m.addVar(vtype=GRB.CONTINUOUS, name=f"w({j},{t})")
        m.update();

for j in set_J:
    for i in set_J:
        for t in time_horizon:
            Q[j, i, t] = m.addVar(vtype=GRB.CONTINUOUS, name=f"Q({j},{i},{t})")
            W1[j, i, t] = m.addVar(vtype=GRB.CONTINUOUS, name=f"W1({j},{i},{t})")
            W2[j, i, t] = m.addVar(vtype=GRB.CONTINUOUS, name=f"W2({j},{i},{t})")
            W3[j, i, t] = m.addVar(vtype=GRB.CONTINUOUS, name=f"W3({j},{i},{t})")
            m.update();


################################" set Q(j,i,1) to the joint state (dist_c1* dist_c2)

for j in set_J:
    for i in set_J:
        m.addConstr(Q[j, i, 1] == joint_state[j-1][i-1])




for j1 in set_J:
    for j2 in set_J:
        for t in time_horizon[:-1]:
            F = sum([transition_matrix[joint_pos(j1p,j2p)-1][joint_pos(j1,j2)-1] *  ( Q[j1p, j2p,t]- W1[j1p, j2p,t] - W2[j1p, j2p,t])
                     for j1p in set_J for j2p in set_J if j1p != j2p])
            E = sum([transition_matrix[joint_pos(j1p,j1p)-1][joint_pos(j1,j2)-1] * ( Q[j1p, j1p,t] - W3[j1p, j1p,t] )
                     for j1p in set_J ])
            m.addConstr(Q[j1, j2, t+1] == F+E)




for j in set_J:
    for i in set_J:
        for t in time_horizon:
            m.addConstr(Q[j,i ,t] >= 0)


for j in set_J:
    for t in time_horizon:
        m.addConstr(q[j, t] >= 0)


for j in set_J:
    for t in time_horizon:
        z = [Q[j, i, t] for i in set_J]
        y = [Q[i, j, t] for i in set_J]
        m.addConstr(q[j, t] ==  sum(z) + sum(y) - Q[j, j, t])


####################### the big M method for linearisation

for j in set_J:
    for t in time_horizon:
        m.addConstr(w[j, t] <= q[j, t])
        m.addConstr(w[j, t] <= v[j, t]*M)
        m.addConstr(w[j, t] >= 0)
        m.addConstr(w[j, t] >= q[j, t] - M*(1 - v[j, t]))

for j in set_J:
    for i in set_J:
        for t in time_horizon:
            if i != j:
                m.addConstr(W1[j, i, t] <= Q[j, i, t])
                m.addConstr(W1[j, i, t] <= v[j, t] * M)
                m.addConstr(W1[j, i, t] >= 0)
                m.addConstr(W1[j, i, t] >= Q[j, i, t] - M*(1 - v[j, t]))


for j in set_J:
    for i in set_J:
        for t in time_horizon:
            if i != j:
                m.addConstr(W2[j, i, t] <= Q[j, i, t])
                m.addConstr(W2[j, i, t] <= v[i, t]*M)
                m.addConstr(W2[j, i, t] >= 0)
                m.addConstr(W2[j, i, t] >= Q[j, i, t] - M*(1 - v[i, t]))


for j in set_J:
    for t in time_horizon:
        m.addConstr(W3[j, j, t] <= Q[j, j, t])
        m.addConstr(W3[j, j, t] <= v[j, t]*M)
        m.addConstr(W3[j, j, t] >= 0)
        m.addConstr(W3[j, j, t] >= Q[j, j, t] - M*(1 - v[j, t]))

############################### creating the model and setting the objective

objective = sum(w[j,t] * t for t in time_horizon for j in set_J )
detection = sum(w.values())
m.setObjective(objective, GRB.MINIMIZE)

# m.setObjective(detection, GRB.MAXIMIZE)
# m.addConstr(detection >= 0.85)

############################################# constrained path
for j in set_J:
    for t in time_horizon[:-1]:
        v_neigh = [v[k, t+1] for k in get_index_neighbors(j)]
        m.addConstr(v[j, t] <= sum (v_neigh))

############################# one searcher each time period
for t in time_horizon:
    cell= [v[j,t] for j in set_J]
    m.addConstr(sum(cell) == 1)



if INITIAL_DRONE_X is not None:
    assert INITIAL_DRONE_Y is not None
    pos = pos_to_index(INITIAL_DRONE_X, INITIAL_DRONE_Y)
    raise RuntimeError(" Not implemented: set initial position of the drone in v(.,.)", (pos, INITIAL_DRONE_X, INITIAL_DRONE_Y) )

############################### Penalty constraint
Pd = sum(w.values())
m.addConstr(Pd >= MIN_PD)



# Running the Optimization
m.update()
print(m.display())
m.write("tmp-model.lp")
m.optimize()
print(m.getJSONSolution())

# Printing out the solution
print(joint_state)


for t in time_horizon:
    for j in set_J:
        print(v[j,t], end=" ")
    print()

print()
for t in time_horizon:
    for j in set_J:
        print(q[j,t], end=" ")
    print()

true_dist = [q[j, 1].X  for j in set_J ]

print()
for t in time_horizon:
    for j in set_J:
        print(w[j, t], end=" ")
    print()


for t in time_horizon:
    for j in set_J:
        for i in set_J:
            print(Q[j, i, t], end=" ")
        print()

for t in time_horizon:
    for j in set_J:
        for i in set_J:
            print(W1[j, i, t], end=" ")
        print()
    print()

for t in time_horizon:
    for i in set_J:
        for j in set_J:
            print(W2[i, j, t], end=" ")
        print()
    print()

for t in time_horizon:
    for j in set_J:
        print(W3[j, j, t], end=" ")
    print()

print(f" the sum of probability of detection is: {Pd}")




print("markov evolution")
density_dist = joint_state

if SAVE_PROB:
    m.write(dir_name+"/problem.mps")
    m.write(dir_name+"/problem.lp")

#%%


def swap(xy):
    return xy[1],xy[0]

time_end = time.time()

if False:
    solution_edges = []
    for edge,selected in v.items():
        #x[(1,2)].X
        if selected.v >= 1:
            print(edge)
            solution_edges.append(edge)

epsilon = 1e-3

drone_pos_list =  []
drone_index_pos_list =  []
drone_prob_detect_list = []
residual_undetected_prob_list = []
for t in range(1, MAX_TIME+1):
    drone_pos = None
    detect_prob = None
    residual_undetected_probs = []
    for j in set_J:
        value = v[j,t].X
        if abs(value) < epsilon:
            v_val = 0
        elif abs(1-value) < epsilon:
            v_val = 1
        else:
            raise ValueError("INvalid boolean value", value)
        if v_val == 1:
            assert drone_pos is None
            drone_pos = j
            detect_prob = w[j,t].X
        residual_undetected_probs.append(q[j,t].X)
    assert drone_pos is not None
    drone_pos_list.append(index_to_pos(drone_pos))
    drone_index_pos_list.append(drone_pos)
    drone_prob_detect_list.append(detect_prob)
    residual_undetected_prob_list.append(residual_undetected_probs)

print(drone_pos_list)
print(drone_prob_detect_list)
print(residual_undetected_prob_list)


# Save the solution in file(s)
m.write(dir_name+"/solution.sol")
# # https://www.gurobi.com/documentation/9.5/refman/json_solution_format.html
with open(dir_name+"/solution.json", "w") as f:
    f.write(m.getJSONSolution())
    #json.dump(m.getJSONSolution(), f)

simple_solution = {
    "SIZE_X": SIZE_X,
    "size_y": SIZE_Y,
    "max_time": MAX_TIME,
    "drone_pos_list": drone_pos_list,
    "drone_prob_detect_list": drone_prob_detect_list,
    "residual_undetected_prob_list": residual_undetected_prob_list,
    #"t": { k:v.X  for k,v in t.items()},
    "objective": m.ObjVal
}

sol_file_name = dir_name+"/solution-short.json"
with open(sol_file_name, "w") as f:
    json.dump(simple_solution, f)

print("solution saved in ", sol_file_name)

# Time information about solving
solving_info = {
    #"time_start": time_start, "time_end": time_end,
    "search-time": time_end-time_start,
    "cpu_times": process.cpu_times(), "cpu_percent": process.cpu_percent(), "times": os.times()
}
with open(dir_name+"/solving-info.json", "w") as f:
    json.dump(solving_info, f)

import pprint
pprint.pprint(v)


prob_map = np.zeros((SIZE_Y, SIZE_X))
for x in range(SIZE_X):
    for y in range(SIZE_Y):
        prob_map[y,x] = true_dist[pos_to_index(x,y)-1]
plot_trajectory.plot_trajectory(dir_name, drone_index_pos_list, SIZE_X, SIZE_Y, prob_map=prob_map)
# # plot_trajectory.plot_trajectory(dir_name, drone_index_pos_list, SIZE_X, SIZE_Y, prob_map=density_dist)
# plot_trajectory.plot_trajectory(dir_name, drone_index_pos_list, SIZE_X, SIZE_Y, prob_map=joint_state)

plt.show()


if m.status == grp.GRB.OPTIMAL:
    print("Optimal solution found.")
else:
    print("No optimal solution found.")

start_time = time.time()
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")









