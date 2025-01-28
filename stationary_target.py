# #!/usr/bin/env python
# # coding: utf-8
#
# #%%
# import math
# import random
# from itertools import combinations
# import gurobipy as grp
# from gurobipy import GRB
# import matplotlib.pyplot as plt
# import numpy as np
# import time
# import os
# import argparse
# import sys
# import random
# import json
# import time
# import psutil
#
# #%%
#
# M = 100
# DEFAULT_SIZE_X = 3
# DEFAULT_SIZE_Y = 3
# DEFAULT_SEED = 1
# DEFAULT_HOVER_TIME = 5
# DEFAULT_SPEED = 1
#
# prefix = "ucsst"
#
# def get_args(args_str=None):
#     if args_str is not None:
#         arg_list = args_str.split(" ")
#     elif "pydevconsole" in sys.argv[0]:
#         arg_list = sys.argv[3:]
#     else:
#         arg_list = sys.argv [1:]
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--x-size", type=int, default=DEFAULT_SIZE_X)
#     parser.add_argument("--y-size", type=int, default=DEFAULT_SIZE_Y)
#     parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
#     parser.add_argument("--speed", type=float, default=DEFAULT_SPEED)
#     parser.add_argument("--no-show", action="store_true", default=False)
#     parser.add_argument("--hover-time", type=float, default=DEFAULT_HOVER_TIME)
#     parser.add_argument("--save-problem", action="store_true", default=False)
#     parser.add_argument("--uniform-poc", action="store_true", default=False)
#     parser.add_argument("--nb-poc", type=int, default=None)
#     parser.add_argument("--extra-constraint", action="store_true", default=False)
#     parser.add_argument("--adjacent-motion", action="store_true", default=False)
#     parser.add_argument("--special-curved-arrows", action="store_true", default=False)
#     args = parser.parse_args(arg_list)
#     return args
#
# ARGS_STR = None
#
# args = get_args(ARGS_STR)
# SIZE_X = args.x_size
# SIZE_Y = args.y_size
# SEED = args.seed
# NO_SHOW = args.no_show
# HOVER_TIME = args.hover_time
# SAVE_PROB = args.save_problem
# SPEED = args.speed
# UNIFORM_POC = args.uniform_poc
# NB_POC = args.nb_poc
# EXTRA_CONSTRAINT = args.extra_constraint
# ADJACENT_MOTION = args.adjacent_motion
# SPECIAL_CURVED_ARROWS = args.special_curved_arrows
#
# def get_dir_name():
#     global SIZE_X, SIZE_Y, SEED
#     dir_name = prefix + f"-{SIZE_X}x{SIZE_Y}"
#     if SPEED != DEFAULT_SPEED:
#         dir_name += f"-sp{int(SPEED)}" if (int(SPEED) == SPEED) else f"-sp{SPEED}"
#     if HOVER_TIME != DEFAULT_HOVER_TIME:
#         if int(HOVER_TIME) == HOVER_TIME:
#             dir_name += f"-h{int(HOVER_TIME)}"
#         else:
#             dir_name += f"-h{HOVER_TIME}"
#     if EXTRA_CONSTRAINT:
#         dir_name += f"-extra"
#     if UNIFORM_POC:
#         dir_name += "-uniform"
#     if NB_POC is not None:
#         dir_name += f"-poc{NB_POC}"
#
#     if SEED != DEFAULT_SEED:
#         dir_name += f"-s{SEED}"
#     if ADJACENT_MOTION:
#         dir_name += f"-am"
#
#     return dir_name
#
# #%%
#
# time_start = time.time()
# process = psutil.Process()
# _ignored = process.cpu_percent()
#
# np.random.seed(SEED)
# random.seed(SEED)
#
# dir_name = get_dir_name()
# if not os.path.exists(dir_name):
#     os.mkdir(dir_name)
#
# if SIZE_X == 3 and SIZE_Y == 4:
#     prob_map = np.array([[0, 0, 0.5],[0.1, 0.03, 0.07],[0.2, 0.05, 0.05]])
# else:
#     if UNIFORM_POC:
#         prob_map = np.ones(shape=(SIZE_Y, SIZE_X))
#     else:
#         prob_map = np.random.random(size=(SIZE_Y, SIZE_X))
#
#     if NB_POC is not None:
#         pos_list = [(y,x) for x in range(SIZE_X) for y in range(SIZE_Y)]
#         unselected_idx = np.random.choice(range(len(pos_list)), size=SIZE_X*SIZE_Y-NB_POC, replace=False)
#         for i in unselected_idx:
#             prob_map[pos_list[i]] = 0
#
#     prob_map = prob_map / np.sum(prob_map)
#
# start = prob_map[0,0]
# if not NO_SHOW:
#     fig = plt.figure(figsize=(5, 6))
#     plt.imshow(prob_map, cmap="plasma")
#     plt.colorbar() #cmap="plasma")
#     plt.savefig("colormap.pdf")
#     plt.savefig("colormap.png")
#     plt.show()
#
#
#
# #%%
#
# m = grp.Model(name="ucsst Model")
#
# set_X = range(1, SIZE_X+1)
# set_Y = range(1, SIZE_Y+1)
#
# def pos_to_index(x,y):
#     return x+y*SIZE_X+1
#
# def index_to_pos(idx):
#     assert 1 <= idx and idx < pos_to_index(SIZE_X, SIZE_Y)
#     return (idx-1) % SIZE_X, (idx-1) // SIZE_X
#
# set_V = list(range(1, SIZE_X*SIZE_Y+1))
# start_index = 0
# end_index = max(set_V)+1
#
# set_W = set_V.copy()
# set_W.extend([start_index, end_index])
#
# # binary variables for the visited edges:
# x  = { (i,j):m.addVar(vtype=GRB.BINARY,
#                         name="x_{0}_{1}".format(i,j))
#    for i in set_W for j in set_W if i!=j }
#
# print(set_V)
#
# if ADJACENT_MOTION:
#     for i in set_V:
#         for j in set_V:
#             if i==j:
#                 continue
#             xy1 = np.array(index_to_pos(i))
#             xy2 = np.array(index_to_pos(j))
#             if np.abs(xy1-xy2).sum() != 1:
#                 constr_flow = m.addConstr(
#                     lhs=x[(i,j)],
#                     sense=GRB.EQUAL,
#                     rhs=0)
#
#
# # variables for time of visit of each point:
# t = { i: m.addVar(vtype=GRB.CONTINUOUS, name=f"t_{i}", lb=0)
#       for i in set_W }
#
# def compute_all_distances():
#     d = {
#       (i,j): np.linalg.norm( (np.array(index_to_pos(i)) - np.array(index_to_pos(j))) )
#       for i in set_V
#       for  j in set_V }
#     for i in set_W:
#         d[start_index,i] = 0
#         d[i,start_index] = 0
#         d[end_index,i] = 0
#         d[i,end_index] = 0
#     return d
#
# distance_table = compute_all_distances()
# tau = { edge: dist/SPEED for edge,dist in distance_table.items() }
#
#
# # binary variables for the visited edges:
# if EXTRA_CONSTRAINT:
#     y  = { (i,j):m.addVar(vtype=GRB.BINARY,
#                             name="y_{0}_{1}".format(i,j))
#        for i in set_V for j in set_V if i!=j }
#
#
#
# c = 1
# for i in set_V:
#     outgoing_edges = [x[i,j] for j in set_W if j != i]
#     incoming_edges = [x[j,i] for j in set_W if j != i]
#     constr_flow = m.addConstr(
#       lhs=grp.quicksum( incoming_edges),
#       sense=GRB.EQUAL,
#       rhs=grp.quicksum( outgoing_edges))
#     constr_visits = m.addConstr(
#       lhs=grp.quicksum( incoming_edges),sense=GRB.EQUAL, rhs=c) # XXX: can be GRB.LESS_EQUAL
#
# for i in [start_index]:
#     outgoing_edges = [x[i,j] for j in set_W if j != i]
#     incoming_edges = [x[j,i] for j in set_W if j != i]
#     constr_flow = m.addConstr(
#       lhs=grp.quicksum( incoming_edges),
#       sense=GRB.EQUAL,
#       rhs= 0)
#     constr_visits = m.addConstr(
#       lhs=grp.quicksum(outgoing_edges),sense=GRB.EQUAL, rhs=c)
#
# for i in [end_index]:
#     outgoing_edges = [x[i,j] for j in set_W if j != i]
#     incoming_edges = [x[j,i] for j in set_W if j != i]
#     constr_flow = m.addConstr(
#       lhs=grp.quicksum( incoming_edges),
#       sense=GRB.EQUAL,
#       rhs= c)
#     constr_visits = m.addConstr(
#       lhs=grp.quicksum(outgoing_edges),sense=GRB.EQUAL, rhs=0)
#
#
#
# hover = HOVER_TIME
# for i in set_W:
#     for j in set_W:
#         if i == j:
#             continue
#         time_constraint =m.addConstr(t[j] >= t[i] + tau[i,j] + hover - M*(1-x[i,j]) )
#
# if EXTRA_CONSTRAINT:
#     for i in set_V:
#         for j in set_V:
#             if i == j:
#                 continue
#             m.addConstr(t[j] >= t[i] + tau[i, j] + hover - M * (y[i, j]))
#             m.addConstr(t[i] >= t[j] + tau[i, j] + hover - M * (1 - y[i, j]))
#
#
#
# if SAVE_PROB:
#     m.write(dir_name+"/problem.mps")
#     m.write(dir_name+"/problem.lp")
#
# #%%
#
#
# def swap(xy):
#     return xy[1],xy[0]
# objective = grp.quicksum(t[i]* prob_map[swap(index_to_pos(i))] for i in set_V)
# m.ModelSense = grp.GRB.MINIMIZE
# m.setObjective(objective)
# m.optimize()
#
# time_end = time.time()
#
#
# #%%
#
#
# solution_edges = []
#
# for edge,selected in x.items():
#     if selected.X >= 1:
#         print(edge)
#         solution_edges.append(edge)
#
#
# #%%
#
# # Save the solution in file(s)
# m.write(dir_name+"/solution.sol")
# with open(dir_name+"/solution.json", "w") as f:
#     json.dump(m.getJSONSolution(), f)
#
# simple_solution = {
#     "edge_list": solution_edges,
#     "t": { k:v.X  for k,v in t.items()},
#     "objective": m.ObjVal
# }
# with open(dir_name+"/solution-short.json", "w") as f:
#     json.dump(simple_solution, f)
#
#
#
# solving_info = {
#     "time_start": time_start, "time_end": time_end, "time_elapsed": time_end-time_start,
#     "cpu_times": process.cpu_times(), "cpu_percent": process.cpu_percent(), "times": os.times()
# }
# with open(dir_name+"/solving-info.json", "w") as f:
#     json.dump(solving_info, f)
#
# #%%
#
# import networkx as nx
# import matplotlib.pyplot as plt
# from netgraph import Graph
# import netgraph
#
# def highlight_cell(x, y, ax=None, **kwargs):
#     rect = plt.Rectangle((x-.5, y-.5), 1, 1, fill=False, **kwargs)
#     ax = ax or plt.gca()
#     ax.add_patch(rect)
#     return rect
#
# # Highlight all cells
# for x in range(SIZE_X):
#     for y in range(SIZE_Y):
#         highlight_cell(x, y)
#
#
# node_position = {}
# node_labels = {}
# for y in range(SIZE_Y):
#     for x in range(SIZE_X):
#         idx = pos_to_index(x, y)
#         node_position[idx] = np.array([x, y])
#         prob_value = prob_map[y, x]
#         node_labels[idx] = f'{idx}\n p={prob_value:.2f}'
#
#
# special_solutions_edges = []
#
# for u,v in solution_edges:
#     if u in set_V and v in set_V:
#         ux,uy = node_position[u]
#         vx,vy = node_position[v]
#         if (ux==vx or uy==vy) and SPECIAL_CURVED_ARROWS:
#             if abs(ux-vx)>=2 or abs(uy-vy)>=2:
#                 special_solutions_edges.append((u,v))
#
# first_part_solution_edges = [edge for edge in solution_edges if edge not in special_solutions_edges]
# second_part_solution_edges = list(set(solution_edges).difference(set(first_part_solution_edges)))
#
# # Create a directed graph for the solution edges
# G = nx.DiGraph()
# for edge in first_part_solution_edges:
#     if edge[0] in set_V and edge[1] in set_V:
#         G.add_edge(edge[0], edge[1])
#
# def nodes_from_edge_list(edge_list):
#     return set([u for u,v in edge_list] + [v for u,v in edge_list])
#
# nodes_in_solution = nodes_from_edge_list(solution_edges) # actually == set_V
# for node in nodes_in_solution:
#     if node in set_V:
#         G.add_node(node)
#
# node_colors = {v:"w" for v in set_V }
# first_node = [v for (u,v) in solution_edges if u == start_index][0]
# last_node = [u for (u,v) in solution_edges if v == end_index][0]
# node_colors[first_node] = "green"
# node_colors[last_node]  = "black"
#
# print("G base")
# Graph(G,
#       node_labels=node_labels,
#       edge_label_fontdict=dict(size=5, fontweight='bold'),
#       node_layout=node_position,
#       node_color = node_colors,
#       edge_layout='curved',
#       edge_layout_kwargs=dict(k=0.1),
#       node_size=18,
#       origin=(0,0), scale=(SIZE_X, SIZE_Y),
#       edge_color="w",
#       edge_alpha=1,
#       edge_width=3,
#       arrows=True)
#
# additional_edge_graph = nx.DiGraph()
# for u,v in second_part_solution_edges:
#     if u in set_V and v in set_V:
#         additional_edge_graph.add_edge(u, v)
#
# print("additional G")
#
# additional_edge_graph = Graph(
#     additional_edge_graph,
#    node_layout=node_position,
#    edge_layout='arc',
#    edge_layout_kwargs=dict(rad=0.3),
#    edge_color="w",
#    edge_alpha=1,
#    edge_width=3,
#    arrows=True,
#    node_size=18,
#   origin=(0,0), scale=(SIZE_X, SIZE_Y),
#    )
#
# plt.imshow(prob_map, cmap="plasma")
#
# print(dir_name + "/fig-optimal-path.pdf")
# plt.savefig(dir_name + "/fig-optimal-path.pdf", bbox_inches='tight')
# plt.savefig(dir_name + "/fig-optimal-path.png", bbox_inches='tight')
#
#
# if not NO_SHOW:
#     plt.show()
#
# print(dir_name, solving_info)
#
#
#
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
import time
import psutil

#%%

M = 100
DEFAULT_SIZE_X = 3
DEFAULT_SIZE_Y = 3
DEFAULT_SEED = 1
DEFAULT_HOVER_TIME = 1
DEFAULT_SPEED = 1

prefix = "ucsst"

def get_args(args_str=None):
    if args_str is not None:
        arg_list = args_str.split(" ")
    elif "pydevconsole" in sys.argv[0]:
        arg_list = sys.argv[3:]
    else:
        arg_list = sys.argv [1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("--x-size", type=int, default=DEFAULT_SIZE_X)
    parser.add_argument("--y-size", type=int, default=DEFAULT_SIZE_Y)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--speed", type=float, default=DEFAULT_SPEED)
    parser.add_argument("--no-show", action="store_true", default=False)
    parser.add_argument("--hover-time", type=float, default=DEFAULT_HOVER_TIME)
    parser.add_argument("--save-problem", action="store_true", default=False)
    parser.add_argument("--uniform-poc", action="store_true", default=False)
    parser.add_argument("--nb-poc", type=int, default=None)
    parser.add_argument("--extra-constraint", action="store_true", default=False)
    parser.add_argument("--adjacent-motion", action="store_true", default=False)
    parser.add_argument("--special-curved-arrows", action="store_true", default=False)
    args = parser.parse_args(arg_list)
    return args

ARGS_STR = None

args = get_args(ARGS_STR)
SIZE_X = args.x_size
SIZE_Y = args.y_size
SEED = args.seed
NO_SHOW = args.no_show
HOVER_TIME = args.hover_time
SAVE_PROB = args.save_problem
SPEED = args.speed
UNIFORM_POC = args.uniform_poc
NB_POC = args.nb_poc
EXTRA_CONSTRAINT = args.extra_constraint
ADJACENT_MOTION = args.adjacent_motion
SPECIAL_CURVED_ARROWS = args.special_curved_arrows

def get_dir_name():
    global SIZE_X, SIZE_Y, SEED
    dir_name = prefix + f"-{SIZE_X}x{SIZE_Y}"
    if SPEED != DEFAULT_SPEED:
        dir_name += f"-sp{int(SPEED)}" if (int(SPEED) == SPEED) else f"-sp{SPEED}"
    if HOVER_TIME != DEFAULT_HOVER_TIME:
        if int(HOVER_TIME) == HOVER_TIME:
            dir_name += f"-h{int(HOVER_TIME)}"
        else:
            dir_name += f"-h{HOVER_TIME}"
    if EXTRA_CONSTRAINT:
        dir_name += f"-extra"
    if UNIFORM_POC:
        dir_name += "-uniform"
    if NB_POC is not None:
        dir_name += f"-poc{NB_POC}"

    if SEED != DEFAULT_SEED:
        dir_name += f"-s{SEED}"
    if ADJACENT_MOTION:
        dir_name += f"-am"

    return dir_name

#%%

time_start = time.time()
process = psutil.Process()
_ignored = process.cpu_percent()

np.random.seed(SEED)
random.seed(SEED)

dir_name = get_dir_name()
if not os.path.exists(dir_name):
    os.mkdir(dir_name)

if SIZE_X == 3 and SIZE_Y == 4:
    prob_map = np.array([[0, 0, 0.5],[0.1, 0.03, 0.07],[0.2, 0.05, 0.05]])
else:
    if UNIFORM_POC:
        prob_map = np.ones(shape=(SIZE_Y, SIZE_X))
    else:
        prob_map = np.random.random(size=(SIZE_Y, SIZE_X))

    if NB_POC is not None:
        pos_list = [(y,x) for x in range(SIZE_X) for y in range(SIZE_Y)]
        unselected_idx = np.random.choice(range(len(pos_list)), size=SIZE_X*SIZE_Y-NB_POC, replace=False)
        for i in unselected_idx:
            prob_map[pos_list[i]] = 0

    prob_map = prob_map / np.sum(prob_map)

# Flip the probability map so that the bottom row corresponds to y=0
prob_map = np.flipud(prob_map)

start = prob_map[0,0]
if not NO_SHOW:
    fig = plt.figure(figsize=(5, 6))
    plt.imshow(prob_map, cmap="plasma", origin='lower')  # origin='lower' so bottom-left is (0,0)
    plt.colorbar()
    plt.savefig("colormap.pdf")
    plt.savefig("colormap.png")
    plt.show()

#%%

m = grp.Model(name="ucsst Model")

set_X = range(1, SIZE_X+1)
set_Y = range(1, SIZE_Y+1)

def pos_to_index(x,y):
    # (x,y) with y=0 is bottom row, x=0 is left column
    # Node numbering: start from bottom-left corner, go left to right, then up.
    #return x + y*SIZE_X + 1
    return x + y * SIZE_X + 1

def index_to_pos(idx):
    # Reverse of pos_to_index
    #return ( (idx-1) % SIZE_X, (idx-1) // SIZE_X )
    x = (idx - 1) % SIZE_X
    y = (idx - 1) // SIZE_X
    return x, y

set_V = list(range(1, SIZE_X*SIZE_Y+1))
start_index = 0
end_index = max(set_V)+1

set_W = set_V.copy()
set_W.extend([start_index, end_index])

# binary variables for the visited edges:
x  = { (i,j):m.addVar(vtype=GRB.BINARY,
                        name="x_{0}_{1}".format(i,j))
   for i in set_W for j in set_W if i!=j }

if ADJACENT_MOTION:
    for i in set_V:
        for j in set_V:
            if i==j:
                continue
            xy1 = np.array(index_to_pos(i))
            xy2 = np.array(index_to_pos(j))
            if np.abs(xy1-xy2).sum() != 1:
                constr_flow = m.addConstr(
                    lhs=x[(i,j)],
                    sense=GRB.EQUAL,
                    rhs=0)

# variables for time of visit of each point:
t = { i: m.addVar(vtype=GRB.CONTINUOUS, name=f"t_{i}", lb=0)
      for i in set_W }

def compute_all_distances():
    d = {
      (i,j): np.linalg.norm( (np.array(index_to_pos(i)) - np.array(index_to_pos(j))) )
      for i in set_V
      for  j in set_V }
    for i in set_W:
        d[start_index,i] = 0
        d[i,start_index] = 0
        d[end_index,i] = 0
        d[i,end_index] = 0
    return d

distance_table = compute_all_distances()
tau = { edge: dist/SPEED for edge,dist in distance_table.items() }

if EXTRA_CONSTRAINT:
    y  = { (i,j):m.addVar(vtype=GRB.BINARY,
                            name="y_{0}_{1}".format(i,j))
       for i in set_V for j in set_V if i!=j }

c = 1
for i in set_V:
    outgoing_edges = [x[i,j] for j in set_W if j != i]
    incoming_edges = [x[j,i] for j in set_W if j != i]
    constr_flow = m.addConstr(
      lhs=grp.quicksum( incoming_edges),
      sense=GRB.EQUAL,
      rhs=grp.quicksum( outgoing_edges))
    constr_visits = m.addConstr(
      lhs=grp.quicksum( incoming_edges),sense=GRB.EQUAL, rhs=c) # XXX: can be GRB.LESS_EQUAL

for i in [start_index]:
    outgoing_edges = [x[i,j] for j in set_W if j != i]
    incoming_edges = [x[j,i] for j in set_W if j != i]
    constr_flow = m.addConstr(
      lhs=grp.quicksum( incoming_edges),
      sense=GRB.EQUAL,
      rhs= 0)
    constr_visits = m.addConstr(
      lhs=grp.quicksum(outgoing_edges),sense=GRB.EQUAL, rhs=c)

for i in [end_index]:
    outgoing_edges = [x[i,j] for j in set_W if j != i]
    incoming_edges = [x[j,i] for j in set_W if j != i]
    constr_flow = m.addConstr(
      lhs=grp.quicksum( incoming_edges),
      sense=GRB.EQUAL,
      rhs= c)
    constr_visits = m.addConstr(
      lhs=grp.quicksum(outgoing_edges),sense=GRB.EQUAL, rhs=0)

hover = HOVER_TIME
for i in set_W:
    for j in set_W:
        if i == j:
            continue
        time_constraint =m.addConstr(t[j] >= t[i] + tau[i,j] + hover - M*(1-x[i,j]) )

if EXTRA_CONSTRAINT:
    for i in set_V:
        for j in set_V:
            if i == j:
                continue
            m.addConstr(t[j] >= t[i] + tau[i, j] + hover - M * (y[i, j]))
            m.addConstr(t[i] >= t[j] + tau[i, j] + hover - M * (1 - y[i, j]))

if SAVE_PROB:
    m.write(dir_name+"/problem.mps")
    m.write(dir_name+"/problem.lp")

#%%

def swap(xy):
    return xy[1], xy[0]

# After flipping prob_map, prob_map[y,x] now corresponds to bottom-left as (y=0, x=0)
# index_to_pos gives (x,y) with bottom-left as (0,0)
# swap(index_to_pos(i)) gives (y,x) for indexing into prob_map.
objective = grp.quicksum(t[i]* prob_map[swap(index_to_pos(i))] for i in set_V)
m.ModelSense = grp.GRB.MINIMIZE
m.setObjective(objective)
m.optimize()

time_end = time.time()

#%%

solution_edges = []
for edge,selected in x.items():
    if selected.X >= 1:
        print(edge)
        solution_edges.append(edge)

#%%

m.write(dir_name+"/solution.sol")
with open(dir_name+"/solution.json", "w") as f:
    json.dump(m.getJSONSolution(), f)

simple_solution = {
    "edge_list": solution_edges,
    "t": { k:v.X  for k,v in t.items()},
    "objective": m.ObjVal
}
with open(dir_name+"/solution-short.json", "w") as f:
    json.dump(simple_solution, f)

solving_info = {
    "time_start": time_start, "time_end": time_end, "time_elapsed": time_end-time_start,
    "cpu_times": process.cpu_times(), "cpu_percent": process.cpu_percent(), "times": os.times()
}
with open(dir_name+"/solving-info.json", "w") as f:
    json.dump(solving_info, f)

#%%

import networkx as nx
import matplotlib.pyplot as plt
from netgraph import Graph
import netgraph

def highlight_cell(x, y, ax=None, **kwargs):
    rect = plt.Rectangle((x-.5, y-.5), 1, 1, fill=False, **kwargs)
    ax = ax or plt.gca()
    ax.add_patch(rect)
    return rect

for x in range(SIZE_X):
    for y in range(SIZE_Y):
        highlight_cell(x, y)

node_position = {}
node_labels = {}
for x in range(SIZE_X):
    for y in range(SIZE_Y):
        idx = pos_to_index(x, y)
        node_position[idx] = np.array([x, y])
        prob_value = prob_map[y, x]
        node_labels[idx] = f'{idx}\n p={prob_value:.2f}'

special_solutions_edges = []
for u,v in solution_edges:
    if u in set_V and v in set_V:
        ux,uy = node_position[u]
        vx,vy = node_position[v]
        if (ux==vx or uy==vy) and SPECIAL_CURVED_ARROWS:
            if abs(ux-vx)>=2 or abs(uy-vy)>=2:
                special_solutions_edges.append((u,v))

first_part_solution_edges = [edge for edge in solution_edges if edge not in special_solutions_edges]
second_part_solution_edges = list(set(solution_edges).difference(set(first_part_solution_edges)))

G = nx.DiGraph()
for edge in first_part_solution_edges:
    if edge[0] in set_V and edge[1] in set_V:
        G.add_edge(edge[0], edge[1])

def nodes_from_edge_list(edge_list):
    return set([u for u,v in edge_list] + [v for u,v in edge_list])

nodes_in_solution = nodes_from_edge_list(solution_edges)
for node in nodes_in_solution:
    if node in set_V:
        G.add_node(node)

node_colors = {v:"w" for v in set_V }
first_node = [v for (u,v) in solution_edges if u == start_index][0]
last_node = [u for (u,v) in solution_edges if v == end_index][0]
node_colors[first_node] = "green"
node_colors[last_node]  = "red"

print("G base")
Graph(G,
      node_labels=node_labels,
      edge_label_fontdict=dict(size=10, fontweight='bold'),
      node_layout=node_position,
      node_color = node_colors,
      edge_layout='curved',
      edge_layout_kwargs=dict(k=0.1),
      node_size=20,
      origin=(0,0), scale=(SIZE_X, SIZE_Y),
      edge_color="w",
      edge_alpha=1,
      edge_width=3,
      arrows=True)

additional_edge_graph = nx.DiGraph()
for u,v in second_part_solution_edges:
    if u in set_V and v in set_V:
        additional_edge_graph.add_edge(u, v)

print("additional G")

additional_edge_graph = Graph(
    additional_edge_graph,
   node_layout=node_position,
   edge_layout='arc',
   edge_layout_kwargs=dict(rad=0.3),
   edge_color="w",
   edge_alpha=1,
   edge_width=3,
   arrows=True,
   node_size=20,
   origin=(0,0), scale=(SIZE_X, SIZE_Y)
)

plt.imshow(prob_map, cmap="plasma", origin='upper')
print(dir_name + "/fig-optimal-path.pdf")
plt.savefig(dir_name + "/fig-optimal-path.pdf", bbox_inches='tight')
plt.savefig(dir_name + "/fig-optimal-path.png", bbox_inches='tight')

if not NO_SHOW:
    plt.show()

print(dir_name, solving_info)
