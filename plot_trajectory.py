
import networkx as nx
import matplotlib.pyplot as plt
from netgraph import Graph
import numpy as np

def highlight_cell(x, y, ax=None, **kwargs):
    rect = plt.Rectangle((x-0.5, y-0.5), 1, 1, **kwargs)
    ax = ax or plt.gca()
    ax.add_patch(rect)
    return rect

def pos_to_index(x, y, SIZE_X):
    return x + y * SIZE_X + 1

def index_to_pos(idx, SIZE_X):
    x = (idx - 1) % SIZE_X
    y = (idx - 1) // SIZE_X
    return x, y

def plot_trajectory(dir_name, trajectory, SIZE_X, SIZE_Y, prob_map=None):


    for x in range(SIZE_X):
        for y in range(SIZE_Y):
            highlight_cell(x, y, fill=False, edgecolor='k', linewidth=1)

    # Build the graph from the trajectory
    G = nx.DiGraph()
    if len(trajectory) == 1:
        G.add_edge(trajectory[0], trajectory[0])
    else:
        for i in range(1, len(trajectory)):
            G.add_edge(trajectory[i-1], trajectory[i])


    node_position_full = {pos_to_index(x, y, SIZE_X): np.array([x, y])
                          for x in range(SIZE_X) for y in range(SIZE_Y)}
    nodes_in_G = list(G.nodes())
    node_position = {n: node_position_full[n] for n in nodes_in_G}


    node_colors = {}
    # Default white
    for n in nodes_in_G:
        node_colors[n] = 'white'
    # Highlight first and last nodes
    if len(trajectory) > 0:
        first_node = trajectory[0]
        if first_node in node_colors:
            node_colors[first_node] = 'green'
    if len(trajectory) > 1:
        last_node = trajectory[-1]
        if last_node in node_colors:
            node_colors[last_node] = 'red'


    if prob_map is not None:
        plt.imshow(prob_map, cmap="plasma", origin='lower',
                   extent=(-0.5, SIZE_X - 0.5, -0.5, SIZE_Y - 0.5))

    # node labels
    node_labels = {}
    for n in nodes_in_G:
        x, y = index_to_pos(n, SIZE_X)
        if prob_map is not None:
            p = prob_map[y, x]
            node_labels[n] = f"{n}\np={p:.2f}"
        else:
            node_labels[n] = f"{n}"

    Graph(
        G,
        node_layout=node_position,
        edge_layout='curved', edge_layout_kwargs=dict(k=0.1),
        origin=(0,0), scale=(SIZE_X,SIZE_Y),
        #edge_label_fontdict=dict(size=5, fontweight='bold'),
        node_size=19,
        node_shape='o',
        node_labels=node_labels,
        node_label_fontdict=dict(size=6, fontweight='bold', color='black', ha='center', va='center'),
        node_label_offset=(0,0),
        node_edge_color='black',
        node_edge_width=2,
        node_color=node_colors,   #
        edge_color='white',
        edge_width=3,
        edge_alpha=1,
        arrows=True
    )

    plt.savefig(dir_name + "/fig-optimal-path.pdf", bbox_inches='tight')
    plt.savefig(dir_name + "/fig-optimal-path.png", bbox_inches='tight')
    plt.show()

