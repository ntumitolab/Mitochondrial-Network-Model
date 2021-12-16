import random as rd
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import math

# Create class: Nodes
class Nodes:
    def __init__(self, name, deg, nei=[]):
        self.init_name = name
        self.name = name  # name of the node
        self.deg = deg  # degree
        self.nei = [i for i in nei]

    def _add_nei(self, nei_obj):
        self.nei.append(nei_obj)

    def _remove_nei(self, nei_obj):
        self.nei.remove(nei_obj)

    def move(self, new_name, new_deg, all_nodes):
        # check if the move is valid
        old_deg = self.deg
        if old_deg == new_deg:
            print(f"Error: Cannot move node from group {old_deg} to group {new_deg}")

        self.name = new_name
        self.deg = new_deg
        all_nodes[old_deg - 1].remove(self)
        all_nodes[new_deg - 1].append(self)


# Functions
def get_propensity(all_nodes, rates):
    x1 = len(all_nodes[0])
    x2 = len(all_nodes[1])
    x3 = len(all_nodes[2])

    a1, b1, a2, b2 = rates
    propensities = [a1 * x1 * (x1 - 1), b1 * x2, a2 * x1 * x2, b2 * x3]
    return propensities


def pick_reaction(p):
    p = np.array(p)
    noise = np.array([-1, -1, -1, -1])
    while any(elem <= 0 for elem in noise):
        noise = np.random.normal(loc=1.0, scale=0.5, size=4)
    rp = p * noise
    x = rd.uniform(0, 1) * sum(rp)
    cum_rp = [rp[0], rp[0] + rp[1], rp[0] + rp[1] + rp[2], sum(rp)]
    for i in range(len(rp)):
        if x <= cum_rp[i]:
            return i + 1
    print("Error: No reaction is picked")


""" No add noise
def pick_reaction(p):
    x = rd.uniform(0,1) * sum(p)
    cum_p = [p[0], p[0]+p[1], p[0]+p[1]+p[2], sum(p)]
    for i in range(len(p)):
        if x <= cum_p[i]:
            return i+1
    print("Error: No reaction is picked")
"""


def get_graph_spring(all_nodes, draw="T"):
    n1 = []
    n2 = []
    n3 = []
    edge_list = []
    for i in range(3):
        if i == 0:
            n1.extend([j.name for j in all_nodes[i]])
            for k in range(len(all_nodes[0])):
                if (
                    sorted([all_nodes[0][k].name, all_nodes[0][k].nei[0].name])
                    not in edge_list
                ):
                    edge_list.append(
                        [all_nodes[0][k].name, all_nodes[0][k].nei[0].name]
                    )
        elif i == 1:
            n2.extend([j.name for j in all_nodes[i]])
            for k in range(len(all_nodes[1])):
                if (
                    sorted([all_nodes[1][k].name, all_nodes[1][k].nei[0].name])
                    not in edge_list
                ):
                    edge_list.append(
                        [all_nodes[1][k].name, all_nodes[1][k].nei[0].name]
                    )
        elif i == 2:
            n3.extend([j.name for j in all_nodes[i]])
            for k in range(len(all_nodes[2])):
                if (
                    sorted([all_nodes[2][k].name, all_nodes[2][k].nei[0].name])
                    not in edge_list
                ):
                    edge_list.append(
                        [all_nodes[2][k].name, all_nodes[2][k].nei[0].name]
                    )
    node_list = list(set(n1 + n2 + n3))

    G = nx.Graph()
    G.add_nodes_from(node_list)
    G.add_edges_from(edge_list)
    # count average deg
    deg_G = [deg for (node, deg) in G.degree()]
    avg_deg_G = sum(deg_G) / len(deg_G)
    info = [
        len(deg_G),
        len([node for (node, deg) in G.degree() if deg == 1]),
        len([node for (node, deg) in G.degree() if deg == 2]),
        len([node for (node, deg) in G.degree() if deg == 3]),
        avg_deg_G,
        G.number_of_edges(),
    ]
    if draw == "T":
        pos = nx.spring_layout(G, k=0.2, iterations=20)
        # pos = nx.planar_layout(G)
        nx.draw_networkx_nodes(G, pos, nodelist=n1, node_color="b")
        nx.draw_networkx_nodes(G, pos, nodelist=n2, node_color="g")
        nx.draw_networkx_nodes(G, pos, nodelist=n3, node_color="r")
        nx.draw_networkx_edges(G, pos, width=2.0, alpha=0.5)
        nx.draw_networkx_labels(G, pos, font_color="w", font_size=10)
        plt.show()

    # print(f'The average degree of the network is {avg_deg_G}')
    # '#Nodes':[], 'Degree1':[], 'Degree2':[], 'Degree3':[],'AvgDegree':[], '#Edges':[]

    return info


def get_graph_planar(all_nodes, draw="T"):
    n1 = []
    n2 = []
    n3 = []
    edge_list = []
    for i in range(3):
        if i == 0:
            n1.extend([j.name for j in all_nodes[i]])
            for k in range(len(all_nodes[0])):
                if (
                    sorted([all_nodes[0][k].name, all_nodes[0][k].nei[0].name])
                    not in edge_list
                ):
                    edge_list.append(
                        [all_nodes[0][k].name, all_nodes[0][k].nei[0].name]
                    )
        elif i == 1:
            n2.extend([j.name for j in all_nodes[i]])
            for k in range(len(all_nodes[1])):
                if (
                    sorted([all_nodes[1][k].name, all_nodes[1][k].nei[0].name])
                    not in edge_list
                ):
                    edge_list.append(
                        [all_nodes[1][k].name, all_nodes[1][k].nei[0].name]
                    )
        elif i == 2:
            n3.extend([j.name for j in all_nodes[i]])
            for k in range(len(all_nodes[2])):
                if (
                    sorted([all_nodes[2][k].name, all_nodes[2][k].nei[0].name])
                    not in edge_list
                ):
                    edge_list.append(
                        [all_nodes[2][k].name, all_nodes[2][k].nei[0].name]
                    )
    node_list = list(set(n1 + n2 + n3))

    G = nx.Graph()
    G.add_nodes_from(node_list)
    G.add_edges_from(edge_list)
    # pos = nx.spring_layout(G, k=0.2, iterations=20)
    if draw == "T":
        pos = nx.planar_layout(G)
        nx.draw_networkx_nodes(G, pos, nodelist=n1, node_color="b")
        nx.draw_networkx_nodes(G, pos, nodelist=n2, node_color="g")
        nx.draw_networkx_nodes(G, pos, nodelist=n3, node_color="r")
        nx.draw_networkx_edges(G, pos, width=2.0, alpha=0.5)
        nx.draw_networkx_labels(G, pos, font_color="w", font_size=10)
        plt.show()


# get the largest connected component
def get_largest_CC(all_nodes, draw="T"):
    n1 = []
    n2 = []
    n3 = []
    edge_list = []
    for i in range(3):
        if i == 0:
            n1.extend([j.name for j in all_nodes[i]])
            for k in range(len(all_nodes[0])):
                if (
                    sorted([all_nodes[0][k].name, all_nodes[0][k].nei[0].name])
                    not in edge_list
                ):
                    edge_list.append(
                        [all_nodes[0][k].name, all_nodes[0][k].nei[0].name]
                    )
        elif i == 1:
            n2.extend([j.name for j in all_nodes[i]])
            for k in range(len(all_nodes[1])):
                if (
                    sorted([all_nodes[1][k].name, all_nodes[1][k].nei[0].name])
                    not in edge_list
                ):
                    edge_list.append(
                        [all_nodes[1][k].name, all_nodes[1][k].nei[0].name]
                    )
        elif i == 2:
            n3.extend([j.name for j in all_nodes[i]])
            for k in range(len(all_nodes[2])):
                if (
                    sorted([all_nodes[2][k].name, all_nodes[2][k].nei[0].name])
                    not in edge_list
                ):
                    edge_list.append(
                        [all_nodes[2][k].name, all_nodes[2][k].nei[0].name]
                    )
    node_list = list(set(n1 + n2 + n3))

    G = nx.Graph()
    G.add_nodes_from(node_list)
    G.add_edges_from(edge_list)
    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
    G0 = G.subgraph(Gcc[0])
    if len(Gcc) <= 1:
        # count average deg
        deg_G0 = [deg for (node, deg) in G0.degree()]
        avg_deg_G0 = sum(deg_G0) / len(deg_G0)
        info = [
            len(deg_G0),
            len([node for (node, deg) in G0.degree() if deg == 1]),
            len([node for (node, deg) in G0.degree() if deg == 2]),
            len([node for (node, deg) in G0.degree() if deg == 3]),
            avg_deg_G0,
            G0.number_of_edges(),
            0,
            0,
            0,
            0,
            0,
            0,
        ]
    else:
        G1 = G.subgraph(Gcc[1])

        # count average deg
        deg_G0 = [deg for (node, deg) in G0.degree()]
        avg_deg_G0 = sum(deg_G0) / len(deg_G0)
        deg_G1 = [deg for (node, deg) in G1.degree()]
        avg_deg_G1 = sum(deg_G1) / len(deg_G1)
        # print(f'The average degree of the largest connected component is {avg_deg_G0}')
        # '#Nodes':[], 'Degree1':[], 'Degree2':[], 'Degree3':[],'AvgDegree':[], '#Edges':[]
        info = [
            len(deg_G0),
            len([node for (node, deg) in G0.degree() if deg == 1]),
            len([node for (node, deg) in G0.degree() if deg == 2]),
            len([node for (node, deg) in G0.degree() if deg == 3]),
            avg_deg_G0,
            G0.number_of_edges(),
            len(deg_G1),
            len([node for (node, deg) in G1.degree() if deg == 1]),
            len([node for (node, deg) in G1.degree() if deg == 2]),
            len([node for (node, deg) in G1.degree() if deg == 3]),
            avg_deg_G1,
            G1.number_of_edges(),
        ]

    if draw == "T":
        deg_map = {1: "blue", 2: "green", 3: "red"}
        # [deg for (node, deg) in G0.degree()] can get every degree of every node from graph G0
        values = [deg_map.get(deg) for (node, deg) in G0.degree()]
        nx.draw(
            G0, node_color=values, with_labels=True, font_color="white", font_size=10
        )
        plt.show()

    return info


# Iteration
# be aware that n needs to be even number
def Iterate(
    c, n, iterate, plot_t0="F", plot_tend="F"
):  # (rate constant, # of iteration, initialized total nodes)

    # decide the rate constant [a1, b1, a2, b2]
    # rate_constant = [0.0005, 0.01, 0.0005, 0.01]

    # start_time = time.time()
    b1 = 0.01
    b2 = (3 / 2) * b1
    c1, c2 = c
    a1 = c1 * b1
    a2 = c2 * b2
    rate_constant = [a1, b1, a2, b2]

    iteration = iterate

    # dataframe
    dic = {
        "Time": [],
        "#Nodes": [],
        "Degree1": [],
        "Degree2": [],
        "Degree3": [],
        "AvgDegree": [],
        "#Edges": [],
        "#Nodes_G0": [],
        "Degree1_G0": [],
        "Degree2_G0": [],
        "Degree3_G0": [],
        "AvgDegree_G0": [],
        "#Edges_G0": [],
        "#Nodes_G1": [],
        "Degree1_G1": [],
        "Degree2_G1": [],
        "Degree3_G1": [],
        "AvgDegree_G1": [],
        "#Edges_G1": [],
    }
    netinfo = pd.DataFrame(dic)  # network information
    netinfo["Time"] = list(range(iteration + 1))

    # initialization
    All_nodes = [[], [], []]
    N = n  # initialized total number of nodes
    init_names = list(range(1, N + 1))
    init_degs = [1] * N

    Reaction_record = []

    for i in range(N):
        All_nodes[0].append(Nodes(init_names[i], init_degs[i]))
    s_idx = list(range(0, N, 2))
    t_idx = list(range(1, N, 2))
    for i in range(int(N / 2)):
        All_nodes[0][s_idx[i]]._add_nei(All_nodes[0][t_idx[i]])
        All_nodes[0][t_idx[i]]._add_nei(All_nodes[0][s_idx[i]])

    all_info = get_graph_spring(All_nodes, plot_t0)
    netinfo.iloc[0, 1:7] = all_info
    gcc_info = get_largest_CC(All_nodes, "F")
    netinfo.iloc[0, 7:19] = gcc_info

    j = 0

    for i in range(iteration):
        #         if i%100 == 0:
        #             print(f'Iteration {i+1}')
        R = pick_reaction(get_propensity(All_nodes, rate_constant))
        # print(f'Reaction {R}')
        Reaction_record.append(R)
        # 1st reaction
        # X1 + X1 -> X2
        if R == 1:
            if not All_nodes[0]:
                print("No node in deg1")
            else:
                flag = 0
                count = 0
                while (flag == 0) & (count <= 20):
                    count += 1
                    first_node, second_node = rd.sample(All_nodes[0], 2)
                    if (first_node.nei[0] != second_node) & (
                        first_node.nei[0].name != second_node.nei[0].name
                    ):
                        flag = 1
                        j += 1
                        first_node.move(N + j, 2, All_nodes)
                        second_node.move(N + j, 2, All_nodes)
                #        first_node._add_nei(second_node)
                #        second_node._add_nei(first_node)

                if count > 20:
                    # cannot pick nodes successfully
                    # print('Cannot do this reaction')
                    Reaction_record[-1] = f"{R} (but failed)"

        # 2nd reaction
        # X1 + X1 <- X2
        if R == 2:
            if not All_nodes[1]:
                print("No node in deg2")
            else:
                [first_node] = rd.sample(All_nodes[1], 1)
                for d2 in [m for m in All_nodes[1] if m != first_node]:
                    if d2.name == first_node.name:
                        second_node = d2

                j += 1
                #        first_node._remove_nei(second_node)
                first_node.move(N + j, 1, All_nodes)
                j += 1
                #       second_node._remove_nei(first_node)
                second_node.move(N + j, 1, All_nodes)

            # 3rd reaction
            # X1 + X2 -> X3
        if R == 3:
            if not All_nodes[0]:
                print("No node in deg1")
            elif not All_nodes[1]:
                print("No node in deg2")
            else:
                [second_node1] = rd.sample(All_nodes[1], 1)
                for d2 in [m for m in All_nodes[1] if m != second_node1]:
                    if d2.name == second_node1.name:
                        second_node2 = d2
                count = 0
                flag = 0
                while (flag == 0) & (count <= 20):
                    count += 1
                    [first_node] = rd.sample(All_nodes[0], 1)
                    if (first_node.name != second_node1.nei[0].name) & (
                        first_node.name != second_node2.nei[0].name
                    ):
                        if (first_node.nei[0].name != second_node1.nei[0].name) & (
                            first_node.nei[0].name != second_node2.nei[0].name
                        ):
                            flag = 1

                            j += 1
                            first_node.move(N + j, 3, All_nodes)
                            second_node1.move(N + j, 3, All_nodes)
                            second_node2.move(N + j, 3, All_nodes)

                if count > 20:
                    # cannot pick nodes successfully
                    # print('Cannot do this reaction')
                    Reaction_record[-1] = f"{R} (but failed)"

        # 4rd reaction
        # X1 + X2 <- X3
        if R == 4:
            if not All_nodes[2]:
                print("No node in deg3")
            else:
                [deg3_node1] = rd.sample(All_nodes[2], 1)
                [deg3_node2, deg3_node3] = [
                    m
                    for m in All_nodes[2]
                    if m.name == deg3_node1.name
                    if m != deg3_node1
                ]
                count = 0
                flag = 0

                j += 1
                deg3_node1.move(N + j, 1, All_nodes)
                j += 1
                deg3_node2.move(N + j, 2, All_nodes)
                deg3_node3.move(N + j, 2, All_nodes)

        if i == iterate - 1:
            get_graph_planar(All_nodes, plot_tend)
            all_info = get_graph_spring(All_nodes, plot_tend)
            netinfo.iloc[i + 1, 1:7] = all_info
            gcc_info = get_largest_CC(All_nodes, "F")
            netinfo.iloc[i + 1, 7:19] = gcc_info
        else:
            all_info = get_graph_spring(All_nodes, "F")
            netinfo.iloc[i + 1, 1:7] = all_info
            gcc_info = get_largest_CC(All_nodes, "F")
            netinfo.iloc[i + 1, 7:19] = gcc_info

    netinfo["Reaction"] = Reaction_record + ["NA"]
    # print(f'Take {time.time() - start_time} second')
    return netinfo


if __name__ == "__main__":
    print("Network Model Upload")
